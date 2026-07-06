#!/usr/bin/env python3
"""
POPF STABL V3 - ENHANCED WITH PREPROCESSING & CORRELATION GROUPING
===================================================================
Enhanced version building on V2 with:
- Integrated preprocessing pipeline (VarianceThreshold, LowInfoFilter, SimpleImputer)
- Correlation-based feature grouping via perc_corr_group_threshold
- Knockoff artificial features for better FDR control with correlated features
- All V2 features preserved (BCa CI, optimization, etc.)

Key Improvements over V2:
- Better handling of missing values and low-variance features
- Reduced feature redundancy through correlation grouping
- More stable feature selection with cleaner input data
- Flexible preprocessing configuration
"""

# ASCII Art Banner
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                                  RAD-PANC                                    ║
║      CR-POPF prediction using radiomics of the head of the pancreas          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import gc
import time
import json
import pickle
try:
    import psutil  # optional, used for system info when available
except Exception:
    psutil = None
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
from scipy import stats
from scipy.stats import bootstrap, gaussian_kde
from scipy.optimize import minimize_scalar

# Sklearn imports
from sklearn.model_selection import (
    RepeatedStratifiedKFold, 
    LeaveOneOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    brier_score_loss, log_loss
)
try:
    from sklearn.calibration import calibration_curve
except ImportError:
    calibration_curve = None
from sklearn.base import clone
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC

# NEW: Preprocessing imports
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

# Additional models
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    HAS_ADVANCED_MODELS = True
except ImportError:
    HAS_ADVANCED_MODELS = False
    print("Warning: XGBoost/LightGBM not installed. Using basic models only.")

# Statsmodels for advanced CI methods
try:
    from statsmodels.discrete.discrete_model import Logit
    from statsmodels.tools import add_constant
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Some CI methods unavailable.")

# STABL imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'dependencies' / 'Stabl'))
from stabl.stabl import Stabl
from stabl.preprocessing import LowInfoFilter  # NEW: STABL preprocessing

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# BeautifulFigures-style plotting utilities (optional)
try:
    # Local utils for consistent styling and multi-format export
    from utils.plotting_utils import (
        create_beautiful_figure,
        save_beautiful_figure,
        setup_plotting,
        NORD_COLORS,
        COLOR_SCHEMES,
    )
    setup_plotting()
    HAS_BEAUTIFUL = True
except Exception:
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_theme(style='darkgrid')
    HAS_BEAUTIFUL = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logger
logger = logging.getLogger('POPF_STABL_V3')


# --- ID normalization helpers (for robust alignment) ---
HONORIFICS = {
    "mr", "mrs", "ms", "mme", "mlle", "mle", "dr", "prof", "monsieur", "madame",
    "m", "mme.", "mlle.", "mr.", "mrs.", "dr.", "prof."
}

def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _canonicalize_id(s: Any) -> str:
    s = "" if s is None else str(s)
    s = _strip_accents(s).lower()
    s = s.replace("-", "_").replace(" ", "_")
    import re
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    tokens = [t for t in s.split("_") if t and t not in HONORIFICS]
    s = "_".join(tokens)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


class PreprocessingPipeline:
    """
    NEW: Configurable preprocessing pipeline for radiomics features.
    Applies variance filtering, NaN filtering, imputation, and scaling.
    """
    
    def __init__(self,
                 use_variance_filter: bool = True,
                 variance_threshold: float = 0.01,
                 use_low_info_filter: bool = True,
                 max_nan_fraction: float = 0.2,
                 impute_strategy: str = "median",
                 use_scaler: bool = True):
        """
        Parameters
        ----------
        use_variance_filter : bool
            Whether to apply variance threshold filtering
        variance_threshold : float
            Minimum variance to keep feature (default 0.01)
        use_low_info_filter : bool
            Whether to apply STABL's LowInfoFilter
        max_nan_fraction : float
            Maximum fraction of NaN values allowed (default 0.2)
        impute_strategy : str
            Strategy for imputation: 'median', 'mean', 'most_frequent'
        use_scaler : bool
            Whether to apply StandardScaler
        """
        self.use_variance_filter = use_variance_filter
        self.variance_threshold = variance_threshold
        self.use_low_info_filter = use_low_info_filter
        self.max_nan_fraction = max_nan_fraction
        self.impute_strategy = impute_strategy
        self.use_scaler = use_scaler
        self.pipeline = None
        self.feature_names_out_ = None
        
    def create_pipeline(self):
        """Create the preprocessing pipeline based on configuration."""
        steps = []
        
        # 1. Low info filter (removes features with too many NaNs)
        if self.use_low_info_filter:
            steps.append(("low_info", LowInfoFilter(max_nan_fraction=self.max_nan_fraction)))
        
        # 2. Imputation (for remaining NaNs)
        steps.append(("impute", SimpleImputer(strategy=self.impute_strategy)))
        
        # 3. Variance threshold (removes near-zero variance features)
        if self.use_variance_filter:
            steps.append(("variance", VarianceThreshold(threshold=self.variance_threshold)))
        
        # 4. Scaling
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        
        self.pipeline = Pipeline(steps)
        return self.pipeline
    
    def fit_transform(self, X, feature_names=None):
        """
        Fit and transform the data through the pipeline.
        
        Returns
        -------
        X_transformed : array-like
            Transformed feature matrix
        feature_names_out : list
            Names of features after filtering
        """
        if self.pipeline is None:
            self.create_pipeline()
        
        logger.info("Applying preprocessing pipeline...")
        logger.info(f"  Input shape: {X.shape}")
        
        # Apply pipeline
        X_transformed = self.pipeline.fit_transform(X)
        
        # Track which features survived
        if feature_names is not None:
            feature_names = np.array(feature_names)
            # Get masks from each step
            surviving_mask = np.ones(len(feature_names), dtype=bool)
            
            for name, step in self.pipeline.steps:
                if hasattr(step, 'get_support'):
                    # This step performs feature selection
                    current_mask = step.get_support()
                    # Update the surviving features
                    surviving_indices = np.where(surviving_mask)[0]
                    surviving_mask[surviving_indices] = current_mask
            
            self.feature_names_out_ = feature_names[surviving_mask].tolist()
            logger.info(f"  Output shape: {X_transformed.shape}")
            logger.info(f"  Features removed: {X.shape[1] - X_transformed.shape[1]}")
        else:
            self.feature_names_out_ = None
            
        return X_transformed, self.feature_names_out_

    def transform(self, X):
        """Transform data using the fitted pipeline without refitting."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessing pipeline not fitted. Call fit_transform first.")
        return self.pipeline.transform(X)


class FirthLogisticRegression:
    """
    Firth's penalized logistic regression for small samples with rare events.
    Properly implemented to reduce small sample bias without over-regularization.
    """
    
    def __init__(self, max_iter=25, tol=1e-4, class_weight='balanced'):
        # NO alpha parameter - Firth's method already provides the right amount of regularization
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.coef_ = None
        self.intercept_ = None
        self.n_features_ = None
        self.classes_ = np.array([0, 1])
        self.n_iter_ = 0
        
    def fit(self, X, y):
        """Fit Firth's penalized logistic regression with proper implementation."""
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Handle class weights
        if self.class_weight == 'balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight('balanced', y)
        else:
            sample_weight = np.ones(n_samples)
        
        # CRITICAL: Normalize weights to sum to n_samples
        # This prevents over-weighting and maintains proper scale
        sample_weight = sample_weight * n_samples / np.sum(sample_weight)
        
        # Add intercept
        X_with_intercept = np.c_[np.ones(n_samples), X]
        
        # Smart initialization: use log-odds for intercept instead of zero
        # This dramatically improves convergence for imbalanced data
        p_init = np.average(y, weights=sample_weight)
        if p_init > 0 and p_init < 1:
            intercept_init = np.log(p_init / (1 - p_init))
        else:
            intercept_init = 0
        beta = np.zeros(n_features + 1)
        beta[0] = intercept_init
        
        # Newton-Raphson with proper Firth correction
        for iteration in range(self.max_iter):
            # Linear predictor with numerical stability
            eta = X_with_intercept @ beta
            eta = np.clip(eta, -500, 500)
            
            # Probabilities with bounds to prevent exact 0 or 1
            p = 1 / (1 + np.exp(-eta))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            
            # Weighted Fisher information matrix
            W = np.diag(sample_weight * p * (1 - p))
            fisher_info = X_with_intercept.T @ W @ X_with_intercept
            
            # Add tiny ridge only for numerical stability (not regularization)
            fisher_info += np.eye(n_features + 1) * 1e-10
            
            try:
                fisher_inv = np.linalg.inv(fisher_info)
            except np.linalg.LinAlgError:
                # If still singular, add slightly more (but still tiny)
                fisher_info += np.eye(n_features + 1) * 1e-6
                fisher_inv = np.linalg.inv(fisher_info)
            
            # Efficient computation of hat matrix diagonal for Firth correction
            # h_i = (X * Fisher^{-1} * X^T)_{ii}
            XF = X_with_intercept @ fisher_inv
            h = np.sum(XF * X_with_intercept, axis=1)
            
            # Score with proper Firth correction
            # Both the standard score and Firth term use the same weights
            # This is the correct formulation!
            score = X_with_intercept.T @ (sample_weight * (y - p + h * (0.5 - p)))
            
            # Update parameters
            delta = fisher_inv @ score
            beta_new = beta + delta
            
            # Check convergence based on parameter change
            if np.max(np.abs(delta)) < self.tol:
                self.n_iter_ = iteration + 1
                break
                
            beta = beta_new
        
        self.intercept_ = beta[0]
        self.coef_ = beta[1:].reshape(1, -1)
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities."""
        n_samples = X.shape[0]
        X_with_intercept = np.c_[np.ones(n_samples), X]
        beta = np.concatenate([[self.intercept_], self.coef_.ravel()])
        eta = X_with_intercept @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -500, 500)))
        return np.c_[1 - p, p]
    
    def predict(self, X):
        """Predict classes."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {
            'max_iter': self.max_iter,
            'tol': self.tol,
            'class_weight': self.class_weight
        }
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class OptimizedModelSelector:
    """
    Performs hyperparameter optimization for final model.
    Legitimate approach for publication - all within CV.
    """
    
    @staticmethod
    def optimize_logistic_regression(X, y, cv_folds=5, random_state=42, fixed_c=None):
        """
        GridSearchCV for Logistic Regression hyperparameters, or use fixed C value.
        
        Returns optimized model with best parameters.
        """
        if fixed_c is not None:
            # Use fixed C value from Optuna optimization
            model = LogisticRegression(
                C=fixed_c,
                penalty='l1',  # Use L1 as it was best in Optuna
                solver='liblinear',
                class_weight='balanced',  # Use balanced for imbalanced data
                max_iter=5000,
                random_state=random_state
            )
            # Return in same format as GridSearchCV
            return model, {'C': fixed_c, 'penalty': 'l1', 'class_weight': 'balanced', 'solver': 'liblinear'}, None
        
        # Expanded parameter grid with broader C range
        param_grid = {
            'C': [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 250, 500, 1000],
            'penalty': ['l1', 'l2'],
            'class_weight': ['balanced', None],
            'solver': ['liblinear']  # Works with both L1 and L2
        }
        
        base_model = LogisticRegression(max_iter=5000, random_state=random_state)
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_
    
    @staticmethod
    def optimize_xgboost(X, y, cv_folds=5, random_state=42):
        """Optimize XGBoost hyperparameters."""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        base_model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=random_state
        )
        
        # Use fewer parameter combinations for speed
        param_grid_reduced = {
            'n_estimators': [100],
            'max_depth': [3, 5],
            'learning_rate': [0.1],
            'subsample': [0.8]
        }
        
        grid_search = GridSearchCV(
            base_model,
            param_grid_reduced,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X, y)
        
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_

    @staticmethod
    def optimize_elasticnet_logistic_regression(X, y, cv_folds=5, random_state=42,
                                                c_grid=None, l1_grid=None, class_weight_options=None):
        """GridSearchCV for Elastic-Net Logistic Regression (penalty='elasticnet', solver='saga')."""
        c_grid = c_grid or [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
        l1_grid = l1_grid or [0.3, 0.5, 0.7]
        class_weight_options = class_weight_options or ['balanced', None]

        base_model = LogisticRegression(max_iter=5000, solver='saga', penalty='elasticnet', random_state=random_state)
        param_grid = {
            'C': c_grid,
            'l1_ratio': l1_grid,
            'class_weight': class_weight_options
        }

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


class BCaConfidenceInterval:
    """
    Bias-Corrected and accelerated (BCa) bootstrap confidence intervals.
    More accurate than percentile method for small samples.
    Now includes smoothed bootstrap option for tighter CIs.
    """
    
    @staticmethod
    def calculate_bca_ci(scores: np.ndarray, 
                        original_estimate: float,
                        confidence_level: float = 0.95,
                        n_bootstrap: int = 2000,
                        smooth: bool = False,
                        bandwidth: str = 'scott') -> Tuple[float, float]:
        """
        Calculate BCa confidence interval with optional smoothing.
        
        Parameters
        ----------
        smooth : bool
            If True, use kernel density estimation to smooth bootstrap distribution
        bandwidth : str
            Bandwidth method for KDE ('scott' or 'silverman')
        """
        scores = np.array(scores)
        n = len(scores)
        
        # Apply smoothing if requested
        if smooth and len(scores) > 10:
            try:
                # Fit kernel density estimate
                kde = gaussian_kde(scores, bw_method=bandwidth)
                # Generate smoothed samples
                n_smooth = max(10000, 10 * len(scores))
                smoothed_scores = kde.resample(n_smooth).ravel()
                # Use smoothed scores for CI calculation
                scores_for_ci = smoothed_scores
            except Exception:
                # Fallback to original scores if smoothing fails
                scores_for_ci = scores
        else:
            scores_for_ci = scores
        
        # Calculate bias correction factor (z0)
        z0 = stats.norm.ppf(np.mean(scores_for_ci < original_estimate))
        
        # Simplified acceleration factor
        a = 0  # Simplified - assumes no acceleration
        
        # Calculate adjusted percentiles
        alpha = 1 - confidence_level
        z_alpha_lower = stats.norm.ppf(alpha / 2)
        z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
        
        # Adjusted percentiles
        a1 = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        a2 = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        # Get BCa interval
        ci_lower = np.percentile(scores_for_ci, 100 * a1)
        ci_upper = np.percentile(scores_for_ci, 100 * a2)
        
        return ci_lower, ci_upper


class MultiMethodValidator:
    """
    Implements multiple validation methods for comprehensive evaluation.
    """
    
    def __init__(self, n_bootstrap: int = 1000, optimize_model: bool = True, fixed_c: float = None,
                 use_firth: bool = False, smooth_ci: bool = False, use_ensemble: bool = False,
                 use_nested_selection: bool = False,
                 run_nested_mc: bool = False, nested_outer: int = 50, nested_test_size: float = 0.2,
                 nested_inner_cv: int = 5, nested_random_state: int = 42, stabl_params: Dict = None,
                 cv_splits: int = 4, cv_repeats: int = 20,
                 # NEW: train-only LR tuning inside CV via Optuna
                 cv_tune_lr: bool = False,
                 cv_tune_trials: int = 30,
                 cv_tune_inner_splits: int = 3,
                 cv_tune_seed: int = 42,
                 lr_c_min: float = 1e-3,
                 lr_c_max: float = 1e2,
                 en_l1_min: float = 0.3,
                 en_l1_max: float = 0.7,
                 nested_stabl_bags: int = 1,
                 nested_add_ridge: bool = False,
                 nested_add_firth: bool = False,
                 nested_add_xgb: bool = False):
        self.n_bootstrap = n_bootstrap
        self.optimize_model = optimize_model
        self.fixed_c = fixed_c
        self.use_firth = use_firth
        self.smooth_ci = smooth_ci
        self.use_ensemble = use_ensemble
        self.use_nested_selection = use_nested_selection
        # Nested Monte Carlo configuration
        self.run_nested_mc = run_nested_mc
        self.nested_outer = nested_outer
        self.nested_test_size = nested_test_size
        self.nested_inner_cv = nested_inner_cv
        self.nested_random_state = nested_random_state
        self.stabl_params = stabl_params or {}
        # Repeated-CV configuration
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        # Tuning config
        self.cv_tune_lr = cv_tune_lr
        self.cv_tune_trials = cv_tune_trials
        self.cv_tune_inner_splits = cv_tune_inner_splits
        self.cv_tune_seed = cv_tune_seed
        self.lr_c_min = lr_c_min
        self.lr_c_max = lr_c_max
        self.en_l1_min = en_l1_min
        self.en_l1_max = en_l1_max
        self.nested_stabl_bags = max(1, int(nested_stabl_bags))
        self.nested_add_ridge = nested_add_ridge
        self.nested_add_firth = nested_add_firth
        self.nested_add_xgb = nested_add_xgb and HAS_ADVANCED_MODELS
        
    def evaluate_all_methods(self, model, X, y, feature_names=None, methods=None):
        """Evaluate model using selected validation methods.
        methods: list of method keys among
            ['bootstrap632+', 'repeated-cv', 'loocv', 'simple-bootstrap', 'nested-mc']
        """
        results = {}
        if methods is None:
            methods = ['bootstrap632+', 'repeated-cv', 'loocv', 'simple-bootstrap']
        
        # Optimize model if requested
        if self.optimize_model and isinstance(model, LogisticRegression):
            print("\n[0/4] Optimizing model hyperparameters...")
            # Detect Elastic-Net vs classic LR
            mparams = model.get_params(deep=True)
            if mparams.get('penalty') == 'elasticnet' or mparams.get('solver') == 'saga':
                # Elastic-Net LR optimization
                optimized_model, best_params, cv_results = OptimizedModelSelector.optimize_elasticnet_logistic_regression(
                    X, y
                )
            else:
                optimized_model, best_params, cv_results = OptimizedModelSelector.optimize_logistic_regression(
                    X, y, fixed_c=self.fixed_c
                )
            results['optimization'] = {
                'best_params': best_params,
                'cv_results': cv_results
            }
            model = optimized_model
            if self.fixed_c:
                print(f"   Using fixed C value: {self.fixed_c}")
            print(f"   Best parameters: {best_params}")
        
        # 1. Bootstrap .632+
        if 'bootstrap632+' in methods:
            print("\n[1/?] Bootstrap .632+ (Conservative)")
            results['bootstrap_632_plus'] = self.bootstrap_632_plus_evaluation(
                model, X, y, feature_names=feature_names, n_bootstrap=self.n_bootstrap
            )
        
        # 2. Repeated Stratified KFold
        if 'repeated-cv' in methods:
            print("[?/ ?] Repeated Stratified KFold (5×20)")
            results['repeated_stratified_cv'] = self.repeated_stratified_cv_evaluation(
                model, X, y, feature_names=feature_names
            )
        
        # 3. Leave-One-Out CV
        if 'loocv' in methods:
            print("[?/ ?] Leave-One-Out CV (Deterministic)")
            results['loocv'] = self.loocv_evaluation(model, X, y, feature_names=feature_names)
        
        # 4. Simple Bootstrap (no correction)
        if 'simple-bootstrap' in methods:
            print("[?/ ?] Simple Bootstrap (No Correction)")
            results['simple_bootstrap'] = self.simple_bootstrap_evaluation(
                model, X, y, feature_names=feature_names, n_bootstrap=self.n_bootstrap // 2
            )

        # Optional: Nested Monte Carlo (outer holdout, inner selection)
        if ('nested-mc' in methods) or self.run_nested_mc:
            print("\n[EXTRA] Nested Monte Carlo evaluation (unbiased outer holdout)")
            try:
                results['nested_monte_carlo'] = self.nested_monte_carlo_evaluation(
                    model, X, y, feature_names=feature_names,
                    n_outer=self.nested_outer,
                    test_size=self.nested_test_size,
                    inner_cv=self.nested_inner_cv,
                    random_state=self.nested_random_state
                )
            except Exception as e:
                results['nested_monte_carlo'] = {'error': str(e)}
        
        # Calculate consensus
        results['consensus'] = self.calculate_consensus(results)
        
        return results

    def nested_monte_carlo_evaluation(self, base_model, X, y, feature_names=None,
                                       n_outer: int = 50, test_size: float = 0.2,
                                       inner_cv: int = 5, random_state: int = 42):
        """
        Nested Monte Carlo evaluation for unbiased AUROC.
        - Outer loop: StratifiedShuffleSplit into train/test.
        - Inner: feature selection via EnhancedParallelEnsembleSTABL on training only and
                 optional hyperparameter optimization via inner CV.
        """
        if feature_names is None:
            feature_names = np.array([f"f{i}" for i in range(X.shape[1])])

        splitter = StratifiedShuffleSplit(n_splits=n_outer, test_size=test_size, random_state=random_state)
        aucs = []
        kept = 0
        feature_freq: Dict[str, int] = {}

        for i, (tr, te) in enumerate(splitter.split(X, y), 1):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue

            # Train-only feature selection
            stabl_cfg = {
                'n_runs': self.stabl_params.get('n_runs', 5),
                'n_bootstraps': self.stabl_params.get('n_bootstraps', 200),
                'consensus_threshold': self.stabl_params.get('consensus_threshold', 0.5),
                'n_workers': self.stabl_params.get('n_workers', 1),
                'artificial_type': self.stabl_params.get('artificial_type', 'knockoff'),
                'perc_corr_group_threshold': self.stabl_params.get('perc_corr_group_threshold', None),
                'use_preprocessing': self.stabl_params.get('use_preprocessing', True),
                'preprocessing_config': self.stabl_params.get('preprocessing_config', {}),
                'c_value': self.stabl_params.get('c_value', None),
                'random_state': random_state + i
            }

            selector = EnhancedParallelEnsembleSTABL(**stabl_cfg)
            selector.fit(X_tr, y_tr, feature_names=np.array(feature_names))

            sel_feats = selector.selected_features_ or []
            if not sel_feats:
                continue
            for f in sel_feats:
                feature_freq[f] = feature_freq.get(f, 0) + 1

            # Transform data consistently and select columns
            if selector.use_preprocessing:
                X_tr_pre = selector.preprocessor_.transform(X_tr)
                X_te_pre = selector.preprocessor_.transform(X_te)
                pre_names = selector.preprocessed_feature_names_
                idx = [i for i, f in enumerate(pre_names) if f in sel_feats]
                if not idx:
                    continue
                X_tr_sel = X_tr_pre[:, idx]
                X_te_sel = X_te_pre[:, idx]
            else:
                idx = [i for i, f in enumerate(feature_names) if f in sel_feats]
                if not idx:
                    continue
                X_tr_sel = X_tr[:, idx]
                X_te_sel = X_te[:, idx]

            # Inner model optimization on training set only (if enabled)
            if self.optimize_model and isinstance(base_model, LogisticRegression):
                model, _, _ = OptimizedModelSelector.optimize_logistic_regression(
                    X_tr_sel, y_tr, cv_folds=inner_cv, fixed_c=self.fixed_c
                )
            else:
                model = clone(base_model)
                model.fit(X_tr_sel, y_tr)

            # Predict and score on outer test
            try:
                y_prob = model.predict_proba(X_te_sel)[:, 1]
            except AttributeError:
                y_prob = model.decision_function(X_te_sel)

            if len(np.unique(y_te)) == 2:
                aucs.append(roc_auc_score(y_te, y_prob))
                kept += 1

        if not aucs:
            return None

        aucs = np.array(aucs)
        ci = [np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)]
        return {
            'auc': float(np.mean(aucs)),
            'std': float(np.std(aucs)),
            'ci': [float(ci[0]), float(ci[1])],
            'n_outer': kept,
            'scores': aucs.tolist(),
            'feature_frequency': feature_freq
        }
    
    def bootstrap_632_plus_evaluation(self, model, X, y, feature_names=None, n_bootstrap=1000):
        """Bootstrap .632+ with correct error-based weighting and CIs.

        Implements Efron & Tibshirani .632+ for error metrics, adapted to AUC by
        using err = 1 - AUC and the no-information error err0 = 0.5 (random AUC = 0.5).
        """
        
        n_samples = len(y)
        train_aucs = []
        oob_aucs = []
        auc632p_list = []
        skipped = 0
        
        for i in tqdm(range(n_bootstrap), desc="Bootstrap .632+"):
            # Stratified bootstrap
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            pos_boot_idx = resample(pos_idx, replace=True, n_samples=len(pos_idx))
            neg_boot_idx = resample(neg_idx, replace=True, n_samples=len(neg_idx))
            
            idx = np.concatenate([pos_boot_idx, neg_boot_idx])
            oob_idx = list(set(range(n_samples)) - set(idx))
            
            if len(oob_idx) > 5 and len(np.unique(y[oob_idx])) == 2:
                try:
                    if self.use_nested_selection and feature_names is not None:
                        # Train-only feature selection
                        selector = EnhancedParallelEnsembleSTABL(**self.stabl_params)
                        selector.fit(X[idx], y[idx], feature_names=np.array(feature_names))
                        sel_feats = selector.selected_features_ or []
                        if not sel_feats:
                            skipped += 1
                            continue
                        # Transform with fitted preprocessor
                        X_train_pre = selector.preprocessor_.transform(X[idx])
                        X_oob_pre = selector.preprocessor_.transform(X[oob_idx])
                        pre_names = selector.preprocessed_feature_names_
                        sel_idx = [j for j, f in enumerate(pre_names) if f in sel_feats]
                        if not sel_idx:
                            skipped += 1
                            continue
                        X_train_sel = X_train_pre[:, sel_idx]
                        X_oob_sel = X_oob_pre[:, sel_idx]
                    else:
                        # Fallback: scale only (legacy behavior)
                        if not hasattr(self, '_preprocessing_applied'):
                            scaler = StandardScaler()
                            X_train_sel = scaler.fit_transform(X[idx])
                            X_oob_sel = scaler.transform(X[oob_idx])
                        else:
                            X_train_sel = X[idx]
                            X_oob_sel = X[oob_idx]

                    # Train and evaluate
                    model_boot = clone(model)
                    model_boot.fit(X_train_sel, y[idx])
                    
                    # Get predictions
                    y_train_pred = model_boot.predict_proba(X_train_sel)[:, 1]
                    y_oob_pred = model_boot.predict_proba(X_oob_sel)[:, 1]
                    
                    train_auc = roc_auc_score(y[idx], y_train_pred)
                    oob_auc = roc_auc_score(y[oob_idx], y_oob_pred)

                    # Store per-bootstrap AUCs
                    train_aucs.append(train_auc)
                    oob_aucs.append(oob_auc)
                
                except Exception:
                    skipped += 1
            else:
                skipped += 1
        
        # Calculate .632+ estimate and CIs
        if oob_aucs:
            train_aucs = np.array(train_aucs, dtype=float)
            oob_aucs = np.array(oob_aucs, dtype=float)
            err0 = 0.5  # no-information error for AUC (1 - 0.5)
            eps = 1e-8

            # Per-bootstrap .632+ using error formulation
            err_tr = 1.0 - train_aucs
            err_oob = 1.0 - oob_aucs
            denom = np.maximum(err0 - err_tr, eps)
            R = np.clip((err_oob - err_tr) / denom, 0.0, 1.0)
            w = 0.632 / (1.0 - 0.368 * R)
            err_632p = (1.0 - w) * err_tr + w * err_oob
            auc632p_list = (1.0 - err_632p).tolist()

            auc_632_plus = float(np.mean(auc632p_list))

            # CIs based on distribution of the .632+ estimator
            ci_methods = {}
            # 1. Percentile
            ci_methods['percentile'] = [
                float(np.percentile(auc632p_list, 2.5)),
                float(np.percentile(auc632p_list, 97.5)),
            ]
            # 2. BCa (optionally smoothed)
            bca = BCaConfidenceInterval()
            ci_methods['bca'] = bca.calculate_bca_ci(
                np.array(auc632p_list, dtype=float), auc_632_plus,
                smooth=self.smooth_ci
            )
            # 3. Normal approx
            se = float(np.std(auc632p_list))
            ci_methods['normal'] = [
                max(0.0, auc_632_plus - 1.96 * se),
                min(1.0, auc_632_plus + 1.96 * se),
            ]
            
            return {
                'auc': auc_632_plus,
                'ci_methods': ci_methods,
                'ci': ci_methods['bca'],  # Primary CI
                'n_bootstrap': len(auc632p_list),
                'skipped': skipped,
                'per_bootstrap_auc': auc632p_list,
            }
        
        return None
    
    def repeated_stratified_cv_evaluation(self, model, X, y, feature_names=None, n_splits=None, n_repeats=None):
        """Repeated Stratified KFold evaluation.
        If use_nested_selection is True, performs train-only STABL feature selection per split.
        """
        if n_splits is None:
            n_splits = self.cv_splits
        if n_repeats is None:
            n_repeats = self.cv_repeats
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
        # Always compute train-only preprocessing per split and collect OOF predictions
        # (more robust and enables pooled ROC visualization)
        if not self.use_nested_selection:
            # Train-only impute + scale per split on the frozen panel
            oof_pred = np.zeros(len(y), dtype=float)
            oof_mask = np.zeros(len(y), dtype=bool)
            fold_scores_list = []
            fold_tuning = []
            fold_tuning = []
            for tr, te in cv.split(X, y):
                X_tr, X_te = X[tr], X[te]
                y_tr, y_te = y[tr], y[te]
                # train-only preprocessing
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                X_tr_i = imputer.fit_transform(X_tr)
                X_te_i = imputer.transform(X_te)
                X_tr_s = scaler.fit_transform(X_tr_i)
                X_te_s = scaler.transform(X_te_i)
                m = clone(model)
                # Optional train-only tuning via Optuna
                tuned_info = None
                try:
                    if self.cv_tune_lr and isinstance(m, LogisticRegression):
                        m, tuned_info = self._optuna_tune_lr(X_tr_s, y_tr, m)
                except Exception:
                    tuned_info = None
                m.fit(X_tr_s, y_tr)
                try:
                    p = m.predict_proba(X_te_s)[:, 1]
                except Exception:
                    s = m.decision_function(X_te_s)
                    p = 1 / (1 + np.exp(-s))
                if len(np.unique(y_te)) == 2:
                    fold_scores_list.append(roc_auc_score(y_te, p))
                oof_pred[te] = p
                oof_mask[te] = True
                if tuned_info is not None:
                    fold_tuning.append(tuned_info)
            scores = np.array(fold_scores_list, dtype=float)
            # OOF metrics
            oof_auc = float(roc_auc_score(y[oof_mask], oof_pred[oof_mask])) if len(np.unique(y[oof_mask])) == 2 else None
            # Stratified bootstrap CI on OOF
            oof_ci = None
            if oof_auc is not None:
                rng = np.random.RandomState(42)
                idx_all = np.arange(np.sum(oof_mask))
                y_o = y[oof_mask]
                p_o = oof_pred[oof_mask]
                scores_bs = []
                # class-balanced resampling
                pos_idx = np.where(y_o == 1)[0]
                neg_idx = np.where(y_o == 0)[0]
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    for _ in range(2000):
                        bs_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=rng)
                        bs_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=rng)
                        bs = np.concatenate([bs_pos, bs_neg])
                        y_b = y_o[bs]
                        if len(np.unique(y_b)) < 2:
                            continue
                        scores_bs.append(roc_auc_score(y_b, p_o[bs]))
                    if scores_bs:
                        oof_ci = [float(np.percentile(scores_bs, 2.5)), float(np.percentile(scores_bs, 97.5))]
        else:
            # Nested selection per split
            scores_list = []
            oof_pred = np.zeros(len(y), dtype=float)
            oof_mask = np.zeros(len(y), dtype=bool)
            fold_tuning = []
            for fold_idx, (tr, te) in enumerate(cv.split(X, y), 1):
                bag_predictions = []
                base_seed = self.stabl_params.get('random_state', 42)
                for bag in range(self.nested_stabl_bags):
                    stabl_cfg = dict(self.stabl_params)
                    stabl_cfg['random_state'] = base_seed + fold_idx * 1000 + bag
                    selector = EnhancedParallelEnsembleSTABL(**stabl_cfg)
                    selector.fit(X[tr], y[tr], feature_names=np.array(feature_names))
                    sel_feats = selector.selected_features_ or []
                    if not sel_feats:
                        continue
                    if selector.use_preprocessing:
                        X_tr_pre = selector.preprocessor_.transform(X[tr])
                        X_te_pre = selector.preprocessor_.transform(X[te])
                        pre_names = selector.preprocessed_feature_names_
                        idx = [i for i, f in enumerate(pre_names) if f in sel_feats]
                        if not idx:
                            continue
                        X_tr_sel = X_tr_pre[:, idx]
                        X_te_sel = X_te_pre[:, idx]
                    else:
                        idx = [i for i, f in enumerate(feature_names) if f in sel_feats]
                        if not idx:
                            continue
                        X_tr_sel = X[tr][:, idx]
                        X_te_sel = X[te][:, idx]

                    imputer = SimpleImputer(strategy='median')
                    X_tr_imp = imputer.fit_transform(X_tr_sel)
                    X_te_imp = imputer.transform(X_te_sel)
                    scaler = StandardScaler()
                    X_tr_s = scaler.fit_transform(X_tr_imp)
                    X_te_s = scaler.transform(X_te_imp)

                    component_probs: List[np.ndarray] = []
                    base_model = clone(model)
                    tuned_info = None
                    if self.cv_tune_lr and isinstance(base_model, LogisticRegression):
                        try:
                            base_model, tuned_info = self._optuna_tune_lr(X_tr_s, y[tr], base_model)
                        except Exception:
                            tuned_info = None
                    try:
                        base_model.fit(X_tr_s, y[tr])
                        logistic_prob = base_model.predict_proba(X_te_s)[:, 1]
                    except AttributeError:
                        base_model.fit(X_tr_s, y[tr])
                        decision = base_model.decision_function(X_te_s)
                        logistic_prob = 1.0 / (1.0 + np.exp(-decision))
                    component_probs.append(logistic_prob)
                    if tuned_info is not None:
                        tuned_info = dict(tuned_info)
                        tuned_info.update({'fold': fold_idx, 'bag': bag})
                        fold_tuning.append(tuned_info)

                    if self.nested_add_ridge:
                        ridge_prob = self._ridge_nested_prediction(X_tr_s, y[tr], X_te_s)
                        if ridge_prob is not None:
                            component_probs.append(ridge_prob)

                    if self.nested_add_firth:
                        try:
                            firth = FirthLogisticRegression(class_weight='balanced')
                            firth.fit(X_tr_s, y[tr])
                            component_probs.append(firth.predict_proba(X_te_s)[:, 1])
                        except Exception:
                            pass

                    if self.nested_add_xgb:
                        try:
                            sample_weight = compute_sample_weight('balanced', y[tr])
                            xgb_clf = XGBClassifier(
                                n_estimators=200,
                                max_depth=2,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_lambda=1.0,
                                reg_alpha=0.0,
                                random_state=stabl_cfg['random_state'],
                                eval_metric='logloss',
                                n_jobs=1,
                            )
                            xgb_clf.fit(X_tr_imp, y[tr], sample_weight=sample_weight)
                            component_probs.append(xgb_clf.predict_proba(X_te_imp)[:, 1])
                        except Exception:
                            pass

                    if not component_probs:
                        continue

                    bag_pred = np.mean(np.column_stack(component_probs), axis=1)
                    bag_predictions.append(bag_pred)

                if not bag_predictions:
                    continue

                y_prob = np.mean(np.column_stack(bag_predictions), axis=1)
                if len(np.unique(y[te])) == 2:
                    scores_list.append(roc_auc_score(y[te], y_prob))
                oof_pred[te] = y_prob
                oof_mask[te] = True
            scores = np.array(scores_list) if scores_list else np.array([])
            oof_auc = float(roc_auc_score(y[oof_mask], oof_pred[oof_mask])) if (oof_mask.any() and len(np.unique(y[oof_mask])) == 2) else None
            oof_ci = None
            if oof_auc is not None:
                rng = np.random.RandomState(42)
                y_o = y[oof_mask]
                p_o = oof_pred[oof_mask]
                pos_idx = np.where(y_o == 1)[0]
                neg_idx = np.where(y_o == 0)[0]
                scores_bs = []
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    for _ in range(2000):
                        bs_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=rng)
                        bs_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=rng)
                        bs = np.concatenate([bs_pos, bs_neg])
                        y_b = y_o[bs]
                        if len(np.unique(y_b)) < 2:
                            continue
                        scores_bs.append(roc_auc_score(y_b, p_o[bs]))
                    if scores_bs:
                        oof_ci = [float(np.percentile(scores_bs, 2.5)), float(np.percentile(scores_bs, 97.5))]

        if scores.size == 0:
            return None

        ci_methods = {
            'percentile': [np.percentile(scores, 2.5), np.percentile(scores, 97.5)],
            'normal': [
                np.mean(scores) - 1.96 * np.std(scores) / np.sqrt(len(scores)),
                np.mean(scores) + 1.96 * np.std(scores) / np.sqrt(len(scores))
            ]
        }

        return {
            'auc': float(np.mean(scores)),
            'auc_median': float(np.median(scores)),
            'std': float(np.std(scores)),
            'ci': ci_methods['percentile'],
            'ci_methods': ci_methods,
            'n_iterations': n_splits * n_repeats,
            'scores': scores.tolist(),
            'oof': {
                'y_true': y[oof_mask].astype(int).tolist() if 'oof_mask' in locals() else None,
                'y_pred': oof_pred[oof_mask].astype(float).tolist() if 'oof_mask' in locals() else None,
                'index': np.where(oof_mask)[0].astype(int).tolist() if 'oof_mask' in locals() else None,
                'auc': oof_auc,
                'ci': oof_ci,
            } if ('oof_mask' in locals()) else None,
            'tuning': fold_tuning if 'fold_tuning' in locals() and fold_tuning else None
        }

    def _optuna_tune_lr(self, X_tr, y_tr, base_model):
        """Tune LogisticRegression C (and l1_ratio if elasticnet) via Optuna on TRAIN only.
        Returns (best_model, info_dict).
        """
        try:
            import optuna  # type: ignore
        except Exception:
            # Fallback: simple grid on C
            grid = [0.1, 0.25, 0.5, 1.0, 2.5]
            best_c = grid[0]
            best_auc = -1.0
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=min(self.cv_tune_inner_splits, max(2, int(np.bincount(y_tr.astype(int)).min()))), shuffle=True, random_state=self.cv_tune_seed)
            for C in grid:
                m = clone(base_model)
                params = m.get_params(deep=True)
                if params.get('penalty') == 'elasticnet':
                    m.set_params(C=C, l1_ratio=float((self.en_l1_min + self.en_l1_max)/2.0))
                else:
                    m.set_params(C=C)
                scores = cross_val_score(m, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
                if len(scores) and float(np.mean(scores)) > best_auc:
                    best_auc = float(np.mean(scores)); best_c = C
            m = clone(base_model)
            params = m.get_params(deep=True)
            if params.get('penalty') == 'elasticnet':
                m.set_params(C=best_c, l1_ratio=float((self.en_l1_min + self.en_l1_max)/2.0))
                info = {'C': best_c, 'l1_ratio': float((self.en_l1_min + self.en_l1_max)/2.0), 'inner': 'grid'}
            else:
                m.set_params(C=best_c)
                info = {'C': best_c, 'inner': 'grid'}
            return m, info

        # Optuna path
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        penalty = base_model.get_params(deep=True).get('penalty')
        def objective(trial):
            C = trial.suggest_float('C', float(self.lr_c_min), float(self.lr_c_max), log=True)
            m = clone(base_model)
            if penalty == 'elasticnet':
                l1 = trial.suggest_float('l1_ratio', float(self.en_l1_min), float(self.en_l1_max))
                m.set_params(C=C, l1_ratio=l1)
            else:
                m.set_params(C=C)
            n_splits = min(self.cv_tune_inner_splits, max(2, int(np.bincount(y_tr.astype(int)).min())))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.cv_tune_seed)
            scores = cross_val_score(m, X_tr, y_tr, cv=skf, scoring='roc_auc', n_jobs=-1)
            return float(np.mean(scores)) if len(scores) else 0.5
        sampler = optuna.samplers.TPESampler(seed=self.cv_tune_seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=int(self.cv_tune_trials), n_jobs=1)
        best_params = study.best_params
        m_best = clone(base_model)
        if penalty == 'elasticnet':
            m_best.set_params(C=float(best_params.get('C', 1.0)), l1_ratio=float(best_params.get('l1_ratio', (self.en_l1_min + self.en_l1_max)/2.0)))
            info = {'C': float(best_params.get('C', 1.0)), 'l1_ratio': float(best_params.get('l1_ratio', (self.en_l1_min + self.en_l1_max)/2.0)), 'inner': 'optuna'}
        else:
            m_best.set_params(C=float(best_params.get('C', 1.0)))
            info = {'C': float(best_params.get('C', 1.0)), 'inner': 'optuna'}
        return m_best, info

    def _ridge_nested_prediction(self, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> Optional[np.ndarray]:
        """Fit a ridge (L2) logistic regression with a small C grid on train and predict test."""
        try:
            class_counts = np.bincount(y_tr.astype(int))
            min_count = int(class_counts.min()) if class_counts.size == 2 else 0
        except Exception:
            min_count = 0
        splits = max(2, min(5, min_count)) if min_count > 0 else 2
        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.cv_tune_seed)
        c_grid = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        best_auc = -np.inf
        best_model = None
        for C in c_grid:
            clf = LogisticRegression(penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=5000, C=C)
            try:
                scores = cross_val_score(clf, X_tr, y_tr, cv=skf, scoring='roc_auc')
                score = float(np.mean(scores)) if len(scores) else 0.5
            except Exception:
                score = 0.5
            if score > best_auc:
                best_auc = score
                best_model = clf
        if best_model is None:
            return None
        try:
            best_model.fit(X_tr, y_tr)
            return best_model.predict_proba(X_te)[:, 1]
        except Exception:
            return None
    
    def loocv_evaluation(self, model, X, y, feature_names=None):
        """Leave-One-Out Cross-Validation (deterministic).
        If use_nested_selection is True, performs selection for each LOOCV split (expensive).
        """
        
        loo = LeaveOneOut()
        predictions = []
        y_true = []
        
        if not self.use_nested_selection:
            # Note: X may already be scaled from preprocessing
            if not hasattr(self, '_preprocessing_applied'):
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X
            X_iter = X_scaled
        else:
            X_iter = X  # we'll explicitly transform per split using selector
        
        for train_idx, test_idx in tqdm(loo.split(X_iter), 
                                       total=len(y), 
                                       desc="LOOCV"):
            X_train, X_test = X_iter[train_idx], X_iter[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if self.use_nested_selection and feature_names is not None:
                # Fit selection on train only
                selector = EnhancedParallelEnsembleSTABL(**self.stabl_params)
                selector.fit(X_train, y_train, feature_names=np.array(feature_names))
                sel_feats = selector.selected_features_ or []
                if not sel_feats:
                    continue
                if selector.use_preprocessing:
                    X_train_pre = selector.preprocessor_.transform(X_train)
                    X_test_pre = selector.preprocessor_.transform(X_test)
                    pre_names = selector.preprocessed_feature_names_
                    idx = [i for i, f in enumerate(pre_names) if f in sel_feats]
                    if not idx:
                        continue
                    X_train, X_test = X_train_pre[:, idx], X_test_pre[:, idx]
                else:
                    idx = [i for i, f in enumerate(feature_names) if f in sel_feats]
                    if not idx:
                        continue
                    X_train, X_test = X_train[:, idx], X_test[:, idx]

            # Train model
            model_loo = clone(model)
            model_loo.fit(X_train, y_train)
            
            # Predict
            y_pred = model_loo.predict_proba(X_test)[:, 1]
            predictions.append(y_pred[0])
            y_true.append(y_test[0])
        
        # Calculate AUC
        auc_score = roc_auc_score(y_true, predictions)
        
        # Bootstrap CI for LOOCV
        n_bootstrap = 1000
        bootstrap_aucs = []
        
        for _ in range(n_bootstrap):
            idx = resample(range(len(y_true)), n_samples=len(y_true))
            boot_y = np.array(y_true)[idx]
            boot_pred = np.array(predictions)[idx]
            
            if len(np.unique(boot_y)) == 2:
                bootstrap_aucs.append(roc_auc_score(boot_y, boot_pred))
        
        ci = [np.percentile(bootstrap_aucs, 2.5), 
              np.percentile(bootstrap_aucs, 97.5)] if bootstrap_aucs else [auc_score, auc_score]
        
        return {
            'auc': auc_score,
            'ci': ci,
            'n_iterations': len(y),
            'deterministic': True,
            'y_true': np.array(y_true).tolist(),
            'y_pred': np.array(predictions).tolist()
        }
    
    def simple_bootstrap_evaluation(self, model, X, y, feature_names=None, n_bootstrap=500):
        """Simple bootstrap without .632+ correction.
        If use_nested_selection is True, performs train-only selection for each bootstrap sample.
        """
        
        bootstrap_scores = []
        
        for i in tqdm(range(n_bootstrap), desc="Simple Bootstrap"):
            # Bootstrap sample
            idx = resample(range(len(y)), n_samples=len(y), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            oob_idx = list(set(range(len(y))) - set(idx))
            
            if len(oob_idx) > 5 and len(np.unique(y_boot)) == 2:
                if self.use_nested_selection and feature_names is not None:
                    selector = EnhancedParallelEnsembleSTABL(**self.stabl_params)
                    selector.fit(X_boot, y_boot, feature_names=np.array(feature_names))
                    sel_feats = selector.selected_features_ or []
                    if not sel_feats:
                        continue
                    if selector.use_preprocessing:
                        X_boot_pre = selector.preprocessor_.transform(X_boot)
                        X_oob_pre = selector.preprocessor_.transform(X[oob_idx])
                        pre_names = selector.preprocessed_feature_names_
                        col_idx = [j for j, f in enumerate(pre_names) if f in sel_feats]
                        if not col_idx:
                            continue
                        X_boot_sel = X_boot_pre[:, col_idx]
                        X_oob_sel = X_oob_pre[:, col_idx]
                    else:
                        col_idx = [j for j, f in enumerate(feature_names) if f in sel_feats]
                        if not col_idx:
                            continue
                        X_boot_sel = X_boot[:, col_idx]
                        X_oob_sel = X[oob_idx][:, col_idx]
                else:
                    # Legacy scaling only
                    if not hasattr(self, '_preprocessing_applied'):
                        scaler = StandardScaler()
                        X_boot_sel = scaler.fit_transform(X_boot)
                        X_oob_sel = scaler.transform(X[oob_idx])
                    else:
                        X_boot_sel = X_boot
                        X_oob_sel = X[oob_idx]

                # Train and evaluate
                model_boot = clone(model)
                model_boot.fit(X_boot_sel, y_boot)
                
                y_pred = model_boot.predict_proba(X_oob_sel)[:, 1]
                
                if len(np.unique(y[oob_idx])) == 2:
                    auc_score = roc_auc_score(y[oob_idx], y_pred)
                    bootstrap_scores.append(auc_score)
        
        if bootstrap_scores:
            return {
                'auc': np.mean(bootstrap_scores),
                'ci': [np.percentile(bootstrap_scores, 2.5), 
                       np.percentile(bootstrap_scores, 97.5)],
                'std': np.std(bootstrap_scores),
                'n_bootstrap': len(bootstrap_scores)
            }
        
        return None
    
    def calculate_consensus(self, results):
        """Calculate consensus estimates from all methods."""
        
        aucs = []
        for method, result in results.items():
            if method not in ['consensus', 'optimization'] and result is not None:
                aucs.append(result['auc'])
        
        if aucs:
            # Median as robust central tendency
            median_auc = np.median(aucs)
            
            # MAD (Median Absolute Deviation) as robust spread
            mad = np.median(np.abs(aucs - median_auc))
            
            # Range
            min_auc = np.min(aucs)
            max_auc = np.max(aucs)
            
            return {
                'median': median_auc,
                'mad': mad,
                'mean': np.mean(aucs),
                'std': np.std(aucs),
                'range': [min_auc, max_auc],
                'n_methods': len(aucs)
            }
        
        return None


def _loocv_predictions_for_model(model, X, y):
    """Compute LOOCV predictions with train-only impute+scale regardless of global flags.

    Returns dict with y_true, y_pred, auc, and bootstrap CI.
    """
    loo = LeaveOneOut()
    y_true = []
    y_pred = []
    for tr, te in loo.split(X):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        # Train-only impute + scale
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_tr_i = imputer.fit_transform(X_tr)
        X_te_i = imputer.transform(X_te)
        X_tr_s = scaler.fit_transform(X_tr_i)
        X_te_s = scaler.transform(X_te_i)
        m = clone(model)
        m.fit(X_tr_s, y_tr)
        try:
            p = m.predict_proba(X_te_s)[:, 1]
        except Exception:
            s = m.decision_function(X_te_s)
            p = 1 / (1 + np.exp(-s))
        y_true.append(int(y_te[0]))
        y_pred.append(float(p[0]))
    auc_score = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else float('nan')
    # Bootstrap CI on LOOCV predictions
    rng = np.random.RandomState(42)
    idx_all = np.arange(len(y_true))
    scores = []
    for _ in range(1000):
        bs = resample(idx_all, replace=True, n_samples=len(idx_all), random_state=rng)
        yy = np.array(y_true)[bs]
        pp = np.array(y_pred)[bs]
        if len(np.unique(yy)) == 2:
            try:
                scores.append(roc_auc_score(yy, pp))
            except Exception:
                continue
    ci = [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))] if scores else [auc_score, auc_score]
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'auc': float(auc_score),
        'ci': ci,
    }


def _permutation_test_auc(model, X, y, n_permutations=1000, test_size=0.2, random_state=42):
    """Run label permutation test for a fixed feature panel and model."""

    rng = np.random.RandomState(random_state)
    X = np.asarray(X)
    y = np.asarray(y)

    permuted_aucs = []

    for i in range(int(n_permutations)):
        y_perm = y.copy()
        rng.shuffle(y_perm)

        stratify = y_perm if (len(np.unique(y_perm)) > 1 and len(y_perm) > 2) else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_perm,
                test_size=test_size,
                stratify=stratify,
                random_state=random_state + i,
            )
        except ValueError:
            continue

        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_train_i = imputer.fit_transform(X_train)
        X_test_i = imputer.transform(X_test)
        X_train_s = scaler.fit_transform(X_train_i)
        X_test_s = scaler.transform(X_test_i)

        m = clone(model)
        try:
            m.fit(X_train_s, y_train)
        except Exception:
            continue

        try:
            if hasattr(m, 'predict_proba'):
                y_prob = m.predict_proba(X_test_s)[:, 1]
            else:
                decision = m.decision_function(X_test_s)
                y_prob = 1.0 / (1.0 + np.exp(-decision))
        except Exception:
            continue

        if len(np.unique(y_test)) < 2:
            continue

        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            continue
        permuted_aucs.append(float(auc))

    return permuted_aucs


class EnhancedReporting:
    """Enhanced reporting with comprehensive validation comparison and feature visualization."""
    
    @staticmethod
    def create_feature_report(selected_features, feature_freq, consensus_threshold=0.5):
        """Create detailed feature selection report."""
        
        report = "\n" + "="*80 + "\n"
        report += "🎯 SELECTED FEATURES REPORT\n"
        report += "="*80 + "\n\n"
        
        # Sort features by frequency
        sorted_features = sorted(feature_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Show top 20 features with visual bars
        for i, (feat, freq) in enumerate(sorted_features[:20], 1):
            # Create visual progress bar
            bar_length = 30
            filled = int(freq * bar_length)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Color coding
            if freq >= 0.8:
                status = '🟢'
            elif freq >= 0.6:
                status = '🟡'
            else:
                status = '🔴'
            
            # Check if selected
            selected = '[✓]' if feat in selected_features else '[ ]'
            
            # Truncate feature name if too long
            feat_display = feat[:50] + '...' if len(feat) > 50 else feat
            
            report += f"{i:2d}. {selected} {feat_display:<50} {status}\n"
            report += f"    {bar} {freq*100:.1f}%\n\n"
        
        # Summary statistics
        report += "\n📊 Summary:\n"
        report += f"  Total unique features selected: {len(sorted_features)}\n"
        report += f"  Features meeting consensus: {len([f for f, freq in sorted_features if freq >= consensus_threshold])}\n"
        report += f"  Final features selected: {len(selected_features)}\n"
        
        return report
    
    @staticmethod
    def create_comparison_table(results: Dict) -> str:
        """Create formatted comparison table."""
        
        table = "\n" + "="*80 + "\n"
        table += "COMPREHENSIVE VALIDATION COMPARISON\n"
        table += "="*80 + "\n\n"
        
        # Add optimization results if present
        if 'optimization' in results and results['optimization']:
            opt = results['optimization']
            table += "HYPERPARAMETER OPTIMIZATION:\n"
            table += f"  Best parameters: {opt['best_params']}\n"
            table += "-"*80 + "\n\n"
        
        # Header
        table += f"{'Method':<25} | {'AUC':^7} | {'95% CI':^15} | {'Width':^7} | {'Note':<20}\n"
        table += "-"*80 + "\n"
        
        # Process each method
        method_order = ['bootstrap_632_plus', 'repeated_stratified_cv', 'loocv', 'simple_bootstrap', 'nested_monte_carlo']
        notes = {
            'bootstrap_632_plus': 'Conservative',
            'repeated_stratified_cv': 'Optimistic', 
            'loocv': 'Deterministic',
            'simple_bootstrap': 'No correction',
            'nested_monte_carlo': 'Unbiased (nested MCCV)'
        }
        
        for method in method_order:
            if method in results and results[method] is not None:
                result = results[method]
                auc = result['auc']
                
                # Get CI (handle different formats)
                if 'ci' in result and result['ci'] is not None:
                    ci_lower, ci_upper = result['ci']
                    ci_str = f"[{ci_lower:.3f}, {ci_upper:.3f}]"
                    width = ci_upper - ci_lower
                else:
                    ci_str = "N/A"
                    width = 0
                
                # Format method name
                method_display = method.replace('_', ' ').title()
                if method == 'bootstrap_632_plus':
                    method_display = 'Bootstrap .632+'
                elif method == 'repeated_stratified_cv':
                    method_display = 'Stratified CV (repeated)'
                elif method == 'loocv':
                    method_display = 'Leave-One-Out'
                elif method == 'simple_bootstrap':
                    method_display = 'Simple Bootstrap'
                
                note = notes.get(method, '')
                
                table += f"{method_display:<25} | {auc:^7.3f} | {ci_str:^15} | {width:^7.3f} | {note:<20}\n"
        
        # Add consensus
        if 'consensus' in results and results['consensus'] is not None:
            consensus = results['consensus']
            table += "-"*80 + "\n"
            table += f"\nConsensus estimate: {consensus['median']:.3f} ± {consensus['mad']:.3f} (median ± MAD)\n"
            table += f"Range across methods: [{consensus['range'][0]:.3f}, {consensus['range'][1]:.3f}]\n"
        
        table += "\n" + "="*80 + "\n"
        
        return table
    
    @staticmethod
    def create_ci_comparison(results: Dict) -> str:
        """Create CI method comparison for Bootstrap .632+."""
        
        if 'bootstrap_632_plus' not in results:
            return ""
        
        boot_result = results['bootstrap_632_plus']
        if 'ci_methods' not in boot_result:
            return ""
        
        ci_methods = boot_result['ci_methods']
        
        table = "\nCONFIDENCE INTERVAL METHODS COMPARISON\n"
        table += "-"*50 + "\n"
        table += f"{'CI Method':<20} | {'Lower':^7} | {'Upper':^7} | {'Width':^7}\n"
        table += "-"*50 + "\n"
        
        for method, (lower, upper) in ci_methods.items():
            width = upper - lower
            table += f"{method.capitalize():<20} | {lower:^7.3f} | {upper:^7.3f} | {width:^7.3f}\n"
        
        table += "-"*50 + "\n"
        table += "Note: BCa (Bias-Corrected) is most accurate for small samples\n"
        
        return table
    
    @staticmethod
    def create_performance_plots(
        results: Dict,
        selected_features: List[str],
        feature_freq: Dict[str, float],
        output_dir: Path,
    ) -> List[str]:
        """Render each report figure separately to individual files."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []

        def _make_fig(kind: str = 'wide'):
            if HAS_BEAUTIFUL:
                return create_beautiful_figure(kind)
            size_map = {'wide': (12, 6), 'tall': (8, 10), 'square': (8, 8)}
            return plt.subplots(figsize=size_map.get(kind, (10, 6)))

        def _save(fig, base_name: str):
            base = output_dir / base_name
            if HAS_BEAUTIFUL:
                save_beautiful_figure(fig, base)
                plt.close(fig)
                saved_paths.append(str(base.with_suffix('.png')))
            else:
                fig.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
                fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight')
                plt.close(fig)
                saved_paths.append(str(base.with_suffix('.png')))

        # 1. Validation method comparison
        method_specs = [
            ('bootstrap_632_plus', 'Bootstrap .632+'),
            ('repeated_stratified_cv', 'Stratified CV (repeated)'),
            ('loocv', 'Leave-One-Out'),
            ('simple_bootstrap', 'Simple Bootstrap'),
        ]
        methods = []
        aucs = []
        ci_deltas = []
        for key, label in method_specs:
            metric = results.get(key)
            if not metric or metric.get('auc') is None:
                continue
            auc = float(metric['auc'])
            methods.append(label)
            aucs.append(auc)
            if metric.get('ci') is not None:
                lo, hi = metric['ci']
                ci_deltas.append((max(0.0, auc - lo), max(0.0, hi - auc)))
            elif metric.get('std') is not None:
                std = float(metric['std'])
                ci_deltas.append((std, std))
            else:
                ci_deltas.append((0.0, 0.0))

        if methods:
            fig, ax = _make_fig('wide')
            x_pos = np.arange(len(methods))
            yerr = np.array(ci_deltas).T if ci_deltas else None
            if HAS_BEAUTIFUL:
                colors = [NORD_COLORS['nord14'] if auc == max(aucs) else NORD_COLORS['nord9'] for auc in aucs]
                ax.bar(x_pos, aucs, yerr=yerr, capsize=6, alpha=0.9,
                       color=colors, edgecolor=NORD_COLORS['nord3'], linewidth=2)
                ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
            else:
                ax.bar(x_pos, aucs, yerr=yerr, capsize=5, alpha=0.8)
                ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.set_ylabel('AUC')
            ax.set_ylim([0, 1])
            ax.set_title('Validation Method Comparison')
            ax.grid(True, alpha=0.3)
            for xpos, auc in zip(x_pos, aucs):
                ax.text(xpos, auc + 0.02, f"{auc:.3f}", ha='center', va='bottom', fontsize=10)
            _save(fig, 'validation_methods')

            # Manuscript helper chart using shared utility
            try:
                from utils.plotting_utils import plot_validation_comparison

                helper_map = {}
                for key, _ in method_specs:
                    if key in results and results[key]:
                        entry = {'auc': results[key].get('auc')}
                        if 'ci' in results[key] and results[key]['ci'] is not None:
                            entry['ci'] = results[key]['ci']
                        elif 'std' in results[key]:
                            entry['std'] = results[key]['std']
                        helper_map[key] = entry
                if helper_map:
                    plot_validation_comparison(helper_map, output_dir / 'validation_comparison')
                    saved_paths.append(str((output_dir / 'validation_comparison').with_suffix('.png')))
            except Exception:
                pass

        # 2. Top consensus features
        if feature_freq:
            sorted_feats = sorted(feature_freq.items(), key=lambda kv: kv[1], reverse=True)
            top_feats = sorted_feats[:10]
            if top_feats:
                fig, ax = _make_fig('tall')
                names = [f if len(f) <= 40 else f[:37] + '...' for f, _ in top_feats]
                values = [freq for _, freq in top_feats]
                y_pos = np.arange(len(names))
                if HAS_BEAUTIFUL:
                    ax.barh(y_pos, values, color=NORD_COLORS['nord9'],
                            edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
                else:
                    ax.barh(y_pos, values, alpha=0.8)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names)
                ax.invert_yaxis()
                ax.set_xlabel('Selection Frequency')
                ax.set_xlim([0, 1])
                ax.set_title('Top Consensus Features')
                for i, val in enumerate(values):
                    ax.text(val + 0.01, i, f"{val:.2f}", va='center', fontsize=12)
                ax.grid(True, axis='x', alpha=0.3)
                _save(fig, 'selected_features')

        # 3. Confidence interval width comparison
        ci_methods = results.get('bootstrap_632_plus', {}).get('ci_methods')
        if ci_methods:
            names = list(ci_methods.keys())
            widths = [ci[1] - ci[0] for ci in ci_methods.values()]
            fig, ax = _make_fig('square')
            if HAS_BEAUTIFUL:
                ax.bar(np.arange(len(names)), widths, color=NORD_COLORS['nord13'],
                       edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
            else:
                ax.bar(np.arange(len(names)), widths, alpha=0.8)
            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names, rotation=45)
            ax.set_ylabel('Interval Width')
            ax.set_title('Bootstrap .632+ CI Methods')
            ax.grid(True, alpha=0.3)
            for idx, width in enumerate(widths):
                ax.text(idx, width + 0.005, f"{width:.3f}", ha='center', fontsize=12)
            _save(fig, 'ci_widths')

        # 4. Hyperparameter optimisation trace
        optimization = results.get('optimization')
        if optimization and optimization.get('cv_results') and 'param_C' in optimization['cv_results']:
            cv_results = optimization['cv_results']
            C_values = cv_results['param_C'].data
            mean_scores = cv_results['mean_test_score']
            std_scores = cv_results['std_test_score']
            fig, ax = _make_fig('wide')
            x_pos = np.arange(len(C_values))
            if HAS_BEAUTIFUL:
                ax.errorbar(x_pos, mean_scores, yerr=std_scores, marker='o', capsize=6,
                            alpha=0.9, color=NORD_COLORS['nord10'], linewidth=2)
            else:
                ax.errorbar(x_pos, mean_scores, yerr=std_scores, marker='o', capsize=5,
                            alpha=0.8)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'{c:.3f}' for c in C_values], rotation=45)
            ax.set_xlabel('C parameter')
            ax.set_ylabel('CV AUC')
            ax.set_title('Hyperparameter Optimisation Trace')
            ax.grid(True, alpha=0.3)
            _save(fig, 'optimization_trace')

        # 5. Bootstrap .632+ distribution
        bootstrap_scores = results.get('bootstrap_632_plus', {}).get('per_bootstrap_auc')
        if bootstrap_scores:
            scores = np.asarray(bootstrap_scores, dtype=float)
            fig, ax = _make_fig('wide')
            if HAS_BEAUTIFUL:
                ax.hist(scores, bins=30, color=NORD_COLORS['nord9'],
                        edgecolor=NORD_COLORS['nord3'], linewidth=1.5, alpha=0.85)
            else:
                ax.hist(scores, bins=30, alpha=0.8)
            ax.set_xlabel('AUC')
            ax.set_ylabel('Frequency')
            ax.set_title('Bootstrap .632+ Distribution')
            ax.axvline(scores.mean(), color=NORD_COLORS['nord11'] if HAS_BEAUTIFUL else 'red',
                       linewidth=2, linestyle='--', label=f"Mean {scores.mean():.3f}")
            ax.legend()
            _save(fig, 'bootstrap_distribution')

        # 6. Summary text panel
        summary_lines = ['Performance Summary', '===================', '']
        consensus = results.get('consensus')
        if consensus:
            summary_lines.append(f"Consensus AUC: {consensus['median']:.3f} ± {consensus['mad']:.3f}")
            summary_lines.append(f"Range: [{consensus['range'][0]:.3f}, {consensus['range'][1]:.3f}]")
            summary_lines.append(f"Methods evaluated: {consensus['n_methods']}")
        if optimization and optimization.get('best_params'):
            summary_lines.append('')
            summary_lines.append(f"Best params: {optimization['best_params']}")

        fig, ax = _make_fig('square')
        ax.axis('off')
        ax.text(0.02, 0.98, '\n'.join(summary_lines), ha='left', va='top', fontsize=16,
                family='monospace')
        _save(fig, 'summary')

        return saved_paths


# Enhanced ParallelEnsembleSTABL with preprocessing and correlation grouping
class EnhancedParallelEnsembleSTABL:
    """
    ENHANCED: Parallel ensemble STABL with preprocessing and correlation grouping.
    
    New features:
    - Integrated preprocessing pipeline
    - Correlation-based feature grouping
    - Knockoff artificial features option
    """
    
    def __init__(self,
                 n_runs: int = 10,
                 n_bootstraps: int = 500,
                 threshold: float = 0.5,
                 sample_fraction: float = 0.75,
                 consensus_threshold: float = 0.5,
                 n_workers: int = None,
                 cores_per_worker: int = 2,
                 force_parallel: bool = True,
                 use_fdr: bool = True,
                 explore: bool = True,
                 n_explore: int = 10,
                 random_state: int = 42,
                 # NEW parameters
                 artificial_type: str = 'knockoff',
                 perc_corr_group_threshold: float = None,
                 # Lambda grid / FDR controls
                 lambda_grid: str = 'auto',
                 n_lambda: int = 30,
                 fdr_start: float = 0.1,
                 fdr_end: float = 1.0,
                 fdr_step: float = 0.01,
                 use_preprocessing: bool = True,
                 preprocessing_config: Dict = None,
                 c_value: float = None,
                 hard_threshold: float = None):
        
        self.n_runs = n_runs
        self.n_bootstraps = n_bootstraps
        self.threshold = threshold
        self.sample_fraction = sample_fraction
        self.consensus_threshold = consensus_threshold
        self.n_workers = n_workers or max(2, cpu_count() // 2)
        self.cores_per_worker = cores_per_worker
        self.force_parallel = force_parallel
        self.use_fdr = use_fdr
        self.explore = explore
        self.n_explore = n_explore
        self.random_state = random_state
        
        # NEW attributes
        self.artificial_type = artificial_type
        self.perc_corr_group_threshold = perc_corr_group_threshold
        self.lambda_grid = lambda_grid
        self.n_lambda = n_lambda
        self.fdr_start = fdr_start
        self.fdr_end = fdr_end
        self.fdr_step = fdr_step
        self.use_preprocessing = use_preprocessing
        self.preprocessing_config = preprocessing_config or {}
        self.c_value = c_value
        self.hard_threshold = hard_threshold
        
        self.feature_matrix_ = None
        self.consensus_features_ = None
        self.selected_features_ = None
        self.preprocessor_ = None
        self.original_feature_names_ = None
        self.preprocessed_feature_names_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: np.ndarray) -> 'EnhancedParallelEnsembleSTABL':
        """Fit enhanced parallel ensemble STABL with preprocessing."""
        
        logger.info("="*80)
        logger.info("ENHANCED PARALLEL ENSEMBLE STABL EXECUTION")
        logger.info("="*80)
        
        self.original_feature_names_ = feature_names
        
        # Apply preprocessing if enabled
        if self.use_preprocessing:
            logger.info("Applying preprocessing pipeline...")
            self.preprocessor_ = PreprocessingPipeline(**self.preprocessing_config)
            X_preprocessed, preprocessed_features = self.preprocessor_.fit_transform(X, feature_names)
            self.preprocessed_feature_names_ = preprocessed_features
            feature_names = np.array(preprocessed_features)
        else:
            X_preprocessed = X
            self.preprocessed_feature_names_ = feature_names.tolist()
        
        # Scale data (if not already done by preprocessing)
        if not self.use_preprocessing or not self.preprocessing_config.get('use_scaler', True):
            X_preprocessed = StandardScaler().fit_transform(X_preprocessed)
        
        # Prepare parallel jobs
        jobs = []
        for run_idx in range(self.n_runs):
            jobs.append({
                'run_idx': run_idx,
                'X': X_preprocessed,
                'y': y,
                'feature_names': feature_names,
                'n_features': X_preprocessed.shape[1],
                'seed': self.random_state + run_idx * 1000
            })
        
        # Run parallel ensemble
        with Pool(processes=self.n_workers) as pool:
            process_func = partial(
                self._run_single_stabl_worker,
                n_bootstraps=self.n_bootstraps,
                threshold=self.threshold,
                sample_fraction=self.sample_fraction,
                cores_per_worker=self.cores_per_worker,
                force_parallel=self.force_parallel,
                use_fdr=self.use_fdr,
                explore=self.explore,
                n_explore=self.n_explore,
                artificial_type=self.artificial_type,
                perc_corr_group_threshold=self.perc_corr_group_threshold,
                lambda_grid=self.lambda_grid,
                n_lambda=self.n_lambda,
                fdr_start=self.fdr_start,
                fdr_end=self.fdr_end,
                fdr_step=self.fdr_step,
                hard_threshold=self.hard_threshold,
                c_value=self.c_value  # Pass C value for Optuna optimization
            )
            
            results = list(tqdm(
                pool.imap(process_func, jobs),
                total=len(jobs),
                desc="Ensemble runs"
            ))
        
        # Aggregate results
        self._aggregate_results(results)
        
        logger.info(f"Selected {len(self.selected_features_)} features after consensus")
        
        return self
    
    @staticmethod
    def _run_single_stabl_worker(job_data: Dict, **kwargs) -> Dict:
        """Enhanced worker function with correlation grouping."""
        
        run_idx = job_data['run_idx']
        X = job_data['X']
        y = job_data['y']
        feature_names = job_data['feature_names']
        seed = job_data['seed']
        
        # Setup STABL with enhanced configuration
        base_estimator = LogisticRegression(
            penalty='l1',
            C=kwargs.get('c_value', 1.0),  # Allow C value override from Optuna
            solver='liblinear',
            class_weight='balanced',
            max_iter=10000,  # Increased for better convergence
            random_state=seed
        )
        
        # Enhanced STABL configuration
        stabl = Stabl(
            base_estimator=base_estimator,
            n_bootstraps=kwargs['n_bootstraps'],
            artificial_type=kwargs['artificial_type'],  # NEW: knockoff or random_permutation
            artificial_proportion=1.0,
            sample_fraction=kwargs['sample_fraction'],
            sample_weight_bootstrap='balanced',
            replace=False,
            explore=kwargs['explore'],
            n_explore=kwargs['n_explore'],
            hard_threshold=kwargs['hard_threshold'],
            fdr_threshold_range=(None if kwargs['hard_threshold'] is not None else np.arange(kwargs['fdr_start'], kwargs['fdr_end'], kwargs['fdr_step'])),
            lambda_grid=kwargs['lambda_grid'],
            n_lambda=kwargs['n_lambda'],
            perc_corr_group_threshold=kwargs['perc_corr_group_threshold'],  # NEW: correlation grouping
            n_jobs=kwargs['cores_per_worker'] if kwargs['force_parallel'] else 1,
            random_state=seed,
            verbose=0
        )
        
        # Fit STABL
        stabl.fit(X, y)
        
        # Get results
        selected_mask = stabl.get_support()
        selected_features = feature_names[selected_mask].tolist()
        
        # FDP+ Monitoring: Capture the threshold that was selected
        fdp_threshold = None
        if hasattr(stabl, 'threshold_'):
            fdp_threshold = stabl.threshold_
        elif hasattr(stabl, 'fdr_threshold_'):
            fdp_threshold = stabl.fdr_threshold_
            
        return {
            'run': run_idx,
            'selected_features': selected_features,
            'n_selected': len(selected_features),
            'fdp_threshold': fdp_threshold  # Track FDP+ selected threshold
        }
    
    def _aggregate_results(self, results: List[Dict]):
        """Aggregate results from parallel runs."""
        
        # Collect FDP+ thresholds for reporting
        fdp_thresholds = [r.get('fdp_threshold') for r in results if r.get('fdp_threshold') is not None]
        if fdp_thresholds:
            logger.info(f"FDP+ Thresholds across runs: min={min(fdp_thresholds):.3f}, "
                       f"median={np.median(fdp_thresholds):.3f}, max={max(fdp_thresholds):.3f}")
        
        # Build feature frequency matrix
        all_features = set()
        for r in results:
            all_features.update(r['selected_features'])
        
        feature_list = sorted(all_features)
        feature_freq = {f: 0 for f in feature_list}
        
        for r in results:
            for f in r['selected_features']:
                feature_freq[f] += 1
        
        # Normalize frequencies
        for f in feature_freq:
            feature_freq[f] /= self.n_runs
        
        # Apply consensus threshold
        consensus_features = [f for f, freq in feature_freq.items() 
                            if freq >= self.consensus_threshold]
        
        self.feature_freq_ = feature_freq
        self.consensus_features_ = consensus_features
        self.selected_features_ = consensus_features
        
    def get_support(self, indices=False):
        """Get selected feature mask."""
        return self.selected_features_


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='POPF STABL V3 Enhanced Pipeline')
    
    # Basic arguments (from V2)
    parser.add_argument('--consensus-threshold', type=float, default=0.5,
                       help='Consensus threshold for feature selection (default: 0.5)')
    parser.add_argument('--ensemble-runs', type=int, default=10,
                       help='Number of ensemble STABL runs (default: 10)')
    parser.add_argument('--n-bootstraps', type=int, default=500,
                       help='Number of bootstraps per STABL run (default: 500)')
    parser.add_argument('--n-features', type=int, default=None,
                       help='Force selection of top N features')
    parser.add_argument('--model', type=str, default='xgb',
                       choices=['lr', 'enlr', 'en', 'rf', 'xgb', 'lgb', 'svm', 'ensemble'],
                       help='Model to use for evaluation')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--radiomics-path', type=str, default=None,
                       help='Optional: explicit path to radiomics CSV to use (overrides default data discovery)')
    parser.add_argument('--matches-path', type=str, default=None,
                       help='Path to outcome matching CSV (default: data/outcome_matches.csv)')
    parser.add_argument('--positive-grades', type=str, default='B,C',
                       help="Comma-separated POPF grades considered positive (default: 'B,C'). Example to include 'BL': B,C,BL")
    parser.add_argument('--allow-id-normalization', action='store_true', default=False,
                       help='If set, attempt a normalized ID fallback join (accents/case/honorifics removed) when exact match fails. Default: off (EXACT match only).')
    parser.add_argument('--texture-only', action='store_true', default=False,
                       help='Restrict analysis to texture features only (GLCM/GLRLM/GLSZM/GLDM/NGTDM families).')
    parser.add_argument('--feature-whitelist', type=str, default=None,
                       help='Optional path to a text file (one feature per line). If provided, restricts modeling to this feature subset before any selection.')
    parser.add_argument('--n-workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--validation-bootstrap', type=int, default=2000,
                       help='Number of bootstrap iterations for validation (default: 2000)')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize model hyperparameters')
    parser.add_argument('--c-value', type=float, default=None,
                       help='Use specific C value instead of GridSearchCV (e.g., 0.272 from Optuna)')
    # Elastic-Net LR options
    parser.add_argument('--en-l1-ratio', type=float, default=0.5,
                       help='Elastic-Net LR l1_ratio when using --model enlr/en (default: 0.5)')
    parser.add_argument('--en-c-grid', nargs='+', type=float, default=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5],
                       help='C grid for Elastic-Net LR optimization when --optimize is set')
    parser.add_argument('--en-l1-grid', nargs='+', type=float, default=[0.3, 0.5, 0.7],
                       help='l1_ratio grid for Elastic-Net LR optimization when --optimize is set')
    
    # Enhanced CI arguments (from V2)
    parser.add_argument('--use-firth', action='store_true',
                       help='Use Firth penalized logistic regression for small samples')
    parser.add_argument('--smooth-ci', action='store_true',
                       help='Use kernel density smoothing for bootstrap CIs')
    parser.add_argument('--use-ensemble-ci', action='store_true',
                       help='Use model ensemble for variance reduction')
    
    # NEW V3 arguments - Preprocessing
    parser.add_argument('--use-preprocessing', action='store_true', default=True,
                       help='Enable preprocessing pipeline (default: True)')
    parser.add_argument('--variance-threshold', type=float, default=0.01,
                       help='Variance threshold for feature filtering (default: 0.01)')
    parser.add_argument('--max-nan-fraction', type=float, default=0.2,
                       help='Maximum fraction of NaN values allowed (default: 0.2)')
    parser.add_argument('--impute-strategy', type=str, default='median',
                       choices=['median', 'mean', 'most_frequent'],
                       help='Imputation strategy (default: median)')
    parser.add_argument('--no-scaler', action='store_true',
                       help='Disable StandardScaler in preprocessing')
    
    # NEW V3 arguments - STABL enhancement
    parser.add_argument('--artificial-type', type=str, default='knockoff',
                       choices=['knockoff', 'random_permutation'],
                       help='Type of artificial features for FDR control (default: knockoff)')
    parser.add_argument('--corr-group-threshold', type=float, default=90,
                       help='Percentile threshold for correlation grouping (default: 90 for radiomics)')
    parser.add_argument('--no-corr-grouping', action='store_true',
                       help='Disable correlation-based feature grouping for testing')

    # NEW: FDR and lambda grid controls
    parser.add_argument('--fdr-start', type=float, default=0.1,
                       help='Lower bound of FDR threshold search range (default: 0.1)')
    parser.add_argument('--fdr-end', type=float, default=1.0,
                       help='Upper bound of FDR threshold search range (default: 1.0)')
    parser.add_argument('--fdr-step', type=float, default=0.01,
                       help='Step for FDR threshold search range (default: 0.01)')
    parser.add_argument('--lambda-grid', type=str, default='auto', choices=['auto'],
                       help='Lambda grid strategy (default: auto)')
    parser.add_argument('--n-lambda', type=int, default=30,
                       help='Number of lambda values for auto grid (default: 30)')
    parser.add_argument('--hard-threshold', type=float, default=None,
                       help='Legacy STABL hard selection threshold; when set, bypasses FDR search')

    # NEW: Nested Monte Carlo evaluation options
    parser.add_argument('--nested-mc', action='store_true',
                       help='Run nested Monte Carlo evaluation for unbiased AUROC')
    parser.add_argument('--nested-outer', type=int, default=50,
                       help='Number of outer Monte Carlo splits (default: 50)')
    parser.add_argument('--nested-test-size', type=float, default=0.2,
                       help='Outer test size fraction (default: 0.2)')
    parser.add_argument('--nested-inner-cv', type=int, default=5,
                       help='Inner CV folds for hyperparameter tuning (default: 5)')

    # NEW: CV/Bootstrap with nested per-split selection
    parser.add_argument('--cv-nested-selection', dest='cv_nested_selection', action='store_true',
                       help='Enable train-only STABL feature selection inside CV/boot splits')
    parser.add_argument('--no-cv-nested-selection', dest='cv_nested_selection', action='store_false',
                       help='Disable nested selection inside CV/boot splits (faster, optimistic)')
    parser.set_defaults(cv_nested_selection=True)
    parser.add_argument('--eval-consensus-only', action='store_true',
                       help='When nested selection is OFF, evaluate only on globally selected consensus features')
    
    # NEW: Select validation methods to run
    parser.add_argument('--val-methods', nargs='+',
                       choices=['bootstrap632+', 'repeated-cv', 'loocv', 'simple-bootstrap', 'nested-mc'],
                       default=['bootstrap632+', 'repeated-cv', 'loocv', 'simple-bootstrap'],
                       help='Validation methods to run (default: all except nested-mc)')
    # Permutation testing on frozen panel
    parser.add_argument('--permutation-test', action='store_true',
                       help='Run label permutation test on the frozen feature panel (post-selection)')
    parser.add_argument('--permutation-iterations', type=int, default=1000,
                       help='Number of permutation iterations (default: 1000)')
    parser.add_argument('--permutation-test-size', type=float, default=0.2,
                       help='Test size fraction for each permutation split (default: 0.2)')
    parser.add_argument('--permutation-random-state', type=int, default=42,
                       help='Random seed for permutation testing (default: 42)')
    # Repeated-CV configuration
    parser.add_argument('--cv-splits', type=int, default=4,
                       help='Number of folds for repeated stratified CV (default: 4)')
    parser.add_argument('--cv-repeats', type=int, default=20,
                       help='Number of repeats for repeated stratified CV (default: 20)')

    # NEW: Temporal holdout by scanner StudyDate
    parser.add_argument('--temporal-holdout', action='store_true',
                       help='Perform temporal validation: train on oldest cohort, test on newest')
    parser.add_argument('--scanner-metadata-path', type=str, default=None,
                       help='Path to scanner metadata CSV containing StudyDate')
    parser.add_argument('--scanner-id-col', type=str, default='scanner_patient_name',
                       help="Identifier column in scanner metadata (default: 'scanner_patient_name')")
    parser.add_argument('--date-col', type=str, default='StudyDate',
                       help="Date column name in scanner metadata (default: 'StudyDate')")
    parser.add_argument('--holdout-fraction', type=float, default=0.3,
                       help='Fraction of newest patients to keep as temporal holdout (default: 0.3)')

    # NEW: Evaluate multiple models on the selected panel with LOOCV + ROC plots
    parser.add_argument('--eval-all-models', action='store_true',
                       help='Evaluate LR, EN, RF, SVM, XGB/LGB (if available) on the frozen panel and plot ROC curves')
    parser.add_argument('--extra-panel-k', type=int, default=0,
                       help='Optionally add K top-frequency features (beyond selected) for non-linear models')
    # Train-only LR tuning inside repeated-CV via Optuna
    parser.add_argument('--cv-tune-lr', action='store_true',
                        help='Enable train-only LogisticRegression tuning inside repeated-CV via Optuna')
    parser.add_argument('--cv-tune-trials', type=int, default=30,
                        help='Optuna trials for LR tuning inside CV (default: 30)')
    parser.add_argument('--cv-tune-inner-splits', type=int, default=3,
                        help='Inner CV splits for LR tuning (default: 3, capped by minority class)')
    parser.add_argument('--cv-tune-seed', type=int, default=42,
                        help='Random seed for tuning')
    parser.add_argument('--lr-c-min', type=float, default=1e-3,
                        help='Minimum C for LR tuning (log-uniform)')
    parser.add_argument('--lr-c-max', type=float, default=1e2,
                        help='Maximum C for LR tuning (log-uniform)')
    parser.add_argument('--en-l1-min', type=float, default=0.3,
                        help='Minimum l1_ratio for Elastic-Net tuning')
    parser.add_argument('--en-l1-max', type=float, default=0.7,
                        help='Maximum l1_ratio for Elastic-Net tuning')
    parser.add_argument('--nested-stabl-bags', type=int, default=1,
                        help='Number of independent STABL fits per CV split (averaged).')
    parser.add_argument('--nested-add-ridge', action='store_true',
                        help='Include ridge logistic regression in the nested ensemble.')
    parser.add_argument('--nested-add-firth', action='store_true',
                        help="Include Firth logistic regression in the nested ensemble.")
    parser.add_argument('--nested-add-xgb', action='store_true',
                        help='Include a shallow XGBoost learner in the nested ensemble (requires xgboost).')
    # ROC labeling source
    parser.add_argument('--roc-label-source', type=str, default='oof',
                        choices=['oof', 'cv-mean', '632', 'best'],
                        help="Which AUC to annotate on the ROC figure: pooled OOF (oof), mean CV (cv-mean), .632+ (632), or best of available (best)")
    
    args = parser.parse_args()
    
    # Print banner
    print(BANNER)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'v3_enhanced_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'pipeline.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("Starting POPF STABL V3 Enhanced Pipeline")
    logger.info(f"Output directory: {output_dir}")
    
    # Log configuration
    logger.info("\nV3 Configuration:")
    logger.info(f"  Preprocessing: {args.use_preprocessing}")
    logger.info(f"  Artificial type: {args.artificial_type}")
    logger.info(f"  Correlation grouping: {not args.no_corr_grouping}")
    logger.info(f"  FDR range: start={args.fdr_start}, end={args.fdr_end}, step={args.fdr_step}")
    logger.info(f"  Lambda grid: {args.lambda_grid} (n_lambda={args.n_lambda})")
    logger.info(f"  CV nested selection: {args.cv_nested_selection}")
    logger.info(f"  Validation methods: {args.val_methods}")
    logger.info(f"  Hard threshold: {args.hard_threshold}")
    logger.info(f"  Positive grades considered as events: {args.positive_grades}")
    logger.info(f"  ID matching: {'exact only' if not args.allow_id_normalization else 'exact + normalized fallback'}")
    
    # Load data
    logger.info("\nLoading data...")
    # Prefer env var DATA_DIR; fallback to repository's local data/
    data_dir_env = os.getenv('DATA_DIR')
    data_dir = Path(data_dir_env) if data_dir_env else Path('data')

    # Determine radiomics CSV path
    if args.radiomics_path:
        radiomics_path = Path(args.radiomics_path)
        if not radiomics_path.exists():
            raise FileNotFoundError(f"Provided --radiomics-path not found: {radiomics_path}")
    else:
        # V3: Use RAW radiomics data - preprocessing will be applied in pipeline
        radiomics_path = data_dir / 'pancreatic_head_radiomics.csv'
        if not radiomics_path.exists():
            # Fallback to pre-filtered if raw doesn't exist
            radiomics_path = data_dir / 'radiomics_filtered_unsupervised.csv'
            logger.warning(f"Raw radiomics not found, using: {radiomics_path.name}")
    
    logger.info(f"Using radiomics: {radiomics_path.name}")
    
    # Load radiomics
    radiomics_df = pd.read_csv(radiomics_path)
    
    # Get target and features
    # Identify an ID column for potential temporal split merges
    id_col = 'scanner_patient_name' if 'scanner_patient_name' in radiomics_df.columns else (
        'patient_name' if 'patient_name' in radiomics_df.columns else (
            'patient_id' if 'patient_id' in radiomics_df.columns else None
        )
    )
    if 'cr_popf' in radiomics_df.columns:
        y = radiomics_df['cr_popf'].astype(int).values
        exclude_cols = ['scanner_patient_name', 'patient_name', 'patient_id', 'cr_popf', 'popf_grade']
        feature_cols = [col for col in radiomics_df.columns if col not in exclude_cols]
        # Only keep numeric columns as features
        numeric_cols = radiomics_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        # Optional texture-only restriction
        if args.texture_only:
            def _is_texture(name: str) -> bool:
                name = str(name).lower()
                return ('_glcm_' in name) or ('_glrlm_' in name) or ('_glszm_' in name) or ('_gldm_' in name) or ('_ngtdm_' in name)
            numeric_cols = [c for c in numeric_cols if _is_texture(c)]
        X = radiomics_df[numeric_cols].values
        feature_names = np.array(numeric_cols)
    else:
        # Load matches for target when using raw radiomics
        if args.matches_path:
            matches_path = Path(args.matches_path)
        else:
            matches_path = data_dir / 'outcome_matches.csv'
        if not matches_path.exists():
            raise FileNotFoundError(f"Matches file not found: {matches_path}")
        matches_df = pd.read_csv(matches_path)

        # Determine which column to use for merging
        if 'scanner_patient_name' in radiomics_df.columns:
            merge_col = 'scanner_patient_name'
        elif 'patient_name' in radiomics_df.columns:
            # Normalize radiomics ID column name
            radiomics_df = radiomics_df.rename(columns={'patient_name': 'scanner_patient_name'})
            merge_col = 'scanner_patient_name'
        elif 'patient_id' in radiomics_df.columns:
            radiomics_df = radiomics_df.rename(columns={'patient_id': 'scanner_patient_name'})
            merge_col = 'scanner_patient_name'
        else:
            raise ValueError("No patient identifier column found in radiomics data")

        # Normalize matches ID column
        m_id_col = 'scanner_patient_name' if 'scanner_patient_name' in matches_df.columns else (
            'patient_name' if 'patient_name' in matches_df.columns else (
                'patient_id' if 'patient_id' in matches_df.columns else None
            )
        )
        if m_id_col is None:
            raise KeyError("Matches file must contain an identifier column (scanner_patient_name/patient_name/patient_id)")
        if m_id_col != 'scanner_patient_name':
            matches_df = matches_df.rename(columns={m_id_col: 'scanner_patient_name'})

        if 'popf_grade' not in matches_df.columns:
            raise KeyError("Matches file must contain 'popf_grade' column")

        # Alignment sanity checks before merge
        n_r_rows = len(radiomics_df)
        n_r_ids = radiomics_df['scanner_patient_name'].nunique()
        n_m_ids = matches_df['scanner_patient_name'].nunique()
        logger.info(f"Alignment pre-merge: radiomics rows={n_r_rows}, unique IDs={n_r_ids}; matches unique IDs={n_m_ids}; matches path={matches_path.name}")

        # Merge to get POPF grades (left join to assess missing)
        merged = radiomics_df.merge(
            matches_df[['scanner_patient_name', 'popf_grade']],
            on='scanner_patient_name',
            how='left'
        )

        n_missing = int(merged['popf_grade'].isna().sum())
        if n_missing > 0:
            if args.allow_id_normalization:
                logger.warning(f"Initial exact merge left {n_missing} unmatched IDs. Attempting normalized ID fallback...")
                # Fallback: normalized join on canonical IDs
                merged['scanner_patient_name_canon'] = merged['scanner_patient_name'].map(_canonicalize_id)
                matches_df['_scanner_patient_name_canon'] = matches_df['scanner_patient_name'].map(_canonicalize_id)
                # Deduplicate matches on canonical key, keep first
                matches_canon = matches_df.drop_duplicates('_scanner_patient_name_canon')[['_scanner_patient_name_canon', 'popf_grade']]
                merged = merged.merge(
                    matches_canon.rename(columns={'_scanner_patient_name_canon': 'scanner_patient_name_canon', 'popf_grade': 'popf_grade_canon'}),
                    on='scanner_patient_name_canon', how='left'
                )
                # Fill popf_grade from canonical match where missing
                merged['popf_grade'] = merged['popf_grade'].fillna(merged['popf_grade_canon'])
                merged = merged.drop(columns=['scanner_patient_name_canon', 'popf_grade_canon'])
                n_missing = int(merged['popf_grade'].isna().sum())
            # Drop any remaining unmatched rows
            if n_missing > 0:
                out_dir = Path(args.output_dir) if args.output_dir else Path('results') / f"alignment_nonharm"
                out_dir.mkdir(parents=True, exist_ok=True)
                unmatched_path = out_dir / 'radiomics_unmatched.csv'
                merged.loc[merged['popf_grade'].isna(), ['scanner_patient_name']].to_csv(unmatched_path, index=False)
                logger.warning(f"Dropping {n_missing} radiomics rows without POPF grade after exact{(' + normalized' if args.allow_id_normalization else '')} merge. Unmatched IDs written to {unmatched_path}")
                merged = merged.dropna(subset=['popf_grade']).reset_index(drop=True)

        radiomics_df = merged

        # Create binary CR-POPF outcome using configured positive grades
        pos_grades = set([g.strip().upper() for g in str(args.positive_grades).split(',') if g.strip()])
        radiomics_df['cr_popf'] = radiomics_df['popf_grade'].apply(lambda x: 1 if (str(x).upper() in pos_grades) else 0)
        y = radiomics_df['cr_popf'].astype(int).values

        # Exclude all non-feature columns
        exclude_cols = ['scanner_patient_name', 'patient_name', 'patient_id', 'cr_popf', 'popf_grade']
        feature_cols = [col for col in radiomics_df.columns if col not in exclude_cols]

        # Only keep numeric columns as features
        numeric_cols = radiomics_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        # Optional texture-only restriction
        if args.texture_only:
            def _is_texture(name: str) -> bool:
                name = str(name).lower()
                return ('_glcm_' in name) or ('_glrlm_' in name) or ('_glszm_' in name) or ('_gldm_' in name) or ('_ngtdm_' in name)
            numeric_cols = [c for c in numeric_cols if _is_texture(c)]
        X = radiomics_df[numeric_cols].values
        feature_names = np.array(numeric_cols)

    logger.info(f"Initial data shape: {X.shape}")
    logger.info(f"Class distribution: {np.bincount(y.astype(int))}")

    # Optional: restrict to a whitelist of features before any selection/evaluation
    if args.feature_whitelist:
        wl_path = Path(args.feature_whitelist)
        if not wl_path.exists():
            raise FileNotFoundError(f"Feature whitelist not found: {wl_path}")
        whitelist = [ln.strip() for ln in wl_path.read_text().splitlines() if ln.strip()]
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        keep_idx = [name_to_idx[n] for n in whitelist if n in name_to_idx]
        missing = [n for n in whitelist if n not in name_to_idx]
        if missing:
            logger.warning(f"Whitelist features not present in data and will be ignored: {missing}")
        if not keep_idx:
            raise ValueError("After applying feature whitelist, no features remain. Check names and input data.")
        X = X[:, keep_idx]
        feature_names = feature_names[keep_idx]
        logger.info(f"Applied feature whitelist: kept {len(keep_idx)} features. New shape: {X.shape}")
    
    # Setup preprocessing configuration
    preprocessing_config = {
        'use_variance_filter': True,
        'variance_threshold': args.variance_threshold,
        'use_low_info_filter': True,
        'max_nan_fraction': args.max_nan_fraction,
        'impute_strategy': args.impute_strategy,
        'use_scaler': not args.no_scaler
    }
    
    # Run enhanced parallel ensemble STABL
    logger.info("\nRunning Enhanced Parallel Ensemble STABL with preprocessing...")
    ensemble_stabl = EnhancedParallelEnsembleSTABL(
        n_runs=args.ensemble_runs,
        n_bootstraps=args.n_bootstraps,
        consensus_threshold=args.consensus_threshold,
        n_workers=args.n_workers,
        artificial_type=args.artificial_type,
        perc_corr_group_threshold=args.corr_group_threshold if not args.no_corr_grouping else None,
        lambda_grid=args.lambda_grid,
        n_lambda=args.n_lambda,
        fdr_start=args.fdr_start,
        fdr_end=args.fdr_end,
        fdr_step=args.fdr_step,
        use_preprocessing=args.use_preprocessing,
        preprocessing_config=preprocessing_config,
        c_value=args.c_value  # Pass Optuna-optimized C value
    )
    
    ensemble_stabl.fit(X, y, feature_names)
    
    # Get selected features
    selected_features = ensemble_stabl.selected_features_
    
    if args.n_features and len(selected_features) > args.n_features:
        # Sort by frequency and take top N
        freq_sorted = sorted(ensemble_stabl.feature_freq_.items(), 
                           key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in freq_sorted[:args.n_features]]
    
    logger.info(f"\nSelected {len(selected_features)} features after ensemble consensus")
    
    # Optional temporal holdout evaluation (train on oldest, test on newest by StudyDate)
    if args.temporal_holdout:
        try:
            if not args.scanner_metadata_path:
                raise ValueError('--scanner-metadata-path is required when --temporal-holdout is set')
            meta = pd.read_csv(args.scanner_metadata_path)
            if args.scanner_id_col not in meta.columns:
                raise ValueError(f"Scanner metadata id column '{args.scanner_id_col}' not found")
            if args.date_col not in meta.columns:
                raise ValueError(f"Scanner metadata date column '{args.date_col}' not found")
            if id_col is None:
                raise ValueError('No suitable ID column in radiomics to merge with scanner metadata')

            df_ids = radiomics_df[[id_col]].copy()
            df_ids['_row'] = np.arange(len(df_ids))
            merged = df_ids.merge(meta[[args.scanner_id_col, args.date_col]],
                                  left_on=id_col, right_on=args.scanner_id_col, how='left')
            merged[args.date_col] = pd.to_datetime(merged[args.date_col], errors='coerce')
            valid = merged.dropna(subset=[args.date_col]).sort_values(args.date_col).reset_index(drop=True)
            if valid.empty:
                raise ValueError('No valid StudyDate after merge; check id/date columns')
            n = len(valid)
            n_train = int(np.floor((1.0 - args.holdout_fraction) * n))
            train_rows = valid.iloc[:n_train]['_row'].astype(int).to_numpy()
            test_rows = valid.iloc[n_train:]['_row'].astype(int).to_numpy()

            X_tr, y_tr = X[train_rows], y[train_rows]
            X_te, y_te = X[test_rows], y[test_rows]
            fn = np.array(feature_names)

            logger.info(f"Temporal split (by {args.date_col}): train={len(train_rows)}, test={len(test_rows)}")
            logger.info(f"Train class distribution: {np.bincount(y_tr.astype(int))}")
            logger.info(f"Test class distribution: {np.bincount(y_te.astype(int))}")

            # Train-only STABL selection on temporal train
            stabl_train = EnhancedParallelEnsembleSTABL(
                n_runs=args.ensemble_runs,
                n_bootstraps=args.n_bootstraps,
                consensus_threshold=args.consensus_threshold,
                n_workers=args.n_workers,
                artificial_type=args.artificial_type,
                perc_corr_group_threshold=args.corr_group_threshold if not args.no_corr_grouping else None,
                use_preprocessing=args.use_preprocessing,
                preprocessing_config=preprocessing_config,
                c_value=args.c_value,
                lambda_grid=args.lambda_grid,
                n_lambda=args.n_lambda,
                fdr_start=args.fdr_start,
                fdr_end=args.fdr_end,
                fdr_step=args.fdr_step,
                hard_threshold=args.hard_threshold
            )
            stabl_train.fit(X_tr, y_tr, fn)
            sel_feats = stabl_train.selected_features_ or []
            if args.n_features and len(sel_feats) > args.n_features:
                freq_sorted = sorted(stabl_train.feature_freq_.items(), key=lambda x: x[1], reverse=True)
                sel_feats = [f for f, _ in freq_sorted[:args.n_features]]

            if stabl_train.use_preprocessing:
                X_tr_pre = stabl_train.preprocessor_.transform(X_tr)
                X_te_pre = stabl_train.preprocessor_.transform(X_te)
                pre_names = stabl_train.preprocessed_feature_names_
                idx = [i for i, f in enumerate(pre_names) if f in sel_feats]
                X_tr_sel = X_tr_pre[:, idx]
                X_te_sel = X_te_pre[:, idx]
            else:
                idx = [i for i, f in enumerate(fn) if f in sel_feats]
                X_tr_sel = X_tr[:, idx]
                X_te_sel = X_te[:, idx]

            # Build model for temporal eval
            if args.model == 'lr':
                clf = LogisticRegression(class_weight='balanced', max_iter=5000)
            elif args.model in ('enlr', 'en'):
                clf = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=5000,
                                         class_weight='balanced', l1_ratio=float(args.en_l1_ratio))
            elif args.model == 'rf':
                clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            elif args.model == 'svm':
                clf = SVC(probability=True, class_weight='balanced', random_state=42)
            elif args.model == 'xgb' and HAS_ADVANCED_MODELS:
                clf = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            elif args.model == 'lgb' and HAS_ADVANCED_MODELS:
                clf = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbosity=-1)
            else:
                clf = LogisticRegression(class_weight='balanced', max_iter=5000)

            clf.fit(X_tr_sel, y_tr)
            try:
                prob = clf.predict_proba(X_te_sel)[:, 1]
            except Exception:
                prob = clf.decision_function(X_te_sel)

            auc_val = float(roc_auc_score(y_te, prob)) if len(np.unique(y_te)) == 2 else None
            from sklearn.utils import resample
            boot = []
            rng = np.random.RandomState(42)
            idx_all = np.arange(len(y_te))
            for _ in range(2000):
                idx_bs = resample(idx_all, replace=True, random_state=rng)
                yt = y_te[idx_bs]
                pt = prob[idx_bs]
                if len(np.unique(yt)) == 2:
                    boot.append(roc_auc_score(yt, pt))
            ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None

            temporal = {
                'train_size': int(len(train_rows)),
                'test_size': int(len(test_rows)),
                'selected_features': sel_feats,
                'auc': auc_val,
                'ci': ci,
                'date_col': args.date_col,
                'holdout_fraction': args.holdout_fraction,
            }
            with open(output_dir / 'temporal_holdout.json', 'w') as f:
                json.dump(temporal, f, indent=2)
            logger.info(f"Temporal holdout AUC: {auc_val:.3f} CI: {ci if ci else 'NA'}")
        except Exception as e:
            logger.error(f"Temporal holdout evaluation failed: {e}")

    # Prepare data for evaluation
    # For unbiased evaluation, we will pass full X to the validator and perform
    # train-only selection inside each split (controlled by --cv-nested-selection).
    # The globally selected_features are still reported for interpretability.
    
    # Setup model
    if args.model == 'lr':
        if args.use_firth:
            logger.info("Using Firth's penalized logistic regression")
            model = FirthLogisticRegression(class_weight='balanced')
        else:
            model = LogisticRegression(class_weight='balanced', max_iter=5000)
    elif args.model in ('enlr', 'en'):
        logger.info("Using Elastic-Net Logistic Regression (saga)")
        # Initial model; if --optimize, a grid search will refine C/l1_ratio
        model = LogisticRegression(
            penalty='elasticnet', solver='saga', max_iter=5000,
            class_weight='balanced', l1_ratio=float(args.en_l1_ratio)
        )
    elif args.model == 'rf':
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    elif args.model == 'xgb' and HAS_ADVANCED_MODELS:
        model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    elif args.model == 'lgb' and HAS_ADVANCED_MODELS:
        model = LGBMClassifier(n_estimators=100, class_weight='balanced', random_state=42, verbosity=-1)
    elif args.model == 'svm':
        model = SVC(probability=True, class_weight='balanced', random_state=42)
    elif args.model == 'ensemble':
        estimators = [
            ('lr', LogisticRegression(class_weight='balanced', max_iter=5000)),
            ('rf', RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42))
        ]
        if HAS_ADVANCED_MODELS:
            estimators.append(('xgb', XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')))
        model = VotingClassifier(estimators=estimators, voting='soft')
    else:
        model = LogisticRegression(class_weight='balanced', max_iter=5000)
    
    # Run comprehensive validation
    logger.info("\nRunning Enhanced Multi-Method Validation...")
    # Config for STABL to reuse inside nested MC (train-only)
    nested_stabl_params = {
        'n_runs': args.ensemble_runs,
        'n_bootstraps': args.n_bootstraps,
        'consensus_threshold': args.consensus_threshold,
        'n_workers': args.n_workers,
        'artificial_type': args.artificial_type,
        'perc_corr_group_threshold': args.corr_group_threshold if not args.no_corr_grouping else None,
        'lambda_grid': args.lambda_grid,
        'n_lambda': args.n_lambda,
        'fdr_start': args.fdr_start,
        'fdr_end': args.fdr_end,
        'fdr_step': args.fdr_step,
        'use_preprocessing': args.use_preprocessing,
        'preprocessing_config': preprocessing_config,
        'c_value': args.c_value,
        'hard_threshold': args.hard_threshold,
        'random_state': 42
    }

    validator = MultiMethodValidator(
        n_bootstrap=args.validation_bootstrap,
        optimize_model=args.optimize,
        fixed_c=args.c_value,
        use_firth=args.use_firth,
        smooth_ci=args.smooth_ci,
        use_ensemble=args.use_ensemble_ci,
        use_nested_selection=args.cv_nested_selection,
        run_nested_mc=False,  # Run nested MC separately on full X to avoid leakage
        stabl_params=nested_stabl_params,
        cv_splits=args.cv_splits,
        cv_repeats=args.cv_repeats,
        cv_tune_lr=args.cv_tune_lr,
        cv_tune_trials=args.cv_tune_trials,
        cv_tune_inner_splits=args.cv_tune_inner_splits,
        cv_tune_seed=args.cv_tune_seed,
        lr_c_min=args.lr_c_min,
        lr_c_max=args.lr_c_max,
        en_l1_min=args.en_l1_min,
        en_l1_max=args.en_l1_max,
        nested_stabl_bags=args.nested_stabl_bags,
        nested_add_ridge=args.nested_add_ridge,
        nested_add_firth=args.nested_add_firth,
        nested_add_xgb=args.nested_add_xgb
    )
    
    # Mark that preprocessing has been applied (for validator awareness)
    if args.use_preprocessing and not args.no_scaler:
        validator._preprocessing_applied = True
    
    # Optionally restrict evaluation to consensus features (optimistic) when nested selection is off
    if (not args.cv_nested_selection) and args.eval_consensus_only:
        if selected_features:
            if args.use_preprocessing:
                X_pre = ensemble_stabl.preprocessor_.transform(X)
                pre_names = ensemble_stabl.preprocessed_feature_names_
                col_idx = [i for i, f in enumerate(pre_names) if f in selected_features]
                X_eval = X_pre[:, col_idx]
                eval_feature_names = np.array([pre_names[i] for i in col_idx])
            else:
                col_idx = [i for i, f in enumerate(feature_names) if f in selected_features]
                X_eval = X[:, col_idx]
                eval_feature_names = feature_names[col_idx]
        else:
            X_eval = X
            eval_feature_names = feature_names
    else:
        X_eval = X
        eval_feature_names = feature_names

    validation_results = validator.evaluate_all_methods(
        model, X_eval, y, feature_names=eval_feature_names, methods=args.val_methods
    )

    # Optional: nested Monte Carlo on full feature space (selection inside folds)
    if args.nested_mc:
        logger.info("\nRunning Nested Monte Carlo evaluation (outer holdout, inner selection)...")
        nested_results = validator.nested_monte_carlo_evaluation(
            model, X, y, feature_names=feature_names,
            n_outer=args.nested_outer,
            test_size=args.nested_test_size,
            inner_cv=args.nested_inner_cv,
            random_state=42
        )
        validation_results['nested_monte_carlo'] = nested_results
    
    # Create reports
    reporter = EnhancedReporting()
    
    # Create feature report
    feature_report = reporter.create_feature_report(
        selected_features, 
        ensemble_stabl.feature_freq_ or {},
        args.consensus_threshold
    )
    print(feature_report)
    logger.info(feature_report)
    
    # Print comparison table
    comparison_table = reporter.create_comparison_table(validation_results)
    print(comparison_table)
    logger.info(comparison_table)
    
    # Print CI methods comparison
    ci_comparison = reporter.create_ci_comparison(validation_results)
    if ci_comparison:
        print(ci_comparison)
        logger.info(ci_comparison)
    
    # Create visualizations
    plot_paths = reporter.create_performance_plots(
        validation_results,
        selected_features,
        ensemble_stabl.feature_freq_,
        output_dir
    )
    if plot_paths is None:
        plot_paths = []
    # Also create ROC curve: prefer pooled OOF from repeated-CV, else LOOCV
    roc_path = None
    try:
        from utils.plotting_utils import plot_single_roc
    except Exception:
        plot_single_roc = None
    try:
        roc_base = Path(output_dir) / 'roc_curve'
        # Determine label AUC based on user choice
        label_auc = None
        label_ci = None
        # Gather available metrics
        cv_mean = validation_results.get('repeated_stratified_cv', {}).get('auc') if validation_results.get('repeated_stratified_cv') else None
        cv_oof_auc = validation_results.get('repeated_stratified_cv', {}).get('oof', {}).get('auc') if validation_results.get('repeated_stratified_cv') else None
        cv_oof_ci = validation_results.get('repeated_stratified_cv', {}).get('oof', {}).get('ci') if validation_results.get('repeated_stratified_cv') else None
        auc632 = validation_results.get('bootstrap_632_plus', {}).get('auc') if validation_results.get('bootstrap_632_plus') else None
        auc632_ci = validation_results.get('bootstrap_632_plus', {}).get('ci') if validation_results.get('bootstrap_632_plus') else None

        choice = getattr(args, 'roc_label_source', 'oof')
        if choice == 'cv-mean' and cv_mean is not None:
            label_auc = float(cv_mean)
        elif choice == '632' and auc632 is not None:
            label_auc = float(auc632); label_ci = auc632_ci
        elif choice == 'best':
            cands = []
            if cv_mean is not None: cands.append(('cv-mean', float(cv_mean), None))
            if cv_oof_auc is not None: cands.append(('oof', float(cv_oof_auc), cv_oof_ci))
            if auc632 is not None: cands.append(('632', float(auc632), auc632_ci))
            if cands:
                best_name, best_auc, best_ci = max(cands, key=lambda x: x[1])
                label_auc, label_ci = best_auc, best_ci
        else:
            # default to OOF
            if cv_oof_auc is not None:
                label_auc = float(cv_oof_auc); label_ci = cv_oof_ci

        def _save_roc_plot(y_true, y_pred, base_path, auc_value, auc_ci):
            from sklearn.metrics import roc_curve

            try:
                plot_single_roc(y_true, y_pred, base_path, n_boot=2000, label_auc=auc_value, label_ci=auc_ci)
                return
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(8, 8))
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ax.plot(fpr, tpr, color='tab:blue', linewidth=2, label=f'ROC (AUC = {auc_value:.3f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='tab:red', alpha=0.6)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.legend(loc='lower right')
            fig.savefig(base_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
            fig.savefig(base_path.with_suffix('.pdf'), bbox_inches='tight')
            plt.close(fig)

        if 'repeated_stratified_cv' in validation_results and validation_results['repeated_stratified_cv']:
            rep = validation_results['repeated_stratified_cv']
            oof = rep.get('oof') if rep else None
            if oof and isinstance(oof.get('y_true'), list) and isinstance(oof.get('y_pred'), list) and len(oof['y_true']) == len(oof['y_pred']):
                y_true = np.asarray(oof['y_true'], dtype=int)
                y_pred = np.asarray(oof['y_pred'], dtype=float)
                if len(np.unique(y_true)) == 2:
                    auc_oof = float(roc_auc_score(y_true, y_pred))
                    if label_auc is None:
                        label_auc = auc_oof
                else:
                    auc_oof = float('nan')
                _save_roc_plot(y_true, y_pred, roc_base, label_auc if label_auc is not None else auc_oof, label_ci)
                roc_path = str(roc_base.with_suffix('.png'))
                logger.info(f"ROC curve (OOF repeated-CV) saved to {roc_path}")
        elif 'loocv' in validation_results and validation_results['loocv']:
            lo = validation_results['loocv']
            if isinstance(lo.get('y_true'), list) and isinstance(lo.get('y_pred'), list) and len(lo['y_true']) == len(lo['y_pred']):
                y_true = np.asarray(lo['y_true'], dtype=int)
                y_pred = np.asarray(lo['y_pred'], dtype=float)
                if len(np.unique(y_true)) == 2:
                    auc_lo = float(roc_auc_score(y_true, y_pred))
                    if label_auc is None:
                        label_auc = auc_lo
                else:
                    auc_lo = float('nan')
                _save_roc_plot(y_true, y_pred, roc_base, label_auc if label_auc is not None else auc_lo, label_ci)
                roc_path = str(roc_base.with_suffix('.png'))
                logger.info(f"ROC curve (LOOCV) saved to {roc_path}")
    except Exception as e:
        logger.warning(f"ROC plotting skipped due to error: {e}")

    if roc_path:
        plot_paths.append(roc_path)

    if plot_paths:
        logger.info("Visualization files: %s", ", ".join(plot_paths))

    # Optional: evaluate multiple models on the selected panel and plot ROC per model
    if args.eval_all_models:
        try:
            models = {}
            models['lr'] = LogisticRegression(class_weight='balanced', max_iter=5000)
            models['en'] = LogisticRegression(penalty='elasticnet', solver='saga', class_weight='balanced', max_iter=5000, l1_ratio=float(args.en_l1_ratio))
            models['rf'] = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
            models['svm'] = SVC(probability=True, class_weight='balanced', random_state=42)
            if HAS_ADVANCED_MODELS:
                models['xgb'] = XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='logloss', random_state=42)
                models['lgb'] = LGBMClassifier(n_estimators=300, class_weight='balanced', random_state=42, verbosity=-1)

            # Build panel matrices (selected, and optionally with extra K features by frequency)
            # Base: selected_features
            if args.use_preprocessing:
                X_pre = ensemble_stabl.preprocessor_.transform(X)
                pre_names = np.array(ensemble_stabl.preprocessed_feature_names_)
                sel_idx = [i for i, f in enumerate(pre_names) if f in selected_features]
                X_panel = X_pre[:, sel_idx] if sel_idx else X_pre[:, :0]
                panel_names = pre_names[sel_idx]
                # Extra K by frequency not in selected
                extra = []
                if args.extra_panel_k and ensemble_stabl.feature_freq_:
                    sorted_freq = sorted(ensemble_stabl.feature_freq_.items(), key=lambda x: x[1], reverse=True)
                    for fname, _ in sorted_freq:
                        if fname in selected_features:
                            continue
                        if fname in pre_names:
                            extra.append(fname)
                        if len(extra) >= int(args.extra_panel_k):
                            break
                if extra:
                    extra_idx = [i for i, f in enumerate(pre_names) if f in set(extra)]
                    X_panel_extra = np.concatenate([X_panel, X_pre[:, extra_idx]], axis=1)
                    panel_names_extra = np.concatenate([panel_names, pre_names[extra_idx]])
                else:
                    X_panel_extra, panel_names_extra = X_panel, panel_names
            else:
                name_arr = np.array(feature_names)
                sel_idx = [i for i, f in enumerate(name_arr) if f in selected_features]
                X_panel = X[:, sel_idx] if sel_idx else X[:, :0]
                panel_names = name_arr[sel_idx]
                extra = []
                if args.extra_panel_k and ensemble_stabl.feature_freq_:
                    sorted_freq = sorted(ensemble_stabl.feature_freq_.items(), key=lambda x: x[1], reverse=True)
                    for fname, _ in sorted_freq:
                        if fname in selected_features:
                            continue
                        if fname in name_arr:
                            extra.append(fname)
                        if len(extra) >= int(args.extra_panel_k):
                            break
                if extra:
                    extra_idx = [i for i, f in enumerate(name_arr) if f in set(extra)]
                    X_panel_extra = np.concatenate([X_panel, X[:, extra_idx]], axis=1)
                    panel_names_extra = np.concatenate([panel_names, name_arr[extra_idx]])
                else:
                    X_panel_extra, panel_names_extra = X_panel, panel_names

            # Evaluate per model using LOOCV
            mm_dir = Path(output_dir) / 'multi_model_eval'
            mm_dir.mkdir(exist_ok=True)
            results_models = {}
            for mname, mobj in models.items():
                X_use = X_panel_extra if (args.extra_panel_k and mname in ('rf', 'svm', 'xgb', 'lgb')) else X_panel
                if X_use.shape[1] == 0:
                    continue
                res = _loocv_predictions_for_model(mobj, X_use, y)
                entry = {
                    'auc': float(res['auc']) if res['auc'] is not None else None,
                    'ci': res['ci'],
                    'panel_size': int(X_use.shape[1])
                }
                results_models[mname] = entry
                # Plot ROC per model
                try:
                    from utils.plotting_utils import plot_single_roc
                    plot_single_roc(res['y_true'], res['y_pred'], mm_dir / f'roc_{mname}', n_boot=1000)
                except Exception:
                    pass

            if results_models:
                df_rows = []
                for mname, info in results_models.items():
                    ci = info.get('ci') or [float('nan'), float('nan')]
                    df_rows.append({
                        'model': mname.upper(),
                        'auc': info.get('auc'),
                        'ci_lower': ci[0],
                        'ci_upper': ci[1],
                        'panel_size': info.get('panel_size')
                    })
                model_df = pd.DataFrame(df_rows).set_index('model').sort_values('auc', ascending=False)
                model_df.to_csv(mm_dir / 'summary.csv')
                try:
                    model_df.to_markdown(mm_dir / 'summary.md')
                except Exception:
                    model_df.to_csv(mm_dir / 'summary.md', sep='\t')

                # Heatmap-style comparison of AUCs
                heatmap_data = model_df[['auc']].copy()
                try:
                    if HAS_BEAUTIFUL:
                        fig, ax = create_beautiful_figure('tall')
                    else:
                        fig, ax = plt.subplots(figsize=(6, max(4, 0.6 * len(heatmap_data))))
                    colors = [
                        NORD_COLORS.get('nord6', '#ECEFF4'),
                        NORD_COLORS.get('nord8', '#88C0D0'),
                        NORD_COLORS.get('nord9', '#81A1C1'),
                        NORD_COLORS.get('nord10', '#5E81AC'),
                        NORD_COLORS.get('nord11', '#BF616A'),
                    ]
                    cmap = sns.color_palette(colors, as_cmap=True)
                    sns.heatmap(
                        heatmap_data,
                        annot=True,
                        fmt='.3f',
                        cmap=cmap,
                        vmin=0.4,
                        vmax=1.0,
                        cbar_kws={'label': 'AUC'},
                        ax=ax
                    )
                    ax.set_title('Model AUC Comparison (LOOCV)')
                    ax.set_ylabel('Model')
                    ax.set_xlabel('Metric')
                    heatmap_path = mm_dir / 'model_auc_heatmap'
                    if HAS_BEAUTIFUL:
                        save_beautiful_figure(fig, heatmap_path)
                    else:
                        fig.savefig(heatmap_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
                        fig.savefig(heatmap_path.with_suffix('.pdf'), bbox_inches='tight')
                    plt.close(fig)
                    plot_paths.append(str(heatmap_path.with_suffix('.png')))
                except Exception as heatmap_err:
                    logger.warning(f"Model comparison heatmap skipped: {heatmap_err}")

                # Bar comparison using legacy style
                try:
                    from utils.plotting_utils import plot_validation_comparison
                    plot_validation_comparison(results_models, mm_dir / 'models_comparison')
                    plot_paths.append(str((mm_dir / 'models_comparison').with_suffix('.png')))
                except Exception:
                    pass

                # Save JSON for downstream use
                with open(mm_dir / 'summary.json', 'w') as f:
                    json.dump(results_models, f, indent=2)
                logger.info(
                    "Multi-model evaluation saved under %s (table: summary.csv, heatmap: model_auc_heatmap.png)",
                    mm_dir
                )
        except Exception as e:
            logger.warning(f"Multi-model evaluation skipped: {e}")

    permutation_results = None
    if args.permutation_test:
        if X_eval.shape[1] == 0:
            logger.warning("Permutation test skipped: no features available on the frozen panel")
        else:
            observed_auc = None
            observed_ci = None
            if validation_results.get('repeated_stratified_cv'):
                rep = validation_results['repeated_stratified_cv'] or {}
                oof = rep.get('oof') or {}
                if oof.get('auc') is not None:
                    observed_auc = float(oof['auc'])
                    observed_ci = oof.get('ci')
                elif rep.get('auc') is not None:
                    observed_auc = float(rep['auc'])
                    observed_ci = rep.get('ci')
            if observed_auc is None and validation_results.get('loocv'):
                lo = validation_results['loocv'] or {}
                if lo.get('auc') is not None:
                    observed_auc = float(lo['auc'])
                    observed_ci = lo.get('ci')
            if observed_auc is None and validation_results.get('bootstrap_632_plus'):
                bs = validation_results['bootstrap_632_plus'] or {}
                if bs.get('auc') is not None:
                    observed_auc = float(bs['auc'])
                    observed_ci = bs.get('ci')

            if observed_auc is None:
                logger.warning("Permutation test skipped: unable to determine observed AUC")
            else:
                logger.info(
                    "Running permutation test (%d iterations, test_size=%.2f) on frozen panel",
                    args.permutation_iterations,
                    args.permutation_test_size,
                )
                permuted_aucs = _permutation_test_auc(
                    model,
                    X_eval,
                    y,
                    n_permutations=args.permutation_iterations,
                    test_size=args.permutation_test_size,
                    random_state=args.permutation_random_state,
                )
                if permuted_aucs:
                    permuted_aucs = np.array(permuted_aucs, dtype=float)
                    p_value = float((np.sum(permuted_aucs >= observed_auc) + 1) / (len(permuted_aucs) + 1))
                    permutation_results = {
                        'observed_auc': observed_auc,
                        'observed_ci': observed_ci,
                        'n_permutations': int(len(permuted_aucs)),
                        'p_value': p_value,
                        'significant': bool(p_value < 0.05),
                        'null_mean': float(np.mean(permuted_aucs)),
                        'null_std': float(np.std(permuted_aucs)),
                    }

                    perm_dir = Path(output_dir)
                    perm_plot_base = perm_dir / 'permutation_distribution'
                    try:
                        if HAS_BEAUTIFUL:
                            fig, ax = create_beautiful_figure('wide')
                        else:
                            fig, ax = plt.subplots(figsize=(10, 6))
                        bins = min(40, max(10, len(permuted_aucs) // 25))
                        ax.hist(
                            permuted_aucs,
                            bins=bins,
                            color=NORD_COLORS.get('nord9', '#5E81AC') if HAS_BEAUTIFUL else 'tab:blue',
                            edgecolor=NORD_COLORS.get('nord3', '#4C566A') if HAS_BEAUTIFUL else 'black',
                            alpha=0.85,
                        )
                        ax.axvline(observed_auc, color=NORD_COLORS.get('nord11', '#BF616A') if HAS_BEAUTIFUL else 'tab:red',
                                   linestyle='--', linewidth=2, label=f'Observed AUC = {observed_auc:.3f}')
                        ax.set_xlabel('Permuted AUC')
                        ax.set_ylabel('Frequency')
                        ax.set_title('Permutation Test Null Distribution')
                        ax.legend(loc='upper right')
                        if HAS_BEAUTIFUL:
                            save_beautiful_figure(fig, perm_plot_base)
                        else:
                            fig.savefig(perm_plot_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
                            fig.savefig(perm_plot_base.with_suffix('.pdf'), bbox_inches='tight')
                        plt.close(fig)
                        plot_paths.append(str(perm_plot_base.with_suffix('.png')))
                    except Exception as perm_plot_err:
                        logger.warning(f"Permutation histogram skipped: {perm_plot_err}")

                    with open(perm_dir / 'permutation_test.json', 'w') as f:
                        json.dump({
                            **permutation_results,
                            'null_distribution': permuted_aucs.tolist(),
                            'settings': {
                                'iterations': args.permutation_iterations,
                                'test_size': args.permutation_test_size,
                                'random_state': args.permutation_random_state,
                                'model': args.model,
                            }
                        }, f, indent=2)
                    logger.info(
                        "Permutation test completed (p-value = %.4f, iterations = %d)",
                        p_value,
                        len(permuted_aucs)
                    )
                else:
                    logger.warning("Permutation test failed: no valid permutations completed")

    # Attach OOF IDs to validation_results for robust downstream alignment
    try:
        rep = validation_results.get('repeated_stratified_cv', {}) or {}
        oof = rep.get('oof') or {}
        oof_index = oof.get('index')
        # Determine the ID column we used
        id_series = None
        if 'scanner_patient_name' in radiomics_df.columns:
            id_series = radiomics_df['scanner_patient_name'].astype(str)
        elif 'patient_name' in radiomics_df.columns:
            id_series = radiomics_df['patient_name'].astype(str)
        elif 'patient_id' in radiomics_df.columns:
            id_series = radiomics_df['patient_id'].astype(str)
        if id_series is not None and isinstance(oof_index, list) and len(oof_index) == len(oof.get('y_true', [])):
            oof_ids = id_series.values[np.array(oof_index, dtype=int)].tolist()
            validation_results['repeated_stratified_cv']['oof']['ids'] = oof_ids
    except Exception as _e:
        # Non-fatal; skip if anything goes wrong
        pass

    # Save results
    results = {
        'selected_features': selected_features,
        'feature_frequencies': ensemble_stabl.feature_freq_,
        'validation_results': validation_results,
        'comparison_table': comparison_table,
        'ci_comparison': ci_comparison,
        'feature_report': feature_report,
        'permutation_test': permutation_results,
        'preprocessing_config': preprocessing_config,
        'args': vars(args),
        'v3_enhancements': {
            'preprocessing_applied': args.use_preprocessing,
            'artificial_type': args.artificial_type,
            'correlation_grouping': not args.no_corr_grouping,
            'features_before_preprocessing': len(feature_names),
            'features_after_preprocessing': len(ensemble_stabl.preprocessed_feature_names_) if ensemble_stabl.preprocessed_feature_names_ else len(feature_names)
        }
    }
    
    # Save JSON
    with open(output_dir / 'results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, dict):
                json_results[k] = {str(kk): vv.tolist() if isinstance(vv, np.ndarray) else vv 
                                  for kk, vv in v.items()}
            else:
                json_results[k] = v
        json.dump(json_results, f, indent=2, default=str)
    
    # Save text report
    with open(output_dir / 'report.txt', 'w') as f:
        f.write("POPF STABL V3 ENHANCED RESULTS (FDP+ ENABLED)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Selected features: {len(selected_features)}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Optimization: {'Yes' if args.optimize else 'No'}\n")
        f.write(f"C value: {args.c_value if args.c_value else 'GridSearchCV'}\n")
        f.write(f"Preprocessing: {args.use_preprocessing}\n")
        f.write(f"Artificial type: {args.artificial_type}\n")
        f.write(f"Correlation grouping: {not args.no_corr_grouping}\n")
        f.write(f"FDP+ threshold selection: ENABLED (hard_threshold=None)\n\n")
        f.write(feature_report)
        f.write("\n" + comparison_table)
        if ci_comparison:
            f.write("\n" + ci_comparison)
        if permutation_results:
            f.write("\nPERMUTATION TEST\n")
            f.write("-" * 80 + "\n")
            f.write(f"Observed AUC: {permutation_results['observed_auc']:.3f}\n")
            if permutation_results.get('observed_ci'):
                lo, hi = permutation_results['observed_ci']
                f.write(f"Observed CI: [{lo:.3f}, {hi:.3f}]\n")
            f.write(f"Null mean: {permutation_results['null_mean']:.3f} (std {permutation_results['null_std']:.3f})\n")
            f.write(f"Permutations: {permutation_results['n_permutations']}\n")
            f.write(f"p-value: {permutation_results['p_value']:.4f}\n")
            f.write(
                f"Significant (alpha=0.05): {'Yes' if permutation_results['significant'] else 'No'}\n"
            )

    # Save markdown report
    with open(output_dir / 'report.md', 'w') as f:
        f.write("# POPF STABL V3 Enhanced Results\n\n")
        f.write(f"**Date**: {datetime.now()}\n\n")
        f.write("## Configuration\n")
        f.write(f"- Selected features: {len(selected_features)}\n")
        f.write(f"- Model: {args.model}\n")
        f.write(f"- Ensemble runs: {args.ensemble_runs}\n")
        f.write(f"- Bootstrap iterations: {args.validation_bootstrap}\n")
        f.write(f"- Hyperparameter optimization: {'Yes' if args.optimize else 'No'}\n")
        f.write(f"- **Preprocessing**: {args.use_preprocessing}\n")
        f.write(f"- **Artificial type**: {args.artificial_type}\n")
        f.write(f"- **Correlation grouping**: {not args.no_corr_grouping}\n\n")
        f.write(f"- **FDR range**: start={args.fdr_start}, end={args.fdr_end}, step={args.fdr_step}\n")
        f.write(f"- **Lambda grid**: {args.lambda_grid} (n_lambda={args.n_lambda})\n")
        f.write("## V3 Enhancements\n")
        if results['v3_enhancements']:
            enh = results['v3_enhancements']
            f.write(f"- Features before preprocessing: {enh['features_before_preprocessing']}\n")
            f.write(f"- Features after preprocessing: {enh['features_after_preprocessing']}\n")
            f.write(f"- Reduction: {enh['features_before_preprocessing'] - enh['features_after_preprocessing']} features removed\n\n")
        f.write("## Selected Features\n```\n")
        for i, feat in enumerate(selected_features, 1):
            freq = ensemble_stabl.feature_freq_.get(feat, 0)
            f.write(f"{i:3d}. {feat:<60} (freq: {freq:.2f})\n")
        f.write("```\n\n")
        f.write("## Performance Results\n```\n")
        f.write(comparison_table)
        f.write("```\n")
        if permutation_results:
            f.write("\n## Permutation Test\n")
            f.write(f"- Observed AUC: {permutation_results['observed_auc']:.3f}\n")
            if permutation_results.get('observed_ci'):
                lo, hi = permutation_results['observed_ci']
                f.write(f"- Observed CI: [{lo:.3f}, {hi:.3f}]\n")
            f.write(
                f"- Null mean ± std: {permutation_results['null_mean']:.3f} ± {permutation_results['null_std']:.3f}\n"
            )
            f.write(f"- Permutations: {permutation_results['n_permutations']}\n")
            f.write(f"- p-value: {permutation_results['p_value']:.4f}\n")
            f.write(
                f"- Significant (alpha=0.05): {'Yes' if permutation_results['significant'] else 'No'}\n"
            )

    logger.info(f"\nResults saved to {output_dir}")
    logger.info("V3 Pipeline completed successfully!")


if __name__ == '__main__':
    main()
