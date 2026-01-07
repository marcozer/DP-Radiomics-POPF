#!/usr/bin/env python3
"""
Ultra-Optimized STABL (Discovery-Oriented)
=========================================
This script is positioned as a discovery tool to generate candidate
feature panels and selection frequencies, aligned with the repository
policy:

- Selection is performed once on the full dataset (discovery) or on the
  training set only in temporal holdout mode; the resulting panel is
  frozen and evaluated elsewhere (V3 fixed-panel repeated-CV, temporal
  holdout, etc.).
- Internal AUC estimates produced here are exploratory/optimistic and
  MUST NOT be headlined for publication. Use V3 fixed-panel evaluation
  for primary performance reporting.

Additions in this version:
- Exact POPF alignment using `data/POPF-SCANNER.csv` (drop unmatched rows).
- Configurable positive grades (e.g., B,C or B,C,BL).
- Optional normalized ID fallback join (off by default).
- Discovery-only mode exporting frozen panels and frequencies.
- Optional temporal holdout evaluation: train on oldest subset (by
  StudyDate) and evaluate on the newest subset, with train-only
  selection and frozen panel.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Tuple, List, Dict, Optional, Any, Union
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime
import joblib
from tqdm import tqdm
import json
import pickle

# Core ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.utils import resample
import xgboost as xgb
import lightgbm as lgb

# Import plotting utilities
import sys

# Ensure code directory (where utils/ lives) is importable
SCRIPT_DIR = Path(__file__).resolve()
CODE_DIR = SCRIPT_DIR.parent.parent  # .../code
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))
try:
    from utils.plotting_utils import (
        setup_plotting, plot_model_comparison_with_ci, 
        plot_roc_with_confidence_band, save_beautiful_figure,
        create_beautiful_figure, NORD_COLORS, COLOR_SCHEMES
    )
    setup_plotting()
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    sns.set_theme(style='darkgrid')
    print("Warning: plotting_utils not available, using default matplotlib")

# STABL imports
try:
    from stabl.stabl import Stabl
    STABL_AVAILABLE = True
except ImportError:
    print("Warning: STABL not installed. Install with: pip install git+https://github.com/gregbellan/Stabl.git@v1.0.1-lw")
    STABL_AVAILABLE = False

warnings.filterwarnings('ignore')


# ---------- ComBat Harmonization (train-only fit) ----------
def _combat_train_only(X_tr, X_te, batch_tr, batch_te, verbose=False):
    """Apply ComBat harmonization fitting on training only, then transform test.

    Attempts to use neurocombat_sklearn.CombatModel if available; otherwise
    falls back to a per-batch location/scale standardization re-centered to the
    global training distribution (not EB, but leakage-safe and robust).

    Parameters
    - X_tr, X_te: np.ndarray (n_samples, n_features)
    - batch_tr, batch_te: array-like of batch labels per sample
    """
    # Try neurocombat_sklearn first
    try:
        from neurocombat_sklearn import CombatModel
        model = CombatModel()
        X_tr_h = model.fit_transform(X_tr, batch_tr)
        X_te_h = model.transform(X_te, batch_te)
        if verbose:
            print("Applied ComBat (neurocombat_sklearn)")
        return X_tr_h, X_te_h
    except Exception:
        pass
    # Fallback: per-batch z-score to global train distribution
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    batch_tr = np.asarray(batch_tr)
    batch_te = np.asarray(batch_te)
    g_mean = np.nanmean(X_tr, axis=0)
    g_std = np.nanstd(X_tr, axis=0)
    g_std[g_std < 1e-12] = 1.0
    batches = np.unique(batch_tr)
    batch_mu = {}
    batch_sd = {}
    for b in batches:
        idx = (batch_tr == b)
        mu = np.nanmean(X_tr[idx], axis=0)
        sd = np.nanstd(X_tr[idx], axis=0)
        sd[sd < 1e-12] = 1.0
        batch_mu[b] = mu
        batch_sd[b] = sd
    # Transform train
    X_tr_h = np.empty_like(X_tr)
    for b in batches:
        idx = (batch_tr == b)
        mu = batch_mu[b]; sd = batch_sd[b]
        X_tr_h[idx] = ((X_tr[idx] - mu) / sd) * g_std + g_mean
    # Transform test
    X_te_h = np.empty_like(X_te)
    for i in range(X_te.shape[0]):
        b = batch_te[i]
        mu = batch_mu.get(b, g_mean)
        sd = batch_sd.get(b, g_std)
        X_te_h[i] = ((X_te[i] - mu) / sd) * g_std + g_mean
    if verbose:
        print("Applied ComBat fallback (per-batch z-score to global train)")
    return X_tr_h, X_te_h

# ---------- Reference-based Standardization (train-only, ComBat-ref light) ----------
def _ref_standardize_train_only(X_tr, X_te, batch_tr, batch_te, ref_value, verbose=False):
    """Align per-batch feature distributions to a reference batch using
    train-only per-feature location/scale: x'=(x-mu_b)/sd_b*sd_ref+mu_ref.

    This is a light, leakage-safe alternative to ComBat-ref: no EB pooling,
    only per-batch standardization toward the reference batch statistics
    computed on the training set. For unseen test batches, uses reference stats.
    """
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    batch_tr = np.asarray(batch_tr)
    batch_te = np.asarray(batch_te)
    # Train stats per batch and reference stats
    batches = np.unique(batch_tr)
    ref_mask = (batch_tr == ref_value)
    if not np.any(ref_mask):
        # Fallback: use global training as reference
        ref_mu = np.nanmean(X_tr, axis=0)
        ref_sd = np.nanstd(X_tr, axis=0)
        if verbose:
            print("[ref-std] Reference batch not in training; using global train stats as reference")
    else:
        ref_mu = np.nanmean(X_tr[ref_mask], axis=0)
        ref_sd = np.nanstd(X_tr[ref_mask], axis=0)
    ref_sd[ref_sd < 1e-12] = 1.0
    batch_mu = {}
    batch_sd = {}
    for b in batches:
        m = (batch_tr == b)
        mu = np.nanmean(X_tr[m], axis=0)
        sd = np.nanstd(X_tr[m], axis=0)
        sd[sd < 1e-12] = 1.0
        batch_mu[b] = mu
        batch_sd[b] = sd
    # Transform train
    X_tr_h = np.empty_like(X_tr)
    for b in batches:
        m = (batch_tr == b)
        mu = batch_mu[b]; sd = batch_sd[b]
        X_tr_h[m] = ((X_tr[m] - mu) / sd) * ref_sd + ref_mu
    # Transform test (unseen batches -> use ref stats)
    X_te_h = np.empty_like(X_te)
    for i in range(X_te.shape[0]):
        b = batch_te[i]
        mu = batch_mu.get(b, ref_mu)
        sd = batch_sd.get(b, ref_sd)
        X_te_h[i] = ((X_te[i] - mu) / sd) * ref_sd + ref_mu
    if verbose:
        print(f"[ref-std] Aligned batches to reference='{ref_value}' (train-only stats)")
    return X_tr_h, X_te_h

# ---------- Helpers: LR C tuning via Optuna (with safe fallback) ----------
def _tune_lr_c_optuna(
    X_tr,
    y_tr,
    base_pipe,
    cv_splits=5,
    trials=30,
    seed=42,
    c_min=1e-3,
    c_max=1e2,
    grid_values=None,
):
    """Tune LogisticRegression C with Optuna using train-only CV.
    Returns (best_pipe, best_c, best_cv_auc). Falls back to GridSearchCV if Optuna not available.
    """
    try:
        import optuna
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        c_min = float(max(c_min, 1e-6))
        c_max = float(max(c_max, c_min * 10))

        def objective(trial):
            C = trial.suggest_float('C', c_min, c_max, log=True)
            pipe = clone(base_pipe)
            pipe.set_params(model__C=C)
            cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
            scores = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring='roc_auc', n_jobs=-1)
            return float(np.mean(scores)) if len(scores) else 0.5
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=int(trials), n_jobs=1)
        best_c = float(study.best_params.get('C', 1.0))
        best_cv_auc = float(study.best_value)
        best_pipe = clone(base_pipe)
        best_pipe.set_params(model__C=best_c)
        return best_pipe, best_c, best_cv_auc
    except Exception:
        # Fallback to GridSearchCV with a compact grid
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        if grid_values is not None and len(grid_values):
            grid = list(sorted(set(float(v) for v in grid_values if v > 0)))
        else:
            grid = list(np.geomspace(c_min, c_max, num=6))
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
        gs = GridSearchCV(clone(base_pipe), param_grid={'model__C': grid}, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        best_pipe = gs.best_estimator_
        best_c = float(gs.best_params_.get('model__C', 1.0))
        best_cv_auc = float(gs.best_score_)
        return best_pipe, best_c, best_cv_auc

# ---------- Unsupervised Prefilter Utilities (missing/variance/correlation) ----------
def _prefilter_unsupervised(X: np.ndarray,
                            feat_names: List[str],
                            missing_threshold: float = 0.3,
                            variance_threshold: float = 0.01,
                            correlation_threshold: float = 0.95) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """Apply unsupervised prefiltering: high-missing removal, low-variance removal, high-correlation pruning.

    Returns X_filtered, feat_names_filtered, stats dict.
    """
    fn = np.array(feat_names)
    stats = {'removed_missing': 0, 'removed_variance': 0, 'removed_correlation': 0}
    if X.size == 0 or fn.size == 0:
        return X, feat_names, stats

    # 1) Missing filtering (column-wise)
    miss_frac = np.mean(np.isnan(X), axis=0)
    keep_miss = miss_frac <= float(missing_threshold)
    stats['removed_missing'] = int((~keep_miss).sum())
    X1 = X[:, keep_miss]
    fn1 = fn[keep_miss]
    if X1.shape[1] == 0:
        return X1, fn1.tolist(), stats

    # Impute with per-feature median for downstream variance/correlation
    med = np.nanmedian(X1, axis=0)
    X_imp = np.where(np.isnan(X1), med, X1)

    # 2) Variance filtering
    var = np.var(X_imp, axis=0)
    keep_var = var > float(variance_threshold)
    stats['removed_variance'] = int((~keep_var).sum())
    X2 = X_imp[:, keep_var]
    fn2 = fn1[keep_var]
    var2 = var[keep_var]
    if X2.shape[1] == 0:
        return X2, fn2.tolist(), stats

    # 3) Correlation pruning (Spearman absolute)
    try:
        from scipy.stats import spearmanr
        corr = np.abs(spearmanr(X2)[0])
        # upper triangle
        triu = np.triu(np.ones_like(corr, dtype=bool), k=1)
        to_remove = set()
        # Greedy: for each high-corr pair, drop the lower-variance feature
        pairs = np.where((corr > float(correlation_threshold)) & triu)
        for i, j in zip(pairs[0], pairs[1]):
            if i in to_remove or j in to_remove:
                continue
            if var2[i] < var2[j]:
                to_remove.add(i)
            else:
                to_remove.add(j)
        if to_remove:
            keep_idx = [k for k in range(X2.shape[1]) if k not in to_remove]
            X3 = X2[:, keep_idx]
            fn3 = fn2[keep_idx]
            stats['removed_correlation'] = int(len(to_remove))
        else:
            X3, fn3 = X2, fn2
    except Exception:
        # If scipy not available, skip correlation pruning
        X3, fn3 = X2, fn2

    return X3, fn3.tolist(), stats

class EnsembleSTABL:
    """Ensemble STABL with multiple runs and consensus selection"""
    
    def __init__(self,
                 n_runs: int = 10,
                 threshold: float = 0.8,
                 n_bootstraps: int = 500,
                 regularization: str = 'l1',
                 alpha: float = 0.5,
                 max_features: int = 10,
                 consensus_threshold: float = 0.8,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 C: float = 1.0,
                 tol: float = 1e-3,
                 max_iter: int = 5000,
                 # Optional FDR / lambda grid controls
                 artificial_type: Optional[str] = None,
                 lambda_grid: Optional[Union[str, Dict[str, Any]]] = None,
                 n_lambda: Optional[int] = None,
                 fdr_start: float = 0.1,
                 fdr_end: float = 0.3,
                 fdr_step: float = 0.01,
                 perc_corr_group_threshold: Optional[float] = None,
                 sample_fraction: float = 0.75):
        """
        Parameters
        ----------
        n_runs : int
            Number of STABL runs with different seeds
        threshold : float
            STABL selection threshold per run
        n_bootstraps : int
            Number of bootstraps per STABL run
        regularization : str
            Type of regularization ('l1', 'adaptive_lasso')
        alpha : float
            Elastic net mixing parameter (0=ridge, 1=lasso)
        max_features : int
            Maximum number of features to select
        consensus_threshold : float
            Fraction of runs a feature must appear in
        n_jobs : int
            Number of CPU cores to use (-1 for all cores)
        random_state : int
            Base random seed
        """
        self.n_runs = n_runs
        self.threshold = threshold
        self.n_bootstraps = n_bootstraps
        self.regularization = regularization
        self.alpha = alpha
        self.max_features = max_features
        self.consensus_threshold = consensus_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        # FDR / lambda / grouping
        self.artificial_type = artificial_type
        self.lambda_grid = lambda_grid
        self.n_lambda = n_lambda
        self.fdr_start = fdr_start
        self.fdr_end = fdr_end
        self.fdr_step = fdr_step
        self.perc_corr_group_threshold = perc_corr_group_threshold
        self.sample_fraction = sample_fraction
        
        # Results storage
        self.run_results = []
        self.consensus_features = []
        self.feature_frequencies = Counter()
        self.stability_matrix = None
        
    def _get_base_estimator(self, seed: int) -> Any:
        """Get base estimator based on regularization type"""
        
        if self.regularization == 'l1':
            return LogisticRegression(
                penalty='l1', C=self.C, solver='liblinear',
                class_weight='balanced', max_iter=self.max_iter,
                random_state=seed
            )
        elif self.regularization == 'elasticnet':
            # Elastic-Net logistic regression
            return LogisticRegression(
                penalty='elasticnet', C=self.C, solver='saga',
                l1_ratio=self.alpha, tol=self.tol,
                class_weight='balanced', max_iter=self.max_iter,
                random_state=seed
            )
        elif self.regularization == 'adaptive_lasso':
            # Adaptive LASSO (simplified): weighted L1 would require a pilot fit.
            # Here we approximate with stronger L1 regularization.
            return LogisticRegression(
                penalty='l1', C=self.C, solver='liblinear',
                class_weight='balanced', max_iter=self.max_iter,
                random_state=seed
            )
        
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")
    
    def _get_adaptive_lasso_estimator(self, seed: int):
        """Create adaptive LASSO estimator"""
        # This is a simplified version - in practice would need custom implementation
        return LogisticRegression(
            penalty='l1', C=1.0, solver='liblinear',
            class_weight='balanced', max_iter=5000,
            random_state=seed
        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Run ensemble STABL and select consensus features
        
        Returns
        -------
        consensus_features : List[str]
            Features selected by consensus
        """
        print(f"\nEnsemble STABL ({self.n_runs} runs, {self.regularization})")
        print("="*70)
        
        n_samples, n_features = X.shape
        self.stability_matrix = np.zeros((n_features, self.n_runs))
        
        # Scale features once
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Run STABL multiple times
        for run_idx in range(self.n_runs):
            seed = self.random_state + run_idx * 1000
            print(f"\nRun {run_idx + 1}/{self.n_runs} (seed={seed})")
            
            # Get base estimator for this run
            base_estimator = self._get_base_estimator(seed)
            
            # Initialize STABL (FDR mode if artificial_type provided)
            use_fdr = self.artificial_type is not None and str(self.artificial_type).lower() != 'none'
            try:
                fdr_range = np.arange(float(self.fdr_start), float(self.fdr_end), float(self.fdr_step)) if use_fdr else None
            except Exception:
                fdr_range = None
            stabl = Stabl(
                base_estimator=base_estimator,
                n_bootstraps=self.n_bootstraps,
                artificial_type=(self.artificial_type if use_fdr else None),
                sample_fraction=self.sample_fraction,
                replace=False,
                hard_threshold=(None if use_fdr else self.threshold),
                fdr_threshold_range=fdr_range,
                lambda_grid=(self.lambda_grid if use_fdr else None),
                n_lambda=(self.n_lambda if use_fdr else None),
                perc_corr_group_threshold=self.perc_corr_group_threshold,
                n_jobs=self.n_jobs,  # Use specified cores for parallel bootstrapping
                random_state=seed,
                verbose=0
            )
            
            # Fit STABL
            stabl.fit(X_scaled, y)
            
            # Get selected features
            selected_mask = stabl.get_support()
            selected_features = X.columns[selected_mask].tolist()
            
            # Store results
            self.run_results.append({
                'run': run_idx,
                'selected_features': selected_features,
                'n_selected': len(selected_features)
            })
            
            # Update feature frequencies
            for feat in selected_features:
                self.feature_frequencies[feat] += 1
            
            # Store stability scores
            if hasattr(stabl, 'stabl_scores_'):
                if len(stabl.stabl_scores_.shape) > 1:
                    scores = np.mean(stabl.stabl_scores_, axis=1)
                else:
                    scores = stabl.stabl_scores_
                self.stability_matrix[:, run_idx] = scores
            
            print(f"  Selected {len(selected_features)} features")
        
        # Select consensus features
        self._select_consensus_features()
        
        return self.consensus_features
    
    def _select_consensus_features(self):
        """Select features based on consensus across runs"""
        
        # Calculate selection frequency for each feature
        feature_names = list(self.feature_frequencies.keys())
        selection_freq = {
            feat: count / self.n_runs 
            for feat, count in self.feature_frequencies.items()
        }
        
        # Sort by frequency
        sorted_features = sorted(
            selection_freq.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Apply consensus threshold
        consensus_candidates = [
            feat for feat, freq in sorted_features 
            if freq >= self.consensus_threshold
        ]
        
        # Apply maximum feature constraint
        self.consensus_features = consensus_candidates[:self.max_features]
        
        print(f"\n{'='*70}")
        print(f"Consensus Feature Selection (threshold={self.consensus_threshold})")
        print(f"{'='*70}")
        print(f"Total unique features selected: {len(feature_names)}")
        print(f"Features meeting consensus: {len(consensus_candidates)}")
        print(f"Final features selected: {len(self.consensus_features)}")
        
        # Print selection frequencies
        print("\nSelection Frequencies:")
        for i, (feat, freq) in enumerate(sorted_features[:20], 1):
            selected = "✓" if feat in self.consensus_features else " "
            print(f"{i:3d}. [{selected}] {feat}: {freq:.2f}")
        
        if len(sorted_features) > 20:
            print(f"     ... and {len(sorted_features) - 20} more")


class UltraConservativeValidation:
    """Advanced validation methods for small sample sizes"""
    
    def __init__(self,
                 method: str = 'loocv',
                 n_bootstrap: int = 1000,
                 n_mccv_iterations: int = 200,
                 mccv_test_size: float = 0.2,
                 random_state: int = 42):
        """
        Parameters
        ----------
        method : str
            Validation method ('loocv', 'bootstrap_632', 'bootstrap_632_plus', 'mccv', 'all')
        n_bootstrap : int
            Number of bootstrap iterations
        n_mccv_iterations : int
            Number of Monte Carlo CV iterations
        mccv_test_size : float
            Test set size for MCCV
        random_state : int
            Random seed
        """
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.n_mccv_iterations = n_mccv_iterations
        self.mccv_test_size = mccv_test_size
        self.random_state = random_state
    
    def _calculate_cv_ci(self, scores, method='percentile', n_samples=183):
        """Calculate proper confidence intervals for CV scores
        
        Parameters
        ----------
        scores : array-like
            Cross-validation scores
        method : str
            'percentile' or 'corrected_t'
        n_samples : int
            Total sample size (for corrected_t method)
            
        Returns
        -------
        ci_lower, ci_upper : float
            95% confidence interval bounds
        """
        scores = np.array(scores)
        
        if len(scores) < 2:
            return scores[0], scores[0] if len(scores) == 1 else (0.5, 0.5)
        
        if method == 'percentile':
            # Bootstrap percentile method - most robust
            return np.percentile(scores, 2.5), np.percentile(scores, 97.5)
        
        elif method == 'corrected_t':
            # Nadeau-Bengio correction for dependent CV folds
            k = len(scores)
            n_test = n_samples // k
            n_train = n_samples - n_test
            
            # Correction factor for dependence
            correction = np.sqrt(1/k + n_test/n_train)
            se = np.std(scores, ddof=1) * correction
            mean_score = np.mean(scores)
            
            # 95% CI with t-distribution (more appropriate for small k)
            from scipy import stats
            t_critical = stats.t.ppf(0.975, k-1)
            
            ci_lower = mean_score - t_critical*se
            ci_upper = mean_score + t_critical*se
            
            # Bound to [0, 1]
            return max(0, ci_lower), min(1, ci_upper)
        
        else:
            # Fallback to percentile
            return np.percentile(scores, 2.5), np.percentile(scores, 97.5)
        
    def evaluate_model(self, X: np.ndarray, y: np.ndarray, 
                      model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """
        Evaluate model using specified validation method
        
        Returns
        -------
        results : dict
            Validation results with performance metrics
        """
        if self.method == 'loocv':
            return self._loocv_evaluation(X, y, model_class, model_params)
        elif self.method == 'bootstrap_632':
            return self._bootstrap_632_evaluation(X, y, model_class, model_params)
        elif self.method == 'bootstrap_632_plus':
            return self._bootstrap_632_plus_evaluation(X, y, model_class, model_params)
        elif self.method == 'mccv':
            return self._mccv_evaluation(X, y, model_class, model_params)
        elif self.method == 'all':
            return self._all_methods_evaluation(X, y, model_class, model_params)
        else:
            raise ValueError(f"Unknown validation method: {self.method}")
    
    def _loocv_evaluation(self, X: np.ndarray, y: np.ndarray,
                         model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """Leave-One-Out Cross-Validation"""
        
        loo = LeaveOneOut()
        predictions = np.zeros(len(y))
        probabilities = np.zeros(len(y))
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                prob = model.decision_function(X_test_scaled)
                prob = 1 / (1 + np.exp(-prob))
            
            probabilities[test_idx] = prob
        
        # Calculate metrics
        auc = roc_auc_score(y, probabilities)
        brier = brier_score_loss(y, probabilities)
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y, probabilities)
        
        return {
            'method': 'LOOCV',
            'auc': auc,
            'brier': brier,
            'predictions': probabilities,
            'y_true': y,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        }
    
    def _bootstrap_632_evaluation(self, X: np.ndarray, y: np.ndarray,
                                 model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """Bootstrap .632 evaluation"""
        
        n_samples = len(y)
        boot_aucs = []
        oob_aucs = []
        
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap .632"):
            # Create bootstrap sample
            boot_idx = resample(np.arange(n_samples), n_samples=n_samples,
                              random_state=self.random_state + i)
            oob_idx = np.array([idx for idx in range(n_samples) if idx not in boot_idx])
            
            if len(oob_idx) < 5 or len(np.unique(y[boot_idx])) < 2:
                continue
            
            X_boot, y_boot = X[boot_idx], y[boot_idx]
            X_oob, y_oob = X[oob_idx], y[oob_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_boot_scaled = scaler.fit_transform(X_boot)
            X_oob_scaled = scaler.transform(X_oob)
            
            # Train model
            model = model_class(**model_params)
            model.fit(X_boot_scaled, y_boot)
            
            # Evaluate on bootstrap sample (resubstitution)
            if hasattr(model, 'predict_proba'):
                boot_pred = model.predict_proba(X_boot_scaled)[:, 1]
                oob_pred = model.predict_proba(X_oob_scaled)[:, 1]
            else:
                boot_pred = model.decision_function(X_boot_scaled)
                boot_pred = 1 / (1 + np.exp(-boot_pred))
                oob_pred = model.decision_function(X_oob_scaled)
                oob_pred = 1 / (1 + np.exp(-oob_pred))
            
            try:
                boot_aucs.append(roc_auc_score(y_boot, boot_pred))
                oob_aucs.append(roc_auc_score(y_oob, oob_pred))
            except:
                continue
        
        # Calculate .632 estimate
        mean_boot = np.mean(boot_aucs)
        mean_oob = np.mean(oob_aucs)
        auc_632 = 0.368 * mean_boot + 0.632 * mean_oob
        
        # Calculate CI from bootstrap distribution
        ci_lower, ci_upper = self._calculate_cv_ci(oob_aucs, method='percentile')
        
        return {
            'method': 'Bootstrap .632',
            'auc': auc_632,
            'auc_boot': mean_boot,
            'auc_oob': mean_oob,
            'auc_ci': [ci_lower, ci_upper],
            'auc_std': np.std(oob_aucs),
            'n_iterations': len(boot_aucs)
        }
    
    def _bootstrap_632_plus_evaluation(self, X: np.ndarray, y: np.ndarray,
                                      model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """Bootstrap .632+ evaluation (correct error-based weighting for AUC)."""

        # Build bootstrap distributions of train and OOB AUCs
        n_samples = len(y)
        train_aucs = []
        oob_aucs = []

        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap .632+"):
            boot_idx = resample(np.arange(n_samples), n_samples=n_samples, random_state=self.random_state + i)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]
            if len(oob_idx) < 5 or len(np.unique(y[oob_idx])) < 2:
                continue

            X_boot, y_boot = X[boot_idx], y[boot_idx]
            X_oob, y_oob = X[oob_idx], y[oob_idx]

            scaler = StandardScaler()
            X_boot_s = scaler.fit_transform(X_boot)
            X_oob_s = scaler.transform(X_oob)

            model = model_class(**model_params)
            model.fit(X_boot_s, y_boot)

            if hasattr(model, 'predict_proba'):
                prob_boot = model.predict_proba(X_boot_s)[:, 1]
                prob_oob = model.predict_proba(X_oob_s)[:, 1]
            else:
                prob_boot = model.decision_function(X_boot_s)
                prob_oob = model.decision_function(X_oob_s)
                prob_boot = 1 / (1 + np.exp(-prob_boot))
                prob_oob = 1 / (1 + np.exp(-prob_oob))

            try:
                train_aucs.append(roc_auc_score(y_boot, prob_boot))
                oob_aucs.append(roc_auc_score(y_oob, prob_oob))
            except Exception:
                continue

        if not oob_aucs:
            raise ValueError("No valid bootstrap replicates for .632+")

        # Convert to numpy
        train_aucs = np.array(train_aucs, dtype=float)
        oob_aucs = np.array(oob_aucs, dtype=float)
        err0 = 0.5  # no-information error for AUC
        eps = 1e-8

        err_tr = 1.0 - train_aucs
        err_oob = 1.0 - oob_aucs
        denom = np.maximum(err0 - err_tr, eps)
        R = np.clip((err_oob - err_tr) / denom, 0.0, 1.0)
        w = 0.632 / (1.0 - 0.368 * R)
        err_632p = (1.0 - w) * err_tr + w * err_oob
        auc632p = 1.0 - err_632p

        return {
            'method': 'Bootstrap .632+',
            'auc': float(np.mean(auc632p)),
            'auc_ci': [float(np.percentile(auc632p, 2.5)), float(np.percentile(auc632p, 97.5))],
            'auc_std': float(np.std(auc632p)),
            'n_iterations': int(len(auc632p))
        }
    
    def _mccv_evaluation(self, X: np.ndarray, y: np.ndarray,
                        model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """Monte Carlo Cross-Validation"""
        
        from sklearn.model_selection import train_test_split
        
        aucs = []
        briers = []
        
        for i in tqdm(range(self.n_mccv_iterations), desc="Monte Carlo CV"):
            try:
                # Stratified train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=self.mccv_test_size, 
                    stratify=y, random_state=self.random_state + i
                )
                
                # Check if we have enough events in test set
                if len(np.unique(y_test)) < 2 or np.sum(y_test) < 2:
                    continue
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.decision_function(X_test_scaled)
                    y_pred = 1 / (1 + np.exp(-y_pred))
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred)
                brier = brier_score_loss(y_test, y_pred)
                
                aucs.append(auc)
                briers.append(brier)
                
            except Exception as e:
                continue
        
        if len(aucs) == 0:
            raise ValueError("No successful MCCV iterations")
        
        return {
            'method': 'Monte Carlo CV',
            'auc': np.mean(aucs),
            'auc_std': np.std(aucs),
            'auc_median': np.median(aucs),
            'auc_ci': [np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)],
            'brier': np.mean(briers),
            'n_successful_iterations': len(aucs),
            'all_aucs': aucs
        }
    
    def _all_methods_evaluation(self, X: np.ndarray, y: np.ndarray,
                               model_class: Any, model_params: Dict) -> Dict[str, Any]:
        """Run all validation methods for comparison"""
        
        print("\nRunning comprehensive validation comparison...")
        results = {}
        
        # LOOCV
        print("\n1. Leave-One-Out CV...")
        try:
            results['loocv'] = self._loocv_evaluation(X, y, model_class, model_params)
        except Exception as e:
            print(f"   LOOCV failed: {e}")
            results['loocv'] = None
        
        # Monte Carlo CV
        print("\n2. Monte Carlo CV...")
        try:
            results['mccv'] = self._mccv_evaluation(X, y, model_class, model_params)
        except Exception as e:
            print(f"   MCCV failed: {e}")
            results['mccv'] = None
        
        # Bootstrap .632
        print("\n3. Bootstrap .632...")
        try:
            results['bootstrap_632'] = self._bootstrap_632_evaluation(X, y, model_class, model_params)
        except Exception as e:
            print(f"   Bootstrap .632 failed: {e}")
            results['bootstrap_632'] = None
        
        # Bootstrap .632+
        print("\n4. Bootstrap .632+...")
        try:
            results['bootstrap_632_plus'] = self._bootstrap_632_plus_evaluation(X, y, model_class, model_params)
        except Exception as e:
            print(f"   Bootstrap .632+ failed: {e}")
            results['bootstrap_632_plus'] = None
        
        # Create summary
        summary = {
            'method': 'All validation methods',
            'comparison': {}
        }
        
        for method_name, result in results.items():
            if result is not None:
                summary['comparison'][method_name] = {
                    'auc': result['auc'],
                    'method': result['method']
                }
                if 'auc_std' in result:
                    summary['comparison'][method_name]['std'] = result['auc_std']
                if 'auc_ci' in result:
                    summary['comparison'][method_name]['ci'] = result['auc_ci']
        
        # Find best and worst estimates
        aucs = [r['auc'] for r in results.values() if r is not None]
        if aucs:
            summary['best_estimate'] = max(aucs)
            summary['worst_estimate'] = min(aucs)
            summary['range'] = max(aucs) - min(aucs)
            summary['mean_across_methods'] = np.mean(aucs)
        
        summary['detailed_results'] = results
        
        return summary


class PermutationTesting:
    """Permutation testing for statistical significance"""
    
    def __init__(self, n_permutations: int = 1000, random_state: int = 42):
        self.n_permutations = n_permutations
        self.random_state = random_state
        
    def test_significance(self, X: np.ndarray, y: np.ndarray,
                         model_class: Any, model_params: Dict,
                         observed_auc: float) -> Dict[str, Any]:
        """
        Test if observed AUC is significantly better than chance
        
        Returns
        -------
        results : dict
            Permutation test results
        """
        permuted_aucs = []
        
        for i in tqdm(range(self.n_permutations), desc="Permutation testing"):
            # Shuffle labels
            y_perm = y.copy()
            np.random.RandomState(self.random_state + i).shuffle(y_perm)
            
            # Evaluate with shuffled labels
            try:
                # Simple train-test split for speed
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_perm, test_size=0.2, stratify=y_perm,
                    random_state=self.random_state + i
                )
                
                # Scale and train
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model = model_class(**model_params)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred = model.decision_function(X_test_scaled)
                    y_pred = 1 / (1 + np.exp(-y_pred))
                
                auc = roc_auc_score(y_test, y_pred)
                permuted_aucs.append(auc)
                
            except:
                continue
        
        # Calculate p-value
        p_value = np.mean([auc >= observed_auc for auc in permuted_aucs])
        
        return {
            'observed_auc': observed_auc,
            'null_distribution': permuted_aucs,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'null_mean': np.mean(permuted_aucs),
            'null_std': np.std(permuted_aucs)
        }


class UltraOptimizedPipeline:
    """Complete ultra-optimized pipeline with all enhancements"""
    
    def __init__(self,
                 ensemble_runs: int = 10,
                 thresholds: List[float] = [0.7, 0.75, 0.8, 0.9],
                 max_features: int = 10,
                 regularizations: List[str] = ['l1'],
                 validation_method: str = 'bootstrap_632_plus',
                 n_permutations: int = 1000,
                 n_mccv_iterations: int = 200,
                 mccv_test_size: float = 0.2,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 output_dir: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        ensemble_runs : int
            Number of STABL runs per configuration
        thresholds : List[float]
            STABL thresholds to test
        max_features : int
            Maximum features to select
        regularizations : List[str]
            Regularization methods to test
        validation_method : str
            Validation method to use
        n_permutations : int
            Number of permutations for significance testing
        n_mccv_iterations : int
            Number of Monte Carlo CV iterations
        mccv_test_size : float
            Test set size for Monte Carlo CV
        n_jobs : int
            Number of CPU cores to use (-1 for all cores)
        random_state : int
            Random seed
        output_dir : Path or str, optional
            Output directory for results
        """
        self.ensemble_runs = ensemble_runs
        self.thresholds = thresholds
        self.max_features = max_features
        self.regularizations = regularizations
        self.validation_method = validation_method
        self.n_permutations = n_permutations
        self.n_mccv_iterations = n_mccv_iterations
        self.mccv_test_size = mccv_test_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Set output directory
        if output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = Path("results") / f"ultra_optimized_{timestamp}"
        else:
            self.output_dir = Path(output_dir)
        
        # Results storage
        self.results = {}
        self.best_config = None
        self.selected_features = []
        
    def run_pipeline(self, X: pd.DataFrame, y: pd.Series,
                    output_dir: Optional[Path] = None,
                    resume: bool = True) -> Dict[str, Any]:
        """
        Run complete ultra-optimized pipeline
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        output_dir : Path, optional
            Output directory
        resume : bool
            Whether to resume from previous incremental results
        
        Returns
        -------
        results : dict
            Complete pipeline results
        """
        if output_dir is None:
            output_dir = self.output_dir
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("ULTRA-OPTIMIZED STABL PIPELINE")
        print("="*70)
        print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Events: {y.sum()} ({y.mean()*100:.1f}%)")
        print(f"Max features allowed: {self.max_features}")
        print(f"Validation method: {self.validation_method}")
        
        # Check for existing incremental results
        config_results = []
        completed_configs = set()
        
        if resume:
            incremental_dir = output_dir / 'incremental_results'
            if incremental_dir.exists():
                print("\nChecking for previous results...")
                for pkl_file in incremental_dir.glob('*.pkl'):
                    try:
                        config_result = joblib.load(pkl_file)
                        config_results.append(config_result)
                        config_name = f"{config_result['regularization']}_{config_result['threshold']}"
                        completed_configs.add(config_name)
                        print(f"  Loaded: {config_name} (AUC={config_result['best_auc']:.4f})")
                    except Exception as e:
                        print(f"  Warning: Could not load {pkl_file}: {e}")
                
                if len(config_results) > 0:
                    print(f"\nResuming from {len(config_results)} completed configurations")
        
        for reg in self.regularizations:
            for thresh in self.thresholds:
                config_name = f"{reg}_{thresh}"
                
                # Skip if already completed
                if config_name in completed_configs:
                    print(f"\n\nSkipping {config_name} - already completed")
                    continue
                
                print(f"\n\n{'='*70}")
                print(f"Testing: {reg} regularization, threshold={thresh}")
                print(f"{'='*70}")
                
                # Run ensemble STABL
                ensemble = EnsembleSTABL(
                    n_runs=self.ensemble_runs,
                    threshold=thresh,
                    n_bootstraps=100,
                    regularization=reg,
                    max_features=self.max_features,
                    consensus_threshold=0.8,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state
                )
                
                selected_features = ensemble.fit(X, y)
                
                if len(selected_features) == 0:
                    print("No features selected - skipping evaluation")
                    continue
                
                # Evaluate with selected features
                X_selected = X[selected_features]
                
                # Initialize validator
                validator = UltraConservativeValidation(
                    method=self.validation_method,
                    n_bootstrap=1000,
                    n_mccv_iterations=self.n_mccv_iterations,
                    mccv_test_size=self.mccv_test_size,
                    random_state=self.random_state
                )
                
                # Test multiple models
                models = self._get_model_configs(y)
                model_results = {}
                
                for model_name, (model_class, model_params) in models.items():
                    print(f"\nEvaluating {model_name}...")
                    
                    # Validate model
                    val_results = validator.evaluate_model(
                        X_selected.values, y.values,
                        model_class, model_params
                    )
                    
                    # Handle 'all' validation method results
                    if self.validation_method == 'all':
                        # Use mean across methods for best_auc calculation
                        val_results['auc'] = val_results.get('mean_across_methods', 0.5)
                        # Also ensure we have a CI
                        if 'auc_ci' not in val_results and 'comparison' in val_results:
                            # Extract CIs from all methods and use the mean
                            all_cis = []
                            for method_data in val_results['comparison'].values():
                                if 'ci' in method_data:
                                    all_cis.append(method_data['ci'])
                            if all_cis:
                                # Use the widest CI (most conservative)
                                ci_lower = min(ci[0] for ci in all_cis)
                                ci_upper = max(ci[1] for ci in all_cis)
                                val_results['auc_ci'] = [ci_lower, ci_upper]
                    
                    # Permutation testing
                    if val_results['auc'] > 0.6:  # Only test if promising
                        perm_tester = PermutationTesting(
                            n_permutations=self.n_permutations,
                            random_state=self.random_state
                        )
                        
                        perm_results = perm_tester.test_significance(
                            X_selected.values, y.values,
                            model_class, model_params,
                            val_results['auc']
                        )
                        
                        val_results['permutation'] = perm_results
                    
                    model_results[model_name] = val_results
                
                # Store configuration results
                config_result = {
                    'regularization': reg,
                    'threshold': thresh,
                    'n_features': len(selected_features),
                    'features': selected_features,
                    'ensemble': ensemble,
                    'model_results': model_results,
                    'best_auc': max([r.get('auc', 0) for r in model_results.values()]) if model_results else 0
                }
                
                config_results.append(config_result)
                
                # Save incremental result immediately
                self._save_incremental_result(config_result, output_dir)
                print(f"\nSaved incremental result for {reg}_{thresh}")
        
        # Find best configuration
        if len(config_results) == 0:
            print("\nNo configurations completed!")
            return {'error': 'No configurations completed'}
        
        self.best_config = max(config_results, key=lambda x: x['best_auc'])
        self.selected_features = self.best_config['features']
        
        # Create comprehensive report
        self._save_results(config_results, output_dir)
        self._create_visualizations(config_results, output_dir)
        
        print(f"\n\nPipeline complete! Results saved to: {output_dir}")
        
        return {
            'best_config': self.best_config,
            'all_results': config_results,
            'selected_features': self.selected_features
        }
    
    def _get_model_configs(self, y: pd.Series) -> Dict[str, Tuple[Any, Dict]]:
        """Get model configurations"""
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        scale_pos_weight = n_neg / n_pos
        
        return {
            'logistic_regression': (
                LogisticRegression,
                {
                    'penalty': 'l2', 'C': 1.0, 'solver': 'lbfgs',
                    'class_weight': 'balanced', 'max_iter': 1000
                }
            ),
            'random_forest': (
                RandomForestClassifier,
                {
                    'n_estimators': 100, 'max_depth': 3,
                    'class_weight': 'balanced', 
                    'random_state': self.random_state
                }
            ),
            'xgboost': (
                xgb.XGBClassifier,
                {
                    'n_estimators': 50, 'max_depth': 2,
                    'learning_rate': 0.05,
                    'scale_pos_weight': scale_pos_weight,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'random_state': self.random_state
                }
            )
        }
    
    def _save_incremental_result(self, config_result: Dict, output_dir: Path):
        """Save individual configuration result incrementally"""
        
        # Create incremental results directory
        incremental_dir = output_dir / 'incremental_results'
        incremental_dir.mkdir(exist_ok=True)
        
        # Save this configuration's result
        config_name = f"{config_result['regularization']}_{config_result['threshold']}"
        config_file = incremental_dir / f'{config_name}.pkl'
        joblib.dump(config_result, config_file)
        
        # Update summary file
        summary_file = output_dir / 'incremental_summary.txt'
        with open(summary_file, 'a') as f:
            f.write(f"\nConfiguration: {config_name}\n")
            f.write(f"  Features: {config_result['n_features']}\n")
            f.write(f"  Best AUC: {config_result['best_auc']:.4f}\n")
            f.write(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-"*50 + "\n")
    
    def _save_results(self, config_results: List[Dict], output_dir: Path):
        """Save comprehensive results with individual threshold reports"""
        
        # Save raw results
        joblib.dump(config_results, output_dir / 'ultra_optimized_results.pkl')
        
        # Create individual threshold reports and plots
        for config in config_results:
            self._create_threshold_report(config, output_dir)
            self._create_threshold_summary_plot(config, output_dir)
            self._create_threshold_performance_plot(config, output_dir)
        
        # Create cross-threshold comparisons
        if len(config_results) > 1:
            self._create_cross_threshold_comparison(config_results, output_dir)
            self._create_threshold_ci_comparison(config_results, output_dir)
        
        # Create detailed report
        with open(output_dir / 'ultra_optimized_report.txt', 'w') as f:
            f.write("ULTRA-OPTIMIZED STABL PIPELINE REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Best configuration
            best = self.best_config
            f.write("BEST CONFIGURATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Regularization: {best['regularization']}\n")
            f.write(f"STABL threshold: {best['threshold']}\n")
            f.write(f"Features selected: {best['n_features']}\n")
            f.write(f"Best AUC: {best['best_auc']:.4f}\n\n")
            
            # Feature list
            f.write("SELECTED FEATURES\n")
            f.write("-"*30 + "\n")
            for i, feat in enumerate(best['features'], 1):
                f.write(f"{i:3d}. {feat}\n")
            
            f.write("\n\nMODEL PERFORMANCE\n")
            f.write("-"*30 + "\n")
            for model_name, results in best['model_results'].items():
                f.write(f"\n{model_name.upper()}\n")
                f.write(f"  {results['method']} AUC: {results['auc']:.4f}\n")
                
                # Add CI if available
                if 'auc_ci' in results:
                    ci = results['auc_ci']
                    f.write(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n")
                
                if 'permutation' in results:
                    perm = results['permutation']
                    f.write(f"  Permutation p-value: {perm['p_value']:.4f}\n")
                    f.write(f"  Null mean AUC: {perm['null_mean']:.4f} ± {perm['null_std']:.4f}\n")
                    f.write(f"  Significant: {'Yes' if perm['significant'] else 'No'}\n")
            
            # All configurations summary with CI
            f.write("\n\nALL CONFIGURATIONS TESTED\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Config':<20} {'Features':<10} {'Best AUC':<15} {'95% CI':<20}\n")
            f.write("-"*70 + "\n")
            
            for config in sorted(config_results, key=lambda x: x['best_auc'], reverse=True):
                config_name = f"{config['regularization']}_{config['threshold']}"
                
                # Find best model CI
                best_ci = None
                for model_name, results in config['model_results'].items():
                    if results.get('auc', 0) == config['best_auc'] and 'auc_ci' in results:
                        best_ci = results['auc_ci']
                        break
                
                ci_str = f"[{best_ci[0]:.3f}, {best_ci[1]:.3f}]" if best_ci else "N/A"
                f.write(f"{config_name:<20} {config['n_features']:<10} {config['best_auc']:<15.4f} {ci_str:<20}\n")
        
        # Save feature stability matrix if available
        if self.best_config and hasattr(self.best_config['ensemble'], 'stability_matrix'):
            try:
                ensemble = self.best_config['ensemble']
                stability_matrix = ensemble.stability_matrix
                
                # Get the actual feature names from the training data
                # The stability matrix has shape (n_features, n_runs)
                n_features, n_runs = stability_matrix.shape
                
                # Create a simplified stability summary
                stability_data = {
                    'feature_index': list(range(n_features)),
                    'mean_stability': np.mean(stability_matrix, axis=1),
                    'std_stability': np.std(stability_matrix, axis=1),
                    'min_stability': np.min(stability_matrix, axis=1),
                    'max_stability': np.max(stability_matrix, axis=1)
                }
                
                # Add individual run scores
                for run_idx in range(n_runs):
                    stability_data[f'run_{run_idx}'] = stability_matrix[:, run_idx]
                
                stability_df = pd.DataFrame(stability_data)
                
                # Save the stability matrix
                stability_df.to_csv(output_dir / 'feature_stability_scores.csv', index=False)
                
                # Also save selected features with their frequencies
                if hasattr(ensemble, 'feature_frequencies') and ensemble.feature_frequencies:
                    freq_df = pd.DataFrame(
                        list(ensemble.feature_frequencies.items()),
                        columns=['feature', 'selection_count']
                    )
                    freq_df['selection_frequency'] = freq_df['selection_count'] / ensemble.n_runs
                    freq_df = freq_df.sort_values('selection_frequency', ascending=False)
                    freq_df.to_csv(output_dir / 'feature_selection_frequencies.csv', index=False)
                    
            except Exception as e:
                print(f"Warning: Could not save stability matrix: {e}")
    
    def _create_visualizations(self, config_results: List[Dict], output_dir: Path):
        """Create comprehensive visualizations with BeautifulFigures standards"""
        
        # Note: Individual threshold plots are now created in _save_results
        # This method creates the summary visualizations only
        
        if PLOTTING_AVAILABLE:
            # Use BeautifulFigures plotting utilities
            self._create_beautiful_visualizations(config_results, output_dir)
        else:
            # Fallback to default matplotlib
            self._create_default_visualizations(config_results, output_dir)
    
    def _create_beautiful_visualizations(self, config_results: List[Dict], output_dir: Path):
        """Create visualizations using BeautifulFigures standards with Nord theme"""
        
        # 1. Configuration comparison - Split into two separate plots
        configs = []
        aucs = []
        n_features = []
        
        for config in config_results:
            config_name = f"{config['regularization']}_{config['threshold']}"
            configs.append(config_name)
            aucs.append(config['best_auc'])
            n_features.append(config['n_features'])
        
        # 1a. AUC comparison plot
        fig, ax = create_beautiful_figure('wide')
        
        colors = [NORD_COLORS['nord10'] if auc == max(aucs) else NORD_COLORS['nord9'] 
                  for auc in aucs]
        bars = ax.bar(configs, aucs, color=colors, edgecolor=NORD_COLORS['nord3'],
                      linewidth=2, alpha=0.9)
        ax.set_ylabel('Best AUC', fontsize=20)
        ax.set_xlabel('Configuration', fontsize=20)
        ax.set_title('Model Performance by Configuration', fontsize=24, fontweight='bold', pad=20)
        ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
        
        # Add value labels
        for bar, auc in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=16)
        
        ax.legend(fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_beautiful_figure(fig, output_dir / 'configuration_comparison_auc')
        plt.close()
        
        # 1b. Feature count plot
        fig, ax = create_beautiful_figure('wide')
        
        # Create gradient colors based on feature count
        if max(n_features) > 0:
            gradient_colors = [NORD_COLORS['nord14'] if nf <= 5 else 
                              NORD_COLORS['nord13'] if nf <= 10 else 
                              NORD_COLORS['nord11'] for nf in n_features]
        else:
            gradient_colors = [NORD_COLORS['nord9']] * len(n_features)
            
        bars = ax.bar(configs, n_features, color=gradient_colors, 
                      edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
        ax.set_ylabel('Number of Features', fontsize=20)
        ax.set_xlabel('Configuration', fontsize=20)
        ax.set_title('Feature Selection by Configuration', fontsize=24, fontweight='bold', pad=20)
        ax.axhline(10, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, 
                   label='Max recommended')
        ax.axhline(5, color=NORD_COLORS['nord14'], linestyle='--', alpha=0.5, 
                   label='Optimal range')
        
        # Add value labels
        for bar, nf in zip(bars, n_features):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(nf), ha='center', va='bottom', fontsize=16)
        
        ax.legend(fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        save_beautiful_figure(fig, output_dir / 'configuration_comparison_features')
        plt.close()
        
        # 2. Feature consistency plot
        best = self.best_config
        if best and 'ensemble' in best and hasattr(best['ensemble'], 'feature_frequencies'):
            fig, ax = create_beautiful_figure('tall')
            
            # Sort features by frequency
            features = list(best['ensemble'].feature_frequencies.keys())
            frequencies = list(best['ensemble'].feature_frequencies.values())
            sorted_idx = np.argsort(frequencies)[::-1]
            features = [features[i] for i in sorted_idx]
            frequencies = [frequencies[i] for i in sorted_idx]
            
            # Color based on threshold
            colors = [NORD_COLORS['nord14'] if f >= 0.8 else NORD_COLORS['nord13'] 
                     for f in frequencies]
            
            plt.barh(features, frequencies, color=colors, 
                    edgecolor=NORD_COLORS['nord3'], linewidth=1.5, alpha=0.9)
            plt.xlabel('Selection Frequency')
            plt.title(f'Feature Selection Consistency ({best["ensemble"].n_runs} runs)')
            plt.axvline(0.8, color=NORD_COLORS['nord11'], linestyle='--', 
                       alpha=0.5, label='Consensus threshold')
            plt.legend()
            plt.tight_layout()
            save_beautiful_figure(fig, output_dir / 'feature_consistency')
            plt.close()
        
        # 3. Individual model performance plots
        if best and 'model_results' in best:
            for model_name, results in best['model_results'].items():
                self._create_individual_model_plot(model_name, results, output_dir)
        
        # 4. Permutation test plots
        if best and 'model_results' in best:
            for model_name, results in best['model_results'].items():
                if 'permutation' in results:
                    self._create_permutation_plot(
                        results['permutation'], model_name, output_dir
                    )
        
        # 5. ROC curves with confidence intervals
        if best and 'model_results' in best:
            self._create_roc_curves_plot(best['model_results'], output_dir)
        
    
    def _create_individual_model_plot(self, model_name: str, results: Dict, output_dir: Path):
        """Create individual model performance plot with confidence intervals"""
        fig, ax = create_beautiful_figure('square')
        
        # Extract model display name
        display_name = model_name.replace('_', ' ').title()
        
        # Handle different validation result formats
        if 'comparison' in results:
            # Multiple validation methods
            methods = []
            aucs = []
            cis = []
            
            for method, data in results['comparison'].items():
                methods.append(method.replace('_', ' ').title())
                aucs.append(data['auc'])
                
                # Get confidence intervals
                if 'ci' in data:
                    cis.append(data['ci'])
                elif 'std' in data:
                    # Don't convert std to CI - just show without error bars
                    # or use a more conservative approach
                    cis.append((data['auc'], data['auc']))
                else:
                    cis.append((data['auc'], data['auc']))
            
            # Create bar plot
            x = np.arange(len(methods))
            bars = ax.bar(x, aucs, color=NORD_COLORS['nord9'], 
                          edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
            
            # Add error bars
            if cis:
                errors = [[auc - ci[0] for auc, ci in zip(aucs, cis)],
                         [ci[1] - auc for auc, ci in zip(aucs, cis)]]
                ax.errorbar(x, aucs, yerr=errors, fmt='none', 
                           color=NORD_COLORS['nord0'], capsize=5, capthick=2)
            
            # Highlight best
            best_idx = np.argmax(aucs)
            bars[best_idx].set_color(NORD_COLORS['nord14'])
            
            # Styling
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
            ax.set_ylabel('AUC', fontsize=18)
            ax.set_ylim([0.4, 1.0])
            
            # Add value labels
            for i, (bar, auc, ci) in enumerate(zip(bars, aucs, cis)):
                label = f'{auc:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       label, ha='center', va='bottom', fontsize=12)
        else:
            # Single validation method
            auc = results.get('auc', 0)
            method = results.get('method', 'Unknown')
            
            # Create single bar
            bars = ax.bar([0], [auc], color=NORD_COLORS['nord14'],
                          edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
            
            # Add CI if available
            if 'auc_ci' in results:
                ci = results['auc_ci']
                errors = [[auc - ci[0]], [ci[1] - auc]]
                ax.errorbar([0], [auc], yerr=errors, fmt='none',
                           color=NORD_COLORS['nord0'], capsize=10, capthick=3)
                label = f'{auc:.3f}\n[{ci[0]:.3f}, {ci[1]:.3f}]'
            else:
                label = f'{auc:.3f}'
            
            ax.text(0, auc + 0.01, label, ha='center', va='bottom', fontsize=16)
            ax.set_xticks([0])
            ax.set_xticklabels([method], fontsize=16)
            ax.set_ylabel('AUC', fontsize=18)
            ax.set_ylim([0.4, 1.0])
        
        # Common styling
        ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
        ax.set_title(f'{display_name} Performance', fontsize=22, fontweight='bold', pad=20)
        ax.legend(fontsize=14)
        
        plt.tight_layout()
        save_beautiful_figure(fig, output_dir / f'model_performance_{model_name}')
        plt.close()
    
    def _create_model_comparison_plot(self, model_results: Dict, output_dir: Path):
        """Create model comparison plot with confidence intervals"""
        models = []
        aucs = []
        cis = []
        
        # Extract data from model results
        for model_name, results in model_results.items():
            models.append(model_name.replace('_', ' ').title())
            aucs.append(results['auc'])
            
            # Get confidence intervals
            if 'ci' in results:
                cis.append(results['ci'])
            elif 'auc_ci' in results:
                cis.append(results['auc_ci'])
            elif 'std' in results:
                # Don't convert std to CI - just show without error bars
                cis.append((results['auc'], results['auc']))
            else:
                cis.append((results['auc'], results['auc']))
        
        # Create plot
        fig, ax = plot_model_comparison_with_ci(
            models, aucs, cis,
            title="Model Performance Comparison - Radiomics Features"
        )
        
        save_beautiful_figure(fig, output_dir / 'model_comparison')
        plt.close()
    
    def _create_permutation_plot(self, perm_results: Dict, model_name: str, output_dir: Path):
        """Create permutation test visualization with Nord theme"""
        fig, ax = create_beautiful_figure('wide')
        
        # Plot null distribution
        ax.hist(perm_results['null_distribution'], bins=30, alpha=0.7,
                color=NORD_COLORS['nord9'], edgecolor=NORD_COLORS['nord3'],
                label=f'Null distribution (n={len(perm_results["null_distribution"])})')
        
        # Add observed AUC line
        ax.axvline(perm_results['observed_auc'], color=NORD_COLORS['nord11'], 
                  linewidth=3, label=f'Observed AUC: {perm_results["observed_auc"]:.3f}')
        
        # Add null mean
        ax.axvline(perm_results['null_mean'], color=NORD_COLORS['nord0'], 
                  linestyle='--', linewidth=2,
                  label=f'Null mean: {perm_results["null_mean"]:.3f}')
        
        # Styling
        ax.set_xlabel('AUC')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Permutation Test: {model_name.replace("_", " ").title()} (p={perm_results["p_value"]:.4f})')
        ax.legend()
        
        save_beautiful_figure(fig, output_dir / f'permutation_test_{model_name}')
        plt.close()
    
    def _create_roc_curves_plot(self, model_results: Dict, output_dir: Path):
        """Create ROC curves with confidence intervals for all models"""
        
        # Create ROC subdirectory
        roc_dir = output_dir / 'roc_curves'
        roc_dir.mkdir(exist_ok=True)
        
        # Plot individual model ROC curves
        for model_name, results in model_results.items():
            if results is None:
                continue
                
            # Get validation results - handle different formats
            val_data = None
            auc_ci = None
            
            if 'detailed_results' in results:
                # Multiple validation methods - use LOOCV for ROC as it has all predictions
                detailed = results['detailed_results']
                if 'loocv' in detailed and detailed['loocv'] is not None:
                    val_data = detailed['loocv']
                    # Use CI from best method
                    if 'comparison' in results:
                        for method, data in results['comparison'].items():
                            if 'ci' in data:
                                auc_ci = data['ci']
                                break
            elif 'predictions' in results and 'y_true' in results:
                # Single validation method
                val_data = results
                auc_ci = results.get('auc_ci', None)
            elif 'predictions' in results:
                # Has predictions but no y_true stored - skip for now
                continue
                
            if val_data and 'predictions' in val_data and 'y_true' in val_data:
                if auc_ci is None and 'auc_ci' in val_data:
                    auc_ci = val_data['auc_ci']
                elif auc_ci is None:
                    # Default CI
                    auc = val_data.get('auc', 0.5)
                    auc_ci = [auc - 0.1, auc + 0.1]
                    
                self._plot_single_model_roc(
                    val_data['y_true'], 
                    val_data['predictions'],
                    model_name,
                    auc_ci,
                    roc_dir
                )
        
        # Create comparison plot with all models
        self._plot_roc_comparison(model_results, roc_dir)
    
    def _plot_single_model_roc(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str, auc_ci: List[float], output_dir: Path):
        """Plot ROC curve for a single model with bootstrap confidence bands"""
        
        # Calculate base ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)
        
        # Bootstrap for confidence bands
        n_bootstraps = 1000
        tprs = []
        aucs = []
        
        for i in range(n_bootstraps):
            # Bootstrap sample
            indices = np.random.choice(len(y_true), len(y_true), replace=True)
            y_boot = y_true[indices]
            pred_boot = y_pred[indices]
            
            if len(np.unique(y_boot)) < 2:
                continue
            
            # Calculate ROC
            fpr_boot, tpr_boot, _ = roc_curve(y_boot, pred_boot)
            
            # Interpolate to common FPR points
            tpr_interp = np.interp(fpr, fpr_boot, tpr_boot)
            tpr_interp[0] = 0.0
            tprs.append(tpr_interp)
            
            # Calculate AUC
            aucs.append(roc_auc_score(y_boot, pred_boot))
        
        # Calculate confidence bands
        tprs = np.array(tprs)
        mean_tpr = np.mean(tprs, axis=0)
        tpr_upper = np.percentile(tprs, 97.5, axis=0)
        tpr_lower = np.percentile(tprs, 2.5, axis=0)
        
        # Create plot
        fig, ax = create_beautiful_figure('square')
        
        # Plot confidence band
        ax.fill_between(fpr, tpr_lower, tpr_upper, 
                       color=NORD_COLORS['nord9'], alpha=0.2,
                       label='95% CI')
        
        # Plot mean ROC curve
        ax.plot(fpr, tpr, color=NORD_COLORS['nord9'], linewidth=3,
                label=f'ROC (AUC = {auc:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}])')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        
        # Styling
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate', fontsize=16)
        ax.set_ylabel('True Positive Rate', fontsize=16)
        ax.set_title(f'{model_name.replace("_", " ").title()} - ROC Curve', fontsize=18)
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Save
        save_beautiful_figure(fig, output_dir / f'roc_{model_name.lower()}')
        plt.close()
    
    def _plot_roc_comparison(self, model_results: Dict, output_dir: Path):
        """Plot ROC curves for all models on the same plot"""
        
        fig, ax = create_beautiful_figure('square')
        colors = [NORD_COLORS['nord9'], NORD_COLORS['nord11'], NORD_COLORS['nord14']]
        
        valid_models = 0
        for (model_name, results), color in zip(model_results.items(), colors):
            if results is None:
                continue
            
            # Get validation data - handle different formats
            val_data = None
            auc_ci = None
            
            if 'detailed_results' in results:
                # Multiple validation methods - use LOOCV for ROC
                detailed = results['detailed_results']
                if 'loocv' in detailed and detailed['loocv'] is not None:
                    val_data = detailed['loocv']
                    # Use CI from comparison
                    if 'comparison' in results:
                        for method, data in results['comparison'].items():
                            if 'ci' in data:
                                auc_ci = data['ci']
                                break
            elif 'predictions' in results and 'y_true' in results:
                val_data = results
                auc_ci = results.get('auc_ci', None)
            
            if val_data is None or 'predictions' not in val_data or 'y_true' not in val_data:
                continue
                
            # Get predictions and labels
            y_true = val_data['y_true']
            y_pred = val_data['predictions']
                
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            auc = roc_auc_score(y_true, y_pred)
            
            # Get CI if available
            if auc_ci is None and 'auc_ci' in val_data:
                auc_ci = val_data['auc_ci']
            
            if auc_ci:
                label = f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f} [{auc_ci[0]:.3f}-{auc_ci[1]:.3f}])'
            else:
                label = f'{model_name.replace("_", " ").title()} (AUC = {auc:.3f})'
            
            # Plot
            ax.plot(fpr, tpr, color=color, linewidth=3, label=label)
            valid_models += 1
        
        if valid_models > 0:
            # Plot diagonal
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random')
            
            # Styling
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate', fontsize=16)
            ax.set_ylabel('True Positive Rate', fontsize=16)
            ax.set_title('Model Comparison - ROC Curves', fontsize=18)
            ax.legend(loc='lower right', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Save
            save_beautiful_figure(fig, output_dir / 'roc_comparison')
            plt.close()
    
    def _create_default_visualizations(self, config_results: List[Dict], output_dir: Path):
        """Fallback visualization method using default matplotlib"""
        
        configs = []
        aucs = []
        n_features = []
        
        for config in config_results:
            config_name = f"{config['regularization']}_{config['threshold']}"
            configs.append(config_name)
            aucs.append(config['best_auc'])
            n_features.append(config['n_features'])
        
        # 1. Configuration comparison
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # AUC comparison
        bars1 = ax1.bar(configs, aucs, color=plt.cm.viridis(np.array(aucs)))
        ax1.set_ylabel('Best AUC')
        ax1.set_title('Model Performance by Configuration')
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5)
        
        # Add value labels
        for bar, auc in zip(bars1, aucs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom')
        
        # Feature count
        bars2 = ax2.bar(configs, n_features, color=plt.cm.plasma(np.array(n_features)/max(n_features)))
        ax2.set_ylabel('Number of Features')
        ax2.set_xlabel('Configuration')
        ax2.set_title('Feature Selection by Configuration')
        ax2.axhline(10, color='red', linestyle='--', alpha=0.5, label='Max recommended')
        
        # Add value labels
        for bar, n in zip(bars2, n_features):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(n), ha='center', va='bottom')
        
        ax2.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / 'configuration_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature consistency plot
        best = self.best_config
        if best and 'ensemble' in best and hasattr(best['ensemble'], 'feature_frequencies'):
            plt.figure(figsize=(10, 8))
            
            # Get top features
            freq_data = best['ensemble'].feature_frequencies
            top_features = sorted(freq_data.items(), key=lambda x: x[1], reverse=True)[:20]
            
            features = [f[0] for f in top_features]
            frequencies = [f[1] for f in top_features]
            
            plt.barh(features, frequencies, color=plt.cm.RdYlGn(frequencies))
            plt.xlabel('Selection Frequency')
            plt.title(f'Feature Selection Consistency ({best["ensemble"].n_runs} runs)')
            plt.axvline(0.8, color='red', linestyle='--', alpha=0.5, label='Consensus threshold')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / 'feature_consistency.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Permutation test plots
        if best and 'model_results' in best:
            for model_name, results in best['model_results'].items():
                if 'permutation' in results:
                    perm = results['permutation']
                    
                    plt.figure(figsize=(10, 6))
                    plt.hist(perm['null_distribution'], bins=30, alpha=0.7, 
                            label=f'Null distribution (n={len(perm["null_distribution"])})')
                    plt.axvline(perm['observed_auc'], color='red', linewidth=2,
                               label=f'Observed AUC = {perm["observed_auc"]:.3f}')
                    plt.axvline(perm['null_mean'], color='black', linestyle='--',
                               label=f'Null mean = {perm["null_mean"]:.3f}')
                    
                    plt.xlabel('AUC')
                    plt.ylabel('Frequency')
                    plt.title(f'Permutation Test: {model_name} (p={perm["p_value"]:.4f})')
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(output_dir / f'permutation_test_{model_name}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        
        # 4. Validation method comparison (if using 'all')
        if self.validation_method == 'all' and best and 'model_results' in best:
            self._create_validation_comparison_plot(best['model_results'], output_dir)
    
    def _create_validation_comparison_plot(self, model_results: Dict, output_dir: Path):
        """Create individual validation comparison plots for each model"""
        
        for model_name, results in model_results.items():
            if 'comparison' not in results:
                continue
                
            # Create individual plot for this model
            fig, ax = create_beautiful_figure('wide')
            
            methods = []
            aucs = []
            errors = []
            
            for method, data in results['comparison'].items():
                methods.append(method.replace('_', ' ').title())
                aucs.append(data['auc'])
                
                # Get error bars if available
                if 'ci' in data and isinstance(data['ci'], (list, tuple)) and len(data['ci']) == 2:
                    # CI should be [lower, upper]
                    ci_lower, ci_upper = data['ci']
                    # Error bars are distances from the mean
                    lower_err = data['auc'] - ci_lower
                    upper_err = ci_upper - data['auc']
                    # Ensure non-negative
                    errors.append([max(0, lower_err), max(0, upper_err)])
                elif 'std' in data:
                    errors.append([data['std'], data['std']])
                else:
                    errors.append([0, 0])
            
            # Create bar plot with Nord colors
            x = np.arange(len(methods))
            
            # Color scheme: best method gets special color
            best_idx = np.argmax(aucs)
            colors = [NORD_COLORS['nord14'] if i == best_idx else NORD_COLORS['nord9'] 
                     for i in range(len(methods))]
            
            bars = ax.bar(x, aucs, color=colors, edgecolor=NORD_COLORS['nord3'],
                          linewidth=2, alpha=0.9)
            
            # Add error bars if available
            if errors and any(any(e) for e in errors):
                errors = np.array(errors).T
                ax.errorbar(x, aucs, yerr=errors, fmt='none', 
                           color=NORD_COLORS['nord0'], capsize=8, capthick=2)
            
            # Styling
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=16)
            ax.set_ylabel('AUC', fontsize=20)
            ax.set_title(f'Validation Methods - {model_name.replace("_", " ").title()}', 
                        fontsize=24, fontweight='bold', pad=20)
            ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', 
                      alpha=0.5, label='Random')
            ax.set_ylim([0.4, 1.0])
            
            # Add value labels with CI
            for i, (bar, auc) in enumerate(zip(bars, aucs)):
                if errors and any(errors[0]) and any(errors[1]):
                    ci_text = f'{auc:.3f}\n[{auc - errors[0][i]:.3f}, {auc + errors[1][i]:.3f}]'
                else:
                    ci_text = f'{auc:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       ci_text, ha='center', va='bottom', fontsize=14,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                edgecolor='none', alpha=0.8))
            
            # Add legend
            ax.legend(fontsize=16)
            
            # Add annotation for best method
            if results.get('best_estimate') and results.get('worst_estimate'):
                range_text = f"Range: {results['range']:.3f}"
                ax.text(0.98, 0.02, range_text, transform=ax.transAxes,
                       ha='right', va='bottom', fontsize=14,
                       bbox=dict(boxstyle="round,pad=0.5", 
                                facecolor=NORD_COLORS['nord6'], 
                                edgecolor=NORD_COLORS['nord3'], alpha=0.3))
            
            plt.tight_layout()
            save_beautiful_figure(fig, output_dir / f'validation_comparison_{model_name}')
            plt.close()
        
        # Create summary table
        self._create_validation_summary_table(model_results, output_dir)
    
    def _create_validation_summary_table(self, model_results: Dict, output_dir: Path):
        """Create detailed summary table of validation methods"""
        
        with open(output_dir / 'validation_comparison_table.txt', 'w') as f:
            f.write("VALIDATION METHOD COMPARISON\n")
            f.write("="*80 + "\n\n")
            
            # Overall summary across all models
            all_method_aucs = defaultdict(list)
            
            for model_name, results in model_results.items():
                if 'comparison' in results:
                    f.write(f"\n{model_name.upper()}\n")
                    f.write("-"*50 + "\n")
                    f.write(f"{'Method':<20} {'AUC':<10} {'95% CI / ±SD':<25} {'Notes':<25}\n")
                    f.write("-"*80 + "\n")
                    
                    for method, data in results['comparison'].items():
                        auc = data['auc']
                        all_method_aucs[method].append(auc)
                        
                        # Format uncertainty
                        if 'ci' in data:
                            uncertainty = f"[{data['ci'][0]:.3f}, {data['ci'][1]:.3f}]"
                        elif 'std' in data:
                            uncertainty = f"± {data['std']:.3f}"
                        else:
                            uncertainty = "-"
                        
                        # Add notes
                        notes = ""
                        if method == 'loocv':
                            notes = "Pessimistic, deterministic"
                        elif method == 'mccv':
                            notes = "Standard approach"
                        elif method == 'bootstrap_632':
                            notes = "Reduces bias"
                        elif method == 'bootstrap_632_plus':
                            notes = "Corrects overfitting"
                        
                        f.write(f"{method:<20} {auc:<10.4f} {uncertainty:<25} {notes:<25}\n")
                    
                    if 'range' in results:
                        f.write(f"\nRange: {results['range']:.4f} ")
                        f.write(f"(Best: {results['best_estimate']:.4f}, ")
                        f.write(f"Worst: {results['worst_estimate']:.4f})\n")
            
            # Overall summary
            f.write("\n\nOVERALL SUMMARY ACROSS MODELS\n")
            f.write("="*80 + "\n")
            f.write(f"{'Method':<20} {'Mean AUC':<15} {'SD Across Models':<20}\n")
            f.write("-"*55 + "\n")
            
            for method, aucs in all_method_aucs.items():
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                f.write(f"{method:<20} {mean_auc:<15.4f} {std_auc:<20.4f}\n")
            
            # Recommendations
            f.write("\n\nRECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            f.write("1. Primary reporting: Bootstrap .632+ (accounts for overfitting)\n")
            f.write("2. Sensitivity analysis: LOOCV (conservative estimate)\n")
            f.write("3. Comparison: Monte Carlo CV (standard in literature)\n")
            f.write("\nNote: With 157 samples and 22 events, validation method choice\n")
            f.write("significantly impacts results. Report multiple methods for transparency.\n")
    
    def _create_threshold_report(self, config: Dict, output_dir: Path):
        """Create detailed report for a specific threshold"""
        threshold_dir = output_dir / f"threshold_{config['threshold']:.2f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        with open(threshold_dir / 'detailed_report.txt', 'w') as f:
            f.write(f"THRESHOLD {config['threshold']} DETAILED REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Configuration details
            f.write("CONFIGURATION\n")
            f.write("-"*30 + "\n")
            f.write(f"Regularization: {config['regularization']}\n")
            f.write(f"STABL threshold: {config['threshold']}\n")
            f.write(f"Features selected: {config['n_features']}\n")
            f.write(f"Ensemble runs: {config['ensemble'].n_runs}\n\n")
            
            # Selected features
            f.write("SELECTED FEATURES\n")
            f.write("-"*30 + "\n")
            for i, feat in enumerate(config['features'], 1):
                # Get selection frequency if available
                freq = config['ensemble'].feature_frequencies.get(feat, 0) / config['ensemble'].n_runs
                f.write(f"{i:3d}. {feat:<50} (freq: {freq:.2f})\n")
            
            # Model performance with CI
            f.write("\n\nMODEL PERFORMANCE WITH CONFIDENCE INTERVALS\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Model':<20} {'Method':<20} {'AUC':<10} {'95% CI':<20} {'p-value':<10}\n")
            f.write("-"*70 + "\n")
            
            for model_name, results in config['model_results'].items():
                method = results.get('method', 'Unknown')
                auc = results.get('auc', 0)
                
                # Get CI
                if 'auc_ci' in results:
                    ci = results['auc_ci']
                    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                elif 'comparison' in results and results['comparison']:
                    # For 'all' method, get best CI
                    best_ci = [1.0, 0.0]
                    for method_data in results['comparison'].values():
                        if 'ci' in method_data:
                            best_ci[0] = min(best_ci[0], method_data['ci'][0])
                            best_ci[1] = max(best_ci[1], method_data['ci'][1])
                    ci_str = f"[{best_ci[0]:.3f}, {best_ci[1]:.3f}]"
                else:
                    ci_str = "N/A"
                
                # Get p-value
                p_val = "N/A"
                if 'permutation' in results:
                    p_val = f"{results['permutation']['p_value']:.4f}"
                
                f.write(f"{model_name:<20} {method:<20} {auc:<10.4f} {ci_str:<20} {p_val:<10}\n")
            
            # Validation method comparison if available
            if any('comparison' in r for r in config['model_results'].values()):
                f.write("\n\nVALIDATION METHOD COMPARISON\n")
                f.write("-"*70 + "\n")
                
                for model_name, results in config['model_results'].items():
                    if 'comparison' in results:
                        f.write(f"\n{model_name.upper()}\n")
                        f.write(f"{'Method':<25} {'AUC':<10} {'95% CI':<20}\n")
                        f.write("-"*55 + "\n")
                        
                        for val_method, val_data in results['comparison'].items():
                            auc = val_data['auc']
                            if 'ci' in val_data:
                                ci_str = f"[{val_data['ci'][0]:.3f}, {val_data['ci'][1]:.3f}]"
                            else:
                                ci_str = "N/A"
                            f.write(f"{val_method:<25} {auc:<10.4f} {ci_str:<20}\n")
        
        # Save metrics as CSV
        metrics_data = []
        for model_name, results in config['model_results'].items():
            row = {
                'model': model_name,
                'threshold': config['threshold'],
                'n_features': config['n_features'],
                'auc': results.get('auc', 0),
                'method': results.get('method', 'Unknown')
            }
            
            # Add CI if available
            if 'auc_ci' in results:
                row['ci_lower'] = results['auc_ci'][0]
                row['ci_upper'] = results['auc_ci'][1]
            
            # Add p-value if available
            if 'permutation' in results:
                row['p_value'] = results['permutation']['p_value']
                row['significant'] = results['permutation']['significant']
            
            metrics_data.append(row)
        
        pd.DataFrame(metrics_data).to_csv(threshold_dir / 'metrics.csv', index=False)
    
    def _create_threshold_summary_plot(self, config: Dict, output_dir: Path):
        """Create multi-panel summary figure for a specific threshold"""
        threshold_dir = output_dir / f"threshold_{config['threshold']:.2f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 2x2 subplot figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel A: Model performance with CI
        ax1 = fig.add_subplot(gs[0, 0])
        self._add_model_performance_panel(ax1, config)
        ax1.text(-0.1, 1.05, 'A', transform=ax1.transAxes, fontsize=24, fontweight='bold')
        
        # Panel B: Feature selection consistency
        ax2 = fig.add_subplot(gs[0, 1])
        self._add_feature_consistency_panel(ax2, config)
        ax2.text(-0.1, 1.05, 'B', transform=ax2.transAxes, fontsize=24, fontweight='bold')
        
        # Panel C: Validation method comparison
        ax3 = fig.add_subplot(gs[1, 0])
        self._add_validation_comparison_panel(ax3, config)
        ax3.text(-0.1, 1.05, 'C', transform=ax3.transAxes, fontsize=24, fontweight='bold')
        
        # Panel D: Summary statistics
        ax4 = fig.add_subplot(gs[1, 1])
        self._add_summary_stats_panel(ax4, config)
        ax4.text(-0.1, 1.05, 'D', transform=ax4.transAxes, fontsize=24, fontweight='bold')
        
        # Main title
        fig.suptitle(f'Threshold {config["threshold"]} - Comprehensive Analysis', 
                    fontsize=28, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        save_beautiful_figure(fig, threshold_dir / 'performance_summary')
        plt.close()
    
    def _add_model_performance_panel(self, ax, config: Dict):
        """Add model performance comparison with CI error bars"""
        models = []
        aucs = []
        ci_lower = []
        ci_upper = []
        significant = []
        
        for model_name, results in config['model_results'].items():
            models.append(model_name.replace('_', ' ').title())
            aucs.append(results.get('auc', 0))
            
            # Get CI
            if 'auc_ci' in results:
                ci_lower.append(results['auc_ci'][0])
                ci_upper.append(results['auc_ci'][1])
            else:
                ci_lower.append(results.get('auc', 0))
                ci_upper.append(results.get('auc', 0))
            
            # Check significance
            if 'permutation' in results:
                significant.append(results['permutation']['significant'])
            else:
                significant.append(False)
        
        # Create bar plot with error bars
        x = np.arange(len(models))
        colors = [NORD_COLORS['nord14'] if sig else NORD_COLORS['nord9'] 
                 for sig in significant]
        
        bars = ax.bar(x, aucs, color=colors, edgecolor=NORD_COLORS['nord3'],
                      linewidth=2, alpha=0.9)
        
        # Add error bars
        errors = [[auc - lower for auc, lower in zip(aucs, ci_lower)],
                 [upper - auc for auc, upper in zip(aucs, ci_upper)]]
        ax.errorbar(x, aucs, yerr=errors, fmt='none', 
                   color=NORD_COLORS['nord0'], capsize=8, capthick=3)
        
        # Add value labels
        for i, (bar, auc, lower, upper) in enumerate(zip(bars, aucs, ci_lower, ci_upper)):
            label = f'{auc:.3f}\n[{lower:.3f}, {upper:.3f}]'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   label, ha='center', va='bottom', fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                            edgecolor='none', alpha=0.8))
            
            # Add significance star
            if significant[i]:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                       '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        # Styling
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=14)
        ax.set_ylabel('AUC', fontsize=18)
        ax.set_title('Model Performance Comparison', fontsize=20, fontweight='bold')
        ax.set_ylim([0.4, 1.0])
        ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', 
                  alpha=0.5, label='Random')
        ax.axhline(0.7, color=NORD_COLORS['nord13'], linestyle='--', 
                  alpha=0.5, label='Acceptable')
        ax.legend(fontsize=14)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_axisbelow(True)
    
    def _add_feature_consistency_panel(self, ax, config: Dict):
        """Add feature selection consistency plot"""
        if 'ensemble' in config and hasattr(config['ensemble'], 'feature_frequencies'):
            # Get top 15 features
            freq_items = list(config['ensemble'].feature_frequencies.items())
            freq_items.sort(key=lambda x: x[1], reverse=True)
            top_features = freq_items[:15]
            
            features = [f[0] for f in top_features]
            frequencies = [f[1] / config['ensemble'].n_runs for f in top_features]
            
            # Color based on selection in final model
            colors = [NORD_COLORS['nord14'] if feat in config['features'] else NORD_COLORS['nord9']
                     for feat in features]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, frequencies, color=colors,
                          edgecolor=NORD_COLORS['nord3'], linewidth=1.5, alpha=0.9)
            
            # Format feature names
            formatted_features = []
            for feat in features:
                if len(feat) > 40:
                    feat = feat[:37] + '...'
                formatted_features.append(feat)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(formatted_features, fontsize=11)
            ax.set_xlabel('Selection Frequency', fontsize=16)
            ax.set_title(f'Feature Consistency (n={config["ensemble"].n_runs} runs)', 
                        fontsize=18, fontweight='bold')
            ax.set_xlim([0, 1.1])
            
            # Add threshold line
            ax.axvline(0.8, color=NORD_COLORS['nord11'], linestyle='--',
                      alpha=0.5, label='Consensus threshold')
            
            # Add value labels
            for bar, freq in zip(bars, frequencies):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{freq:.2f}', va='center', ha='left', fontsize=11)
            
            ax.legend(fontsize=12)
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_axisbelow(True)
        else:
            ax.text(0.5, 0.5, 'Feature frequency data not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.axis('off')
    
    def _add_validation_comparison_panel(self, ax, config: Dict):
        """Add validation method comparison for best model"""
        # Find best model
        best_model = None
        best_auc = 0
        
        for model_name, results in config['model_results'].items():
            if results.get('auc', 0) > best_auc:
                best_auc = results.get('auc', 0)
                best_model = (model_name, results)
        
        if best_model and 'comparison' in best_model[1]:
            model_name, results = best_model
            
            methods = []
            aucs = []
            cis = []
            
            for method, data in results['comparison'].items():
                methods.append(method.replace('_', ' ').title())
                aucs.append(data['auc'])
                
                if 'ci' in data:
                    cis.append(data['ci'])
                else:
                    cis.append((data['auc'], data['auc']))
            
            # Create bar plot
            x = np.arange(len(methods))
            colors = [NORD_COLORS['nord14'] if auc == max(aucs) else NORD_COLORS['nord9']
                     for auc in aucs]
            
            bars = ax.bar(x, aucs, color=colors, edgecolor=NORD_COLORS['nord3'],
                          linewidth=2, alpha=0.9)
            
            # Add error bars
            if cis:
                errors = [[auc - ci[0] for auc, ci in zip(aucs, cis)],
                         [ci[1] - auc for auc, ci in zip(aucs, cis)]]
                ax.errorbar(x, aucs, yerr=errors, fmt='none',
                           color=NORD_COLORS['nord0'], capsize=6, capthick=2)
            
            # Add value labels
            for bar, auc, ci in zip(bars, aucs, cis):
                label = f'{auc:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       label, ha='center', va='bottom', fontsize=14, fontweight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel('AUC', fontsize=16)
            ax.set_title(f'Validation Methods - {model_name.replace("_", " ").title()}',
                        fontsize=18, fontweight='bold')
            ax.set_ylim([0.4, 1.0])
            ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5)
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_axisbelow(True)
        else:
            ax.text(0.5, 0.5, 'Validation comparison not available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.axis('off')
    
    def _add_summary_stats_panel(self, ax, config: Dict):
        """Add summary statistics panel"""
        ax.axis('off')
        
        # Prepare summary text
        summary_text = f"""Configuration Summary
        
Threshold: {config['threshold']}
Regularization: {config['regularization']}
Features Selected: {config['n_features']}
Ensemble Runs: {config.get('ensemble', {}).n_runs if 'ensemble' in config else 'N/A'}

Best Model Performance:
"""
        
        # Find best model
        best_model = None
        best_auc = 0
        
        for model_name, results in config['model_results'].items():
            if results.get('auc', 0) > best_auc:
                best_auc = results['auc']
                best_model = model_name
                best_results = results
        
        if best_model:
            summary_text += f"\nModel: {best_model.replace('_', ' ').title()}"
            summary_text += f"\nAUC: {best_auc:.4f}"
            
            if 'auc_ci' in best_results:
                ci = best_results['auc_ci']
                summary_text += f"\n95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]"
            
            if 'permutation' in best_results:
                perm = best_results['permutation']
                summary_text += f"\n\nPermutation Test:"
                summary_text += f"\np-value: {perm['p_value']:.4f}"
                summary_text += f"\nSignificant: {'Yes' if perm['significant'] else 'No'}"
        
        # Add selected features (top 5)
        if config['features']:
            summary_text += f"\n\nTop Features:"
            for i, feat in enumerate(config['features'][:5], 1):
                if len(feat) > 35:
                    feat = feat[:32] + '...'
                summary_text += f"\n{i}. {feat}"
            
            if len(config['features']) > 5:
                summary_text += f"\n... and {len(config['features']) - 5} more"
        
        # Display text
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
               fontsize=14, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor=NORD_COLORS['nord6'],
                        edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.3))
    
    def _create_threshold_performance_plot(self, config: Dict, output_dir: Path):
        """Create detailed performance metrics plot with CI distributions"""
        threshold_dir = output_dir / f"threshold_{config['threshold']:.2f}"
        threshold_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data for all models
        model_names = []
        all_aucs = []
        all_cis = []
        all_methods = []
        
        for model_name, results in config['model_results'].items():
            model_names.append(model_name.replace('_', ' ').title())
            
            # Get main AUC and CI
            all_aucs.append(results.get('auc', 0))
            
            if 'auc_ci' in results:
                all_cis.append(results['auc_ci'])
            else:
                all_cis.append((results.get('auc', 0), results.get('auc', 0)))
            
            # Get validation method if available
            all_methods.append(results.get('method', 'Unknown'))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left panel: Forest plot style CI visualization
        y_pos = np.arange(len(model_names))
        
        # Plot CI lines
        for i, (auc, ci) in enumerate(zip(all_aucs, all_cis)):
            # CI line
            ax1.plot([ci[0], ci[1]], [i, i], 'k-', linewidth=3, alpha=0.7)
            
            # CI endpoints
            ax1.plot(ci[0], i, 'k|', markersize=12, markeredgewidth=2)
            ax1.plot(ci[1], i, 'k|', markersize=12, markeredgewidth=2)
            
            # Point estimate
            color = NORD_COLORS['nord14'] if auc > 0.7 else NORD_COLORS['nord13'] if auc > 0.6 else NORD_COLORS['nord11']
            ax1.plot(auc, i, 'o', color=color, markersize=12, 
                    markeredgecolor=NORD_COLORS['nord3'], markeredgewidth=2)
            
            # Add value label
            ax1.text(ci[1] + 0.01, i, f'{auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]',
                    va='center', ha='left', fontsize=12)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(model_names, fontsize=14)
        ax1.set_xlabel('AUC', fontsize=16)
        ax1.set_title('Model Performance with 95% Confidence Intervals', fontsize=18, fontweight='bold')
        ax1.axvline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
        ax1.axvline(0.7, color=NORD_COLORS['nord13'], linestyle='--', alpha=0.5, label='Acceptable')
        ax1.set_xlim([0.4, 1.0])
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.legend(fontsize=12)
        
        # Right panel: CI width comparison
        ci_widths = [ci[1] - ci[0] for ci in all_cis]
        colors = [NORD_COLORS['nord14'] if w < 0.1 else NORD_COLORS['nord13'] if w < 0.15 else NORD_COLORS['nord11']
                 for w in ci_widths]
        
        bars = ax2.barh(y_pos, ci_widths, color=colors,
                       edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
        
        # Add value labels
        for bar, width in zip(bars, ci_widths):
            ax2.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{width:.3f}', va='center', ha='left', fontsize=12)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(model_names, fontsize=14)
        ax2.set_xlabel('CI Width', fontsize=16)
        ax2.set_title('Confidence Interval Width (Uncertainty)', fontsize=18, fontweight='bold')
        ax2.set_xlim([0, max(ci_widths) * 1.2 if ci_widths else 0.3])
        ax2.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_beautiful_figure(fig, threshold_dir / 'model_performance_detailed')
        plt.close()
    
    def _create_cross_threshold_comparison(self, config_results: List[Dict], output_dir: Path):
        """Create comparison plots across all thresholds"""
        comparison_dir = output_dir / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect data across thresholds
        thresholds = []
        best_aucs = []
        n_features = []
        model_performances = defaultdict(lambda: {'thresholds': [], 'aucs': [], 'cis': []})
        
        for config in sorted(config_results, key=lambda x: x['threshold']):
            thresholds.append(config['threshold'])
            best_aucs.append(config['best_auc'])
            n_features.append(config['n_features'])
            
            # Collect per-model data
            for model_name, results in config['model_results'].items():
                model_performances[model_name]['thresholds'].append(config['threshold'])
                model_performances[model_name]['aucs'].append(results.get('auc', 0))
                
                if 'auc_ci' in results:
                    model_performances[model_name]['cis'].append(results['auc_ci'])
                else:
                    auc = results.get('auc', 0)
                    model_performances[model_name]['cis'].append((auc, auc))
        
        # Create multi-panel figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Panel 1: AUC trajectory across thresholds
        ax1 = fig.add_subplot(gs[0, :])
        
        for model_name, data in model_performances.items():
            # Plot line with CI band
            aucs = np.array(data['aucs'])
            cis = np.array(data['cis'])
            
            label = model_name.replace('_', ' ').title()
            color = COLOR_SCHEMES['models'][list(model_performances.keys()).index(model_name) % len(COLOR_SCHEMES['models'])]
            
            ax1.plot(data['thresholds'], aucs, 'o-', color=color, label=label,
                    linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=2)
            
            # Add confidence band
            ax1.fill_between(data['thresholds'], cis[:, 0], cis[:, 1], 
                           color=color, alpha=0.2)
        
        ax1.set_xlabel('STABL Threshold', fontsize=18)
        ax1.set_ylabel('AUC', fontsize=18)
        ax1.set_title('Model Performance Across Thresholds', fontsize=22, fontweight='bold')
        ax1.set_ylim([0.5, 1.0])
        ax1.axhline(0.7, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5)
        ax1.legend(fontsize=14, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)
        
        # Panel 2: Feature count vs performance
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Create scatter plot with threshold labels
        scatter = ax2.scatter(n_features, best_aucs, s=300, c=thresholds,
                            cmap='viridis', edgecolor=NORD_COLORS['nord3'],
                            linewidth=3, alpha=0.9, zorder=10)
        
        # Connect points
        ax2.plot(n_features, best_aucs, 'k--', alpha=0.3, zorder=1)
        
        # Add threshold labels
        for nf, auc, thresh in zip(n_features, best_aucs, thresholds):
            ax2.annotate(f'{thresh}', (nf, auc), xytext=(5, 5),
                       textcoords='offset points', fontsize=14,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                                edgecolor='none', alpha=0.8))
        
        ax2.set_xlabel('Number of Features', fontsize=18)
        ax2.set_ylabel('Best AUC', fontsize=18)
        ax2.set_title('Performance vs Model Complexity', fontsize=20, fontweight='bold')
        ax2.axhspan(0.7, 0.8, alpha=0.1, color=NORD_COLORS['nord13'])
        ax2.axhspan(0.8, 1.0, alpha=0.1, color=NORD_COLORS['nord14'])
        ax2.axvspan(5, 10, alpha=0.1, color=NORD_COLORS['nord14'])
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Threshold', fontsize=14)
        
        # Panel 3: CI width comparison
        ax3 = fig.add_subplot(gs[1, 1])
        
        # Calculate average CI width for each threshold
        avg_ci_widths = []
        for config in sorted(config_results, key=lambda x: x['threshold']):
            ci_widths = []
            for model_name, results in config['model_results'].items():
                if 'auc_ci' in results:
                    ci = results['auc_ci']
                    ci_widths.append(ci[1] - ci[0])
            avg_ci_widths.append(np.mean(ci_widths) if ci_widths else 0)
        
        bars = ax3.bar(range(len(thresholds)), avg_ci_widths,
                       color=[NORD_COLORS['nord14'] if w < 0.1 else NORD_COLORS['nord13'] 
                             if w < 0.15 else NORD_COLORS['nord11'] for w in avg_ci_widths],
                       edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
        
        # Add value labels
        for bar, width in zip(bars, avg_ci_widths):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{width:.3f}', ha='center', va='bottom', fontsize=14)
        
        ax3.set_xticks(range(len(thresholds)))
        ax3.set_xticklabels([f'{t:.2f}' for t in thresholds], fontsize=14)
        ax3.set_xlabel('STABL Threshold', fontsize=18)
        ax3.set_ylabel('Average CI Width', fontsize=18)
        ax3.set_title('Prediction Uncertainty by Threshold', fontsize=20, fontweight='bold')
        ax3.grid(True, axis='y', alpha=0.3)
        ax3.set_axisbelow(True)
        
        plt.tight_layout()
        save_beautiful_figure(fig, comparison_dir / 'auc_trajectory')
        plt.close()
        
        # Save comparison data as CSV
        comparison_data = []
        for config in config_results:
            for model_name, results in config['model_results'].items():
                row = {
                    'threshold': config['threshold'],
                    'model': model_name,
                    'n_features': config['n_features'],
                    'auc': results.get('auc', 0),
                    'method': results.get('method', 'Unknown')
                }
                
                if 'auc_ci' in results:
                    row['ci_lower'] = results['auc_ci'][0]
                    row['ci_upper'] = results['auc_ci'][1]
                    row['ci_width'] = results['auc_ci'][1] - results['auc_ci'][0]
                
                if 'permutation' in results:
                    row['p_value'] = results['permutation']['p_value']
                    row['significant'] = results['permutation']['significant']
                
                comparison_data.append(row)
        
        pd.DataFrame(comparison_data).to_csv(comparison_dir / 'comprehensive_comparison.csv', index=False)
    
    def _create_threshold_ci_comparison(self, config_results: List[Dict], output_dir: Path):
        """Create forest plot style CI comparison across thresholds"""
        comparison_dir = output_dir / 'comparison'
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for forest plot
        plot_data = []
        y_labels = []
        y_pos = 0
        
        # Group by model type
        model_groups = defaultdict(list)
        for config in sorted(config_results, key=lambda x: x['threshold']):
            for model_name, results in config['model_results'].items():
                model_groups[model_name].append({
                    'threshold': config['threshold'],
                    'auc': results.get('auc', 0),
                    'ci': results.get('auc_ci', (results.get('auc', 0), results.get('auc', 0))),
                    'significant': results.get('permutation', {}).get('significant', False)
                })
        
        # Create figure
        fig, ax = create_beautiful_figure('tall')
        
        # Plot each model group
        colors = COLOR_SCHEMES['models']
        for model_idx, (model_name, threshold_data) in enumerate(model_groups.items()):
            model_color = colors[model_idx % len(colors)]
            
            # Add model name as group header
            y_labels.append(f"**{model_name.replace('_', ' ').title()}**")
            y_pos += 1
            
            # Plot each threshold for this model
            for data in threshold_data:
                # Extract data
                auc = data['auc']
                ci = data['ci']
                threshold = data['threshold']
                significant = data['significant']
                
                # Plot CI line
                ax.plot([ci[0], ci[1]], [y_pos, y_pos], color=model_color, 
                       linewidth=3, alpha=0.8)
                
                # Plot CI endpoints
                ax.plot(ci[0], y_pos, '|', color=model_color, markersize=10, 
                       markeredgewidth=2)
                ax.plot(ci[1], y_pos, '|', color=model_color, markersize=10, 
                       markeredgewidth=2)
                
                # Plot point estimate
                marker = '*' if significant else 'o'
                ax.plot(auc, y_pos, marker, color=model_color, markersize=12,
                       markeredgecolor=NORD_COLORS['nord3'], markeredgewidth=2)
                
                # Add label
                y_labels.append(f"  Threshold {threshold}")
                
                # Add AUC and CI text
                ax.text(ci[1] + 0.01, y_pos, f'{auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]',
                       va='center', ha='left', fontsize=11)
                
                y_pos += 1
            
            # Add spacing between models
            y_pos += 0.5
        
        # Styling
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels[::-1], fontsize=12)
        
        # Make model names bold
        for i, label in enumerate(ax.get_yticklabels()):
            if label.get_text().startswith('**'):
                label.set_text(label.get_text().strip('*'))
                label.set_fontweight('bold')
                label.set_fontsize(14)
        
        ax.set_xlabel('AUC with 95% Confidence Interval', fontsize=18)
        ax.set_title('Model Performance Across All Thresholds', fontsize=22, fontweight='bold')
        ax.axvline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
        ax.axvline(0.7, color=NORD_COLORS['nord13'], linestyle='--', alpha=0.5, label='Acceptable')
        ax.set_xlim([0.4, 1.0])
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Add legend
        handles = []
        for model_idx, model_name in enumerate(model_groups.keys()):
            color = colors[model_idx % len(colors)]
            handles.append(plt.Line2D([0], [0], color=color, linewidth=3,
                                    label=model_name.replace('_', ' ').title()))
        
        ax.legend(handles=handles, fontsize=12, loc='lower right')
        
        # Add footnote about significance
        ax.text(0.02, 0.02, '* indicates p < 0.05 vs random', transform=ax.transAxes,
               fontsize=10, style='italic', ha='left', va='bottom')
        
        plt.tight_layout()
        save_beautiful_figure(fig, comparison_dir / 'ci_forest_plot')
        plt.close()


def main():
    """Run ultra-optimized pipeline with example data"""
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Ultra-Optimized STABL Pipeline')
    
    # Data input & alignment (radiomics-only)
    parser.add_argument('--radiomics-path', type=str, default='data/radiomics_prefiltered.csv',
                       help='Path to radiomics CSV (preferred prefiltered)')
    parser.add_argument('--matches-path', type=str, default='data/POPF-SCANNER.csv',
                       help='Path to POPF outcomes CSV (scanner_patient_name, popf_grade)')
    parser.add_argument('--positive-grades', type=str, default='B,C',
                       help="Comma-separated POPF grades considered positive (default: 'B,C'). e.g., 'B,C,BL'")
    parser.add_argument('--allow-id-normalization', action='store_true', default=False,
                       help='Attempt normalized ID fallback join when exact match fails (default: off)')
    parser.add_argument('--texture-only', action='store_true', default=False,
                       help='Restrict analysis to texture features only (GLCM/GLRLM/GLSZM/GLDM/NGTDM).')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results')
    
    # Feature selection arguments
    parser.add_argument('--mode', type=str, default='radiomics',
                       choices=['clinical', 'radiomics', 'both'],
                       help='Feature selection mode')
    parser.add_argument('--n-features', type=int, default=10,
                       help='Number of features to select')
    parser.add_argument('--stabl-regularization', type=str, default='l1',
                       choices=['l1', 'elasticnet', 'adaptive_lasso'],
                       help='Regularization type for STABL base estimator (default: l1)')
    parser.add_argument('--stabl-l1-ratio', type=float, default=0.5,
                       help='l1_ratio for elasticnet (used when --stabl-regularization elasticnet)')
    parser.add_argument('--stabl-c', type=float, default=0.5,
                       help='Inverse regularization strength C for STABL base estimator (smaller is stronger)')
    parser.add_argument('--stabl-tol', type=float, default=1e-3,
                       help='Tolerance for solver convergence (elasticnet uses saga)')
    parser.add_argument('--stabl-max-iter', type=int, default=5000,
                       help='Max iterations for STABL base estimator solver')
    parser.add_argument('--consensus-threshold', type=float, default=0.8,
                       help='Consensus frequency threshold across ensemble runs (default: 0.8)')
    # Optional FDR/lambda grid controls for STABL
    parser.add_argument('--stabl-artificial-type', type=str, default='none',
                       choices=['none', 'random_permutation', 'knockoff'],
                       help='Use artificial features (random_permutation/knockoff) to enable FDR control in STABL')
    parser.add_argument('--lambda-grid', type=str, default=None,
                       help="Lambda grid for STABL; set to 'auto' to auto-generate a path")
    parser.add_argument('--n-lambda', type=int, default=30,
                       help='Number of lambda values when lambda-grid=auto')
    parser.add_argument('--fdr-start', type=float, default=0.10,
                       help='Start of FDR threshold range (if FDR enabled)')
    parser.add_argument('--fdr-end', type=float, default=0.30,
                       help='End of FDR threshold range (if FDR enabled)')
    parser.add_argument('--fdr-step', type=float, default=0.01,
                       help='Step of FDR threshold range (if FDR enabled)')
    parser.add_argument('--stabl-corr-group-threshold', type=float, default=None,
                       help='Percentile threshold for correlation grouping inside STABL (e.g., 90). None disables grouping')
    parser.add_argument('--stabl-sample-fraction', type=float, default=0.75,
                       help='Sample fraction for STABL bootstraps (default 0.75)')
    parser.add_argument('--thresholds', type=float, nargs='+', default=None,
                       help='STABL thresholds to test (default: 0.7 0.75 0.8 0.9)')
    parser.add_argument('--evaluation', type=str, default='loocv',
                       choices=['loocv', 'bootstrap_632', 'bootstrap_632_plus', 'mccv', 'all'],
                       help='Evaluation method (exploratory only; not for headline)')
    
    # Advanced arguments (keeping original names for compatibility)
    parser.add_argument('--ensemble-runs', type=int, default=10,
                       help='Number of STABL runs per configuration')
    parser.add_argument('--n-jobs', type=int, default=-1,
                       help='Number of CPU cores for STABL (-1 for all cores)')
    parser.add_argument('--max-features', type=int, default=None,
                       help='Maximum features to select (overrides n-features if set)')
    parser.add_argument('--validation', type=str, default=None,
                       help='Validation method (overrides evaluation if set)')
    parser.add_argument('--n-mccv', type=int, default=200,
                       help='Number of Monte Carlo CV iterations')
    parser.add_argument('--mccv-test-size', type=float, default=0.2,
                       help='Test set size for Monte Carlo CV')
    parser.add_argument('--permutations', type=int, default=1000,
                       help='Number of permutations for significance testing')

    # Optional unsupervised prefilter (missing/variance/correlation)
    parser.add_argument('--prefilter', action='store_true',
                       help='Apply unsupervised prefilter (missing/variance/correlation) before modeling')
    parser.add_argument('--prefilter-missing-threshold', type=float, default=0.30,
                       help='Maximum missing fraction per feature to keep (default: 0.30)')
    parser.add_argument('--prefilter-variance-threshold', type=float, default=0.01,
                       help='Minimum variance to keep (after median imputation; default: 0.01)')
    parser.add_argument('--prefilter-correlation-threshold', type=float, default=0.95,
                       help='Maximum absolute Spearman correlation to allow (default: 0.95)')
    parser.add_argument('--cv-prefilter-train-only', action='store_true', default=False,
                       help='In CV mode, apply prefilter on training fold only (instead of globally)')
    parser.add_argument('--temporal-prefilter-train-only', action='store_true', default=True,
                       help='In temporal holdout, apply prefilter on training (oldest) only (default: True)')

    # Discovery/export and temporal evaluation
    parser.add_argument('--discovery-only', action='store_true',
                       help='Skip internal validation; export panel and frequencies only')
    parser.add_argument('--export-panel', type=str, default=None,
                       help='Optional path to write panel as a txt (one feature per line)')
    parser.add_argument('--temporal-holdout', action='store_true',
                       help='Perform temporal evaluation: train on oldest, test on newest')
    parser.add_argument('--scanner-metadata-path', type=str, default=None,
                       help='CSV with scanner dates for temporal split')
    parser.add_argument('--scanner-id-col', type=str, default='scanner_patient_name',
                       help='ID column in scanner metadata (default: scanner_patient_name)')
    parser.add_argument('--date-col', type=str, default='StudyDate',
                       help='Date column in scanner metadata (default: StudyDate)')
    parser.add_argument('--holdout-fraction', type=float, default=0.3,
                       help='Fraction of newest patients for temporal holdout (default: 0.3)')
    # Optional ComBat on training set only
    parser.add_argument('--combat-train-only', action='store_true', default=False,
                       help='Apply ComBat harmonization fit on training set only, then transform test')
    parser.add_argument('--combat-metadata-path', type=str, default=None,
                       help='CSV with batch labels for ComBat (defaults to --scanner-metadata-path if not provided)')
    parser.add_argument('--combat-batch-col', type=str, default=None,
                       help='Column in metadata to use as ComBat batch (e.g., Manufacturer/Model/ScannerType)')
    # Optional reference-based standardization (train-only, ComBat-ref light)
    parser.add_argument('--ref-standardize-train-only', action='store_true', default=False,
                       help='Align batches to a reference batch using train-only per-feature mean/SD (ComBat-ref light)')
    parser.add_argument('--ref-batch-col', type=str, default=None,
                       help='Metadata column for reference batch labels (defaults to --combat-batch-col if omitted)')
    parser.add_argument('--ref-batch-value', type=str, default=None,
                       help='Reference batch value (e.g., a specific Manufacturer/Model) to align to')
    parser.add_argument('--train-radiomics-path', type=str, default=None,
                       help='Radiomics CSV to use for TRAINING set in temporal holdout (e.g., harmonized)')
    parser.add_argument('--test-radiomics-path', type=str, default=None,
                       help='Radiomics CSV to use for TEST set in temporal holdout (e.g., non-harmonized)')
    # Scanner-type validation (domain holdout)
    parser.add_argument('--scanner-type-validation', action='store_true',
                       help='Train on dominant scanner type and validate on all other scanner types')
    parser.add_argument('--scanner-type-col', type=str, default='ScannerType',
                       help='Scanner type column in scanner metadata (e.g., manufacturer/model/vendor)')
    parser.add_argument('--dominant-scanner-value', type=str, default=None,
                       help='Optional explicit dominant scanner type to train on; defaults to most frequent type')
    # Repeated CV (train-only selection per fold)
    parser.add_argument('--run-cv', action='store_true',
                       help='Run repeated stratified CV with train-only STABL selection per fold')
    parser.add_argument('--cv-model', choices=['lr', 'xgb'], default='lr',
                       help='Model for CV mode: lr (STABL+LR) or xgb (univariate+XGBoost)')
    parser.add_argument('--cv-splits', type=int, default=5,
                       help='Number of folds for repeated stratified CV (default: 5)')
    parser.add_argument('--cv-repeats', type=int, default=5,
                       help='Number of repeats for repeated stratified CV (default: 5)')
    parser.add_argument('--save-fold-selections', action='store_true', default=False,
                       help='Save per-fold selected feature lists to cv_eval.json')
    parser.add_argument('--cv-grouped', action='store_true', default=False,
                       help='Use grouped CV by scanner type to reduce scanner leakage (requires --scanner-metadata-path)')
    # CV feature selection for XGB
    parser.add_argument('--panel-size', type=int, default=6,
                       help='Final panel size per fold (features kept for model)')
    parser.add_argument('--topk-multiplier', type=float, default=2.0,
                       help='Top-K = min(max_topk, topk_multiplier * events_in_train)')
    parser.add_argument('--max-topk', type=int, default=20,
                       help='Upper bound for Top-K before shrinking to panel size')
    parser.add_argument('--min-train-auc', type=float, default=0.52,
                       help='Minimum train fold univariate AUC to consider a feature; below uses tie-breakers or is dropped')
    parser.add_argument('--nzv-var-threshold', type=float, default=1e-8,
                       help='Near-zero-variance threshold (drop features with variance below)')
    parser.add_argument('--min-unique-ratio', type=float, default=0.01,
                       help='Drop features with unique values / n_train below this ratio')
    # XGBoost hyperparameters
    parser.add_argument('--xgb-depth', type=int, default=3,
                       help='XGBoost max_depth')
    parser.add_argument('--xgb-n-estimators', type=int, default=400,
                       help='XGBoost n_estimators (max rounds)')
    parser.add_argument('--xgb-lr', type=float, default=0.05,
                       help='XGBoost learning rate')
    parser.add_argument('--xgb-subsample', type=float, default=0.7,
                       help='XGBoost subsample')
    parser.add_argument('--xgb-colsample', type=float, default=0.6,
                       help='XGBoost colsample_bytree')
    parser.add_argument('--xgb-reg-alpha', type=float, default=2.0,
                       help='XGBoost L1 regularization (reg_alpha)')
    parser.add_argument('--xgb-reg-lambda', type=float, default=2.0,
                       help='XGBoost L2 regularization (reg_lambda)')
    parser.add_argument('--xgb-min-child-weight', type=float, default=1.0,
                       help='XGBoost min_child_weight')
    parser.add_argument('--xgb-early-stopping', type=int, default=30,
                       help='Early stopping rounds for XGBoost (uses train/dev split within training fold)')
    parser.add_argument('--use-l1-shrink', action='store_true', default=True,
                       help='Apply L1-logistic shrinkage on Top-K features to reach final panel size')
    # Train-only LR tuning and test CI (temporal)
    parser.add_argument('--lr-tune', action='store_true',
                       help='Tune LogisticRegression C on training set only (temporal mode)')
    parser.add_argument('--lr-penalty', choices=['l1', 'l2'], default='l1',
                       help='Penalty for LogisticRegression in temporal eval (default: l1)')
    parser.add_argument('--lr-c-grid', nargs='+', type=float, default=[0.1, 0.25, 0.5, 1, 2.5, 5],
                       help='Grid of C values for LR tuning on training set')
    parser.add_argument('--test-bootstrap', type=int, default=0,
                       help='Number of bootstrap resamples on the test set to compute AUROC CI (0=disable)')
    # Optuna controls for LR tuning
    parser.add_argument('--lr-optuna-trials', type=int, default=30,
                       help='Number of Optuna trials when tuning LR C (train-only)')
    parser.add_argument('--lr-optuna-seed', type=int, default=42,
                       help='Random seed for Optuna LR tuning')
    parser.add_argument('--lr-optuna-c-min', type=float, default=1e-3,
                       help='Lower bound for Optuna search over LogisticRegression C (log-uniform)')
    parser.add_argument('--lr-optuna-c-max', type=float, default=1e2,
                       help='Upper bound for Optuna search over LogisticRegression C (log-uniform)')
    parser.add_argument('--temporal-extra-features', type=int, default=0,
                       help='Append top-N non-consensus STABL features (by frequency) to the temporal panel before LR fitting')
    
    args = parser.parse_args()
    
    # Handle argument overrides for backwards compatibility
    if args.max_features is not None:
        args.n_features = args.max_features
    if args.validation is not None:
        args.evaluation = args.validation
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output_dir = Path("results") / f"ultra_optimized_{timestamp}"
    else:
        args.output_dir = Path(args.output_dir)
    
    # Load radiomics + labels (exact match; drop unmatched rows)
    radiomics_path = Path(args.radiomics_path)
    matches_path = Path(args.matches_path)
    print("Loading radiomics and outcomes (EXACT ID match)...")
    df = pd.read_csv(radiomics_path)
    # Normalize radiomics ID column name
    if 'scanner_patient_name' in df.columns:
        rid = 'scanner_patient_name'
    elif 'patient_name' in df.columns:
        df = df.rename(columns={'patient_name': 'scanner_patient_name'})
        rid = 'scanner_patient_name'
    elif 'patient_id' in df.columns:
        df = df.rename(columns={'patient_id': 'scanner_patient_name'})
        rid = 'scanner_patient_name'
    else:
        raise ValueError('Radiomics CSV lacks an ID column (scanner_patient_name/patient_name/patient_id)')

    if 'cr_popf' not in df.columns:
        m = pd.read_csv(matches_path)
        if 'scanner_patient_name' not in m.columns:
            raise ValueError('Matches CSV must contain scanner_patient_name')
        merged = df.merge(m[['scanner_patient_name', 'popf_grade']], on='scanner_patient_name', how='left')
        n_miss = int(merged['popf_grade'].isna().sum())
        if n_miss > 0 and args.allow_id_normalization:
            # Try normalized fallback merge
            from unicodedata import normalize, combining
            import re
            def canon(s):
                s = '' if s is None else str(s)
                s = ''.join(c for c in normalize('NFKD', s) if not combining(c)).lower()
                s = s.replace('-', '_').replace(' ', '_')
                s = re.sub(r'[^a-z0-9_]+', '_', s)
                s = re.sub(r'_+', '_', s).strip('_')
                return s
            merged['_canon'] = merged['scanner_patient_name'].map(canon)
            m['_canon'] = m['scanner_patient_name'].map(canon)
            m2 = m.drop_duplicates('_canon')[['_canon', 'popf_grade']]
            merged = merged.merge(m2.rename(columns={'popf_grade': 'popf_grade_canon'}), on='_canon', how='left')
            merged['popf_grade'] = merged['popf_grade'].fillna(merged['popf_grade_canon'])
            merged = merged.drop(columns=['_canon', 'popf_grade_canon'])
            n_miss = int(merged['popf_grade'].isna().sum())
        if n_miss > 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            merged.loc[merged['popf_grade'].isna(), ['scanner_patient_name']].to_csv(args.output_dir / 'radiomics_unmatched.csv', index=False)
            print(f"Dropping {n_miss} unmatched radiomics rows; saved to {args.output_dir/'radiomics_unmatched.csv'}")
            merged = merged.dropna(subset=['popf_grade']).reset_index(drop=True)
        pos = set([t.strip().upper() for t in args.positive_grades.split(',') if t.strip()])
        merged['cr_popf'] = merged['popf_grade'].apply(lambda x: 1 if str(x).upper() in pos else 0)
        df = merged

    # Build X, y, feature names
    exclude = {'scanner_patient_name', 'patient_name', 'patient_id', 'cr_popf', 'popf_grade'}
    feat_names = [c for c in df.columns if (c not in exclude and pd.api.types.is_numeric_dtype(df[c]))]
    if args.texture_only:
        fams = ('_glcm_', '_glrlm_', '_glszm_', '_gldm_', '_ngtdm_')
        feat_names = [c for c in feat_names if any(f in c.lower() for f in fams)]
    X_prefiltered = df[feat_names].values
    y = df['cr_popf'].astype(int).values

    # Print succinct data summary at start of run (skip when temporal uses separate train/test files)
    try:
        if not (args.temporal_holdout and (args.train_radiomics_path or args.test_radiomics_path)):
            n_samples = int(len(y))
            n_features = int(len(feat_names))
            n_pos = int((y == 1).sum())
            n_neg = int((y == 0).sum())
            prev = (n_pos / n_samples * 100.0) if n_samples > 0 else float('nan')
            print("\nDATA SUMMARY")
            print(f"Samples: {n_samples} | Features: {n_features}")
            print(f"Class counts: pos={n_pos}, neg={n_neg} (positives {prev:.1f}%)")
    except Exception:
        pass

    # Optional global prefilter (skip here if we plan train-only prefilter inside CV/temporal)
    if args.prefilter:
        if not (args.run_cv and args.cv_prefilter_train_only) and not (args.temporal_holdout and args.temporal_prefilter_train_only):
            X_prefiltered, feat_names, pf_stats = _prefilter_unsupervised(
                X_prefiltered, feat_names,
                missing_threshold=args.prefilter_missing_threshold,
                variance_threshold=args.prefilter_variance_threshold,
                correlation_threshold=args.prefilter_correlation_threshold,
            )
            print(f"Applied global prefilter | kept_features={len(feat_names)} | removed_missing={pf_stats['removed_missing']} | removed_variance={pf_stats['removed_variance']} | removed_corr={pf_stats['removed_correlation']}")

    # Scanner-type validation: train on dominant scanner type, test on others
    if args.scanner_type_validation:
        if not args.scanner_metadata_path:
            raise ValueError('--scanner-metadata-path is required for scanner-type validation')
        meta = pd.read_csv(args.scanner_metadata_path)
        # Normalize header whitespace
        meta.columns = [str(c).strip() for c in meta.columns]

        # Resolve ID and scanner-type columns with canonical fallback
        id_col_req = args.scanner_id_col
        type_col_req = args.scanner_type_col
        id_col = id_col_req if id_col_req in meta.columns else None
        type_col = type_col_req if type_col_req in meta.columns else None

        if id_col is None or type_col is None:
            import re
            def canon(s):
                s = '' if s is None else str(s)
                s = s.strip().lower().replace('-', '_').replace(' ', '_')
                s = re.sub(r'[^a-z0-9_]+', '_', s)
                s = re.sub(r'_+', '_', s).strip('_')
                return s
            meta_map = {canon(c): c for c in meta.columns}
            if id_col is None and canon(id_col_req) in meta_map:
                id_col = meta_map[canon(id_col_req)]
            # Try exact request first, then common synonyms if not found
            if type_col is None:
                if canon(type_col_req) in meta_map:
                    type_col = meta_map[canon(type_col_req)]
                else:
                    # Try common synonyms
                    for cand in (
                        'scanner_type', 'scannertype', 'scanner', 'scanner_name', 'scanner_model',
                        'manufacturer', 'vendor', 'model', 'model_name', 'manufacturer_model'
                    ):
                        if cand in meta_map:
                            type_col = meta_map[cand]
                            break
        if id_col is None or type_col is None:
            raise ValueError(
                f"Scanner metadata must include ID and type columns. Requested id='{args.scanner_id_col}', "
                f"type='{args.scanner_type_col}'. Available: {list(meta.columns)}"
            )

        # Prepare minimal metadata and merge on scanner_patient_name
        meta_copy = meta[[id_col, type_col]].copy()
        meta_copy = meta_copy.rename(columns={id_col: 'scanner_patient_name', type_col: 'scanner_type'})
        # Ensure strings
        meta_copy['scanner_patient_name'] = meta_copy['scanner_patient_name'].astype(str)
        meta_copy['scanner_type'] = meta_copy['scanner_type'].astype(str)

        merged = df.reset_index().merge(meta_copy, on='scanner_patient_name', how='left')
        if 'scanner_type' not in merged.columns:
            raise ValueError('Scanner type column not found after merge')

        # Drop rows with missing scanner type
        missing_types = int(merged['scanner_type'].isna().sum())
        if missing_types > 0:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            merged.loc[merged['scanner_type'].isna(), ['scanner_patient_name']].to_csv(
                args.output_dir / 'scanner_type_unmatched.csv', index=False
            )
            print(f"Dropping {missing_types} rows without scanner_type; saved unmatched IDs to scanner_type_unmatched.csv")
        valid = merged.dropna(subset=['scanner_type']).copy()

        # Determine dominant scanner type
        if args.dominant_scanner_value is not None:
            dominant = str(args.dominant_scanner_value)
            if dominant not in set(valid['scanner_type'].astype(str)):
                available = valid['scanner_type'].value_counts().to_dict()
                raise ValueError(f"dominant scanner '{dominant}' not found in metadata. Available counts: {available}")
        else:
            dominant = valid['scanner_type'].astype(str).value_counts().idxmax()

        # Save counts for transparency
        counts = valid['scanner_type'].astype(str).value_counts().rename_axis('scanner_type').reset_index(name='count')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        counts.to_csv(args.output_dir / 'scanner_type_counts.csv', index=False)

        # Indices for train/test using original df indices
        train_df_idx = valid.loc[valid['scanner_type'].astype(str) == dominant, 'index'].values
        test_df_idx = valid.loc[valid['scanner_type'].astype(str) != dominant, 'index'].values

        if len(train_df_idx) == 0 or len(test_df_idx) == 0:
            raise ValueError('Scanner-type split produced empty train or test set. Check metadata and column selection.')

        X_tr, X_te = X_prefiltered[train_df_idx], X_prefiltered[test_df_idx]
        y_tr, y_te = y[train_df_idx], y[test_df_idx]

        # Sanity check class balance
        if len(np.unique(y_tr)) < 2:
            raise ValueError('Training set has a single class after scanner-type split. Cannot train classifier.')

        # Train-only STABL selection on dominant scanner subset
        thresh = (args.thresholds[0] if args.thresholds else 0.75)
        ensemble = EnsembleSTABL(
            n_runs=args.ensemble_runs,
            threshold=thresh,
            n_bootstraps=500,
            regularization=args.stabl_regularization,
            alpha=args.stabl_l1_ratio,
            max_features=args.n_features,
            consensus_threshold=args.consensus_threshold,
            n_jobs=args.n_jobs,
            random_state=42,
            C=args.stabl_c,
            tol=args.stabl_tol,
            max_iter=args.stabl_max_iter,
            artificial_type=(None if args.stabl_artificial_type == 'none' else args.stabl_artificial_type),
            lambda_grid=args.lambda_grid,
            n_lambda=args.n_lambda,
            fdr_start=args.fdr_start,
            fdr_end=args.fdr_end,
            fdr_step=args.fdr_step,
            perc_corr_group_threshold=args.stabl_corr_group_threshold,
            sample_fraction=args.stabl_sample_fraction
        )
        panel = ensemble.fit(pd.DataFrame(X_tr, columns=feat_names), pd.Series(y_tr))
        sel_idx = [i for i, n in enumerate(feat_names) if n in panel]
        if not sel_idx:
            raise ValueError('No features selected on dominant scanner training set')

        # Build LR pipeline and optionally tune on training set only
        from sklearn.pipeline import Pipeline as SkPipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        from sklearn.metrics import roc_auc_score

        solver = 'liblinear'
        base_pipe = SkPipe([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', solver=solver, max_iter=5000, penalty=args.lr_penalty))
        ])

        best_c = 1.0
        best_cv_auc = None
        pipe = base_pipe
        if len(np.unique(y_tr)) == 2:
            if hasattr(args, 'lr_tune') and args.lr_tune:
                class_counts = np.bincount(y_tr.astype(int))
                min_count = int(class_counts.min()) if class_counts.size == 2 else 0
                cv_splits = max(2, min(5, min_count))
                pipe, best_c, best_cv_auc = _tune_lr_c_optuna(
                    X_tr[:, sel_idx],
                    y_tr,
                    base_pipe,
                    cv_splits=cv_splits,
                    trials=getattr(args, 'lr_optuna_trials', 30),
                    seed=getattr(args, 'lr_optuna_seed', 42),
                    c_min=getattr(args, 'lr_optuna_c_min', 1e-3),
                    c_max=getattr(args, 'lr_optuna_c_max', 1e2),
                    grid_values=getattr(args, 'lr_c_grid', None),
                )
            else:
                pipe.set_params(model__C=1.0)
        pipe.fit(X_tr[:, sel_idx], y_tr)
        # Training (resubstitution) AUC on dominant scanner subset
        train_auc = None
        try:
            prob_tr = pipe.predict_proba(X_tr[:, sel_idx])[:, 1]
            if len(np.unique(y_tr)) == 2:
                train_auc = float(roc_auc_score(y_tr, prob_tr))
        except Exception:
            pass

        # Evaluate on combined non-dominant scanners
        prob = pipe.predict_proba(X_te[:, sel_idx])[:, 1]
        auc_all = roc_auc_score(y_te, prob) if len(np.unique(y_te)) == 2 else None

        # Optional stratified bootstrap CI on combined test
        auc_all_ci = None
        n_boot_eff = 0
        if auc_all is not None and args.test_bootstrap and args.test_bootstrap > 0:
            rng = np.random.RandomState(42)
            pos_idx = np.where(y_te == 1)[0]
            neg_idx = np.where(y_te == 0)[0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            scores = []
            if n_pos > 0 and n_neg > 0:
                for _ in range(int(args.test_bootstrap)):
                    bs_pos = resample(pos_idx, replace=True, n_samples=n_pos, random_state=rng)
                    bs_neg = resample(neg_idx, replace=True, n_samples=n_neg, random_state=rng)
                    idx = np.concatenate([bs_pos, bs_neg])
                    y_b = y_te[idx]
                    if len(np.unique(y_b)) < 2:
                        continue
                    scores.append(roc_auc_score(y_b, prob[idx]))
                n_boot_eff = len(scores)
            if scores:
                auc_all_ci = [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]

        # Per-scanner type evaluation on others
        per_scanner = {}
        others = valid.loc[valid['scanner_type'].astype(str) != dominant, ['scanner_type', 'index']]
        for stype, grp in others.groupby('scanner_type'):
            idx = grp['index'].values
            y_s = y[idx]
            if len(idx) == 0 or len(np.unique(y_s)) < 2:
                per_scanner[str(stype)] = {
                    'n_samples': int(len(idx)),
                    'auc': None,
                    'auc_ci': None,
                    'pos': int((y_s == 1).sum()),
                    'neg': int((y_s == 0).sum())
                }
                continue
            prob_s = pipe.predict_proba(X_prefiltered[idx][:, sel_idx])[:, 1]
            auc_s = roc_auc_score(y_s, prob_s)
            ci_s = None
            if args.test_bootstrap and args.test_bootstrap > 0:
                rng = np.random.RandomState(123)
                pos_idx = np.where(y_s == 1)[0]
                neg_idx = np.where(y_s == 0)[0]
                n_pos, n_neg = len(pos_idx), len(neg_idx)
                scores = []
                if n_pos > 0 and n_neg > 0:
                    for _ in range(int(args.test_bootstrap)):
                        bs_pos = resample(pos_idx, replace=True, n_samples=n_pos, random_state=rng)
                        bs_neg = resample(neg_idx, replace=True, n_samples=n_neg, random_state=rng)
                        bidx = np.concatenate([bs_pos, bs_neg])
                        y_b = y_s[bidx]
                        if len(np.unique(y_b)) < 2:
                            continue
                        scores.append(roc_auc_score(y_b, prob_s[bidx]))
                if scores:
                    ci_s = [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]
            per_scanner[str(stype)] = {
                'n_samples': int(len(idx)),
                'auc': float(auc_s),
                'auc_ci': ci_s,
                'pos': int((y_s == 1).sum()),
                'neg': int((y_s == 0).sum())
            }

        # Save summary
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = {
            'mode': 'scanner-type-validation',
            'dominant_scanner': str(dominant),
            'train_size': int(len(train_df_idx)),
            'test_size': int(len(test_df_idx)),
            'train_pos': int((y_tr == 1).sum()),
            'train_neg': int((y_tr == 0).sum()),
            'test_pos': int((y_te == 1).sum()),
            'test_neg': int((y_te == 0).sum()),
            'selected_features': panel,
            'n_selected_features': int(len(panel)),
            'lr_penalty': args.lr_penalty,
            'best_c': float(best_c) if best_c is not None else None,
            'best_cv_auc_on_train': best_cv_auc,
            'train_auc_resub': train_auc,
            'combined_others_auc': float(auc_all) if auc_all is not None else None,
            'combined_others_auc_ci': auc_all_ci,
            'test_bootstrap_effective': int(n_boot_eff),
            'stabl_config': {
                'regularization': args.stabl_regularization,
                'l1_ratio': args.stabl_l1_ratio,
                'C': args.stabl_c,
                'consensus_threshold': args.consensus_threshold,
                'threshold': float(thresh),
                'artificial_type': args.stabl_artificial_type,
                'lambda_grid': args.lambda_grid,
                'n_lambda': args.n_lambda,
                'fdr_range': [args.fdr_start, args.fdr_end, args.fdr_step]
            },
            'per_scanner': per_scanner,
        }
        with open(args.output_dir / 'scanner_eval.json', 'w') as f:
            json.dump(out, f, indent=2)

        # Pretty print extended summary
        print("\nSCANNER-TYPE VALIDATION SUMMARY")
        print("-"*70)
        print(f"Dominant scanner (train): {dominant}")
        print(f"Train: n={len(train_df_idx)} | pos={out['train_pos']} | neg={out['train_neg']}")
        print(f"Test (all other scanners): n={len(test_df_idx)} | pos={out['test_pos']} | neg={out['test_neg']}")
        print(f"Panel size: {len(panel)} | Features: {', '.join(panel[:10])}{' ...' if len(panel)>10 else ''}")
        if train_auc is not None:
            print(f"Training AUC (dominant scanner, resub): {train_auc:.3f}" + (f" | tuned C={best_c}" if best_cv_auc is not None else ""))
        if auc_all is not None:
            ci_str = f" [{auc_all_ci[0]:.3f}, {auc_all_ci[1]:.3f}]" if auc_all_ci else ""
            bs_str = f" | test bootstraps used: {n_boot_eff}" if auc_all_ci else ""
            print(f"Combined others AUC: {auc_all:.3f}{ci_str}{bs_str}")
        else:
            print("Combined others AUC: N/A (single-class test set)")
        print("STABL:", f"runs={args.ensemble_runs}", f"thr={thresh}", f"consensus={args.consensus_threshold}",
              f"reg={args.stabl_regularization}", f"C={args.stabl_c}", f"l1_ratio={args.stabl_l1_ratio}")
        return

    # Discovery-only mode: select and export panel
    if args.discovery_only and not args.temporal_holdout:
        thresh = (args.thresholds[0] if args.thresholds else 0.75)
        ensemble = EnsembleSTABL(
            n_runs=args.ensemble_runs,
            threshold=thresh,
            n_bootstraps=500,
            regularization=args.stabl_regularization,
            alpha=args.stabl_l1_ratio,
            max_features=args.n_features,
            consensus_threshold=args.consensus_threshold,
            n_jobs=args.n_jobs,
            random_state=42,
            C=args.stabl_c,
            tol=args.stabl_tol,
            max_iter=args.stabl_max_iter,
            artificial_type=(None if args.stabl_artificial_type == 'none' else args.stabl_artificial_type),
            lambda_grid=args.lambda_grid,
            n_lambda=args.n_lambda,
            fdr_start=args.fdr_start,
            fdr_end=args.fdr_end,
            fdr_step=args.fdr_step,
            perc_corr_group_threshold=args.stabl_corr_group_threshold,
            sample_fraction=args.stabl_sample_fraction
        )
        panel = ensemble.fit(pd.DataFrame(X_prefiltered, columns=feat_names), pd.Series(y))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        panel_path = Path(args.export_panel) if args.export_panel else (args.output_dir / 'selected_features.txt')
        with open(panel_path, 'w') as f:
            for feat in panel:
                f.write(f"{feat}\n")
        pd.Series(ensemble.feature_frequencies).sort_values(ascending=False).to_csv(
            args.output_dir / 'feature_selection_frequencies.csv', header=['count']
        )
        with open(args.output_dir / 'discovery_summary.json', 'w') as f:
            json.dump({
                'mode': 'discovery-only',
                'threshold': thresh,
                'ensemble_runs': args.ensemble_runs,
                'n_bootstraps': 500,
                'consensus_threshold': args.consensus_threshold,
                'n_features_cap': args.n_features,
                'stabl_regularization': args.stabl_regularization,
                'stabl_l1_ratio': args.stabl_l1_ratio,
                'stabl_c': args.stabl_c,
                'stabl_artificial_type': args.stabl_artificial_type,
                'lambda_grid': args.lambda_grid,
                'n_lambda': args.n_lambda,
                'fdr_start': args.fdr_start,
                'fdr_end': args.fdr_end,
                'fdr_step': args.fdr_step,
                'radiomics_path': str(radiomics_path),
                'matches_path': str(matches_path),
                'positive_grades': args.positive_grades,
                'id_matching': 'exact' + ('+normalized' if args.allow_id_normalization else ''),
                'panel_file': str(panel_path)
            }, f, indent=2)
        print(f"Discovery panel saved to {panel_path}")
        return

    # Repeated CV with train-only selection per fold
    if args.run_cv:
        from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, GridSearchCV, train_test_split
        from sklearn.pipeline import Pipeline as SkPipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        # Prepare groups if grouped CV requested
        groups = None
        if args.cv_grouped:
            if not args.scanner_metadata_path:
                raise ValueError('--cv-grouped requires --scanner-metadata-path and a type column via --scanner-type-col')
            meta = pd.read_csv(args.scanner_metadata_path)
            meta.columns = [str(c).strip() for c in meta.columns]
            id_col_req = args.scanner_id_col
            type_col_req = args.scanner_type_col
            id_col = id_col_req if id_col_req in meta.columns else None
            type_col = type_col_req if type_col_req in meta.columns else None
            if id_col is None or type_col is None:
                import re
                def canon(s):
                    s = '' if s is None else str(s)
                    s = s.strip().lower().replace('-', '_').replace(' ', '_')
                    s = re.sub(r'[^a-z0-9_]+', '_', s)
                    s = re.sub(r'_+', '_', s).strip('_')
                    return s
                meta_map = {canon(c): c for c in meta.columns}
                if id_col is None and canon(id_col_req) in meta_map:
                    id_col = meta_map[canon(id_col_req)]
                if type_col is None and canon(type_col_req) in meta_map:
                    type_col = meta_map[canon(type_col_req)]
            if id_col is None or type_col is None:
                raise ValueError(f'Cannot find id/type columns in scanner metadata. Available: {list(meta.columns)}')
            meta_copy = meta[[id_col, type_col]].copy().rename(columns={id_col: 'scanner_patient_name', type_col: 'scanner_type'})
            meta_copy['scanner_patient_name'] = meta_copy['scanner_patient_name'].astype(str)
            meta_copy['scanner_type'] = meta_copy['scanner_type'].astype(str)
            merged_grp = df[['scanner_patient_name']].merge(meta_copy, on='scanner_patient_name', how='left')
            if merged_grp['scanner_type'].isna().any():
                merged_grp['scanner_type'] = merged_grp['scanner_type'].fillna('UNKNOWN')
            groups = merged_grp['scanner_type'].astype(str).values

        if args.cv_model == 'xgb':
            print(f"Running repeated stratified CV: {args.cv_splits}x{args.cv_repeats} with train-only univariate selection + XGBoost ...")
            fold_scores = []
            fold_panels = []
            # OOF accumulators
            n_samples = X_prefiltered.shape[0]
            oof_sum = np.zeros(n_samples, dtype=float)
            oof_count = np.zeros(n_samples, dtype=int)

            def safe_univariate_auc(y_true, x_col):
                mask = ~np.isnan(x_col)
                y_m = y_true[mask]
                x_m = x_col[mask]
                if len(y_m) < 5 or len(np.unique(y_m)) < 2:
                    return 0.5
                try:
                    auc = roc_auc_score(y_m, x_m)
                    return float(auc if auc >= 0.5 else 1.0 - auc)
                except Exception:
                    return 0.5

            def abs_pointbiserial(y_true, x_col):
                mask = ~np.isnan(x_col)
                y_m = y_true[mask]
                x_m = x_col[mask]
                if len(y_m) < 5 or len(np.unique(y_m)) < 2:
                    return 0.0
                try:
                    y_m = y_m.astype(float)
                    x_m = x_m.astype(float)
                    # Guard constant vectors
                    if np.nanstd(x_m) < 1e-12:
                        return 0.0
                    r = np.corrcoef(x_m, y_m)[0, 1]
                    if np.isnan(r):
                        return 0.0
                    return float(abs(r))
                except Exception:
                    return 0.0

            # Build split iterator (grouped or not)
            split_iter = None
            if args.cv_grouped and groups is not None:
                try:
                    from sklearn.model_selection import StratifiedGroupKFold
                    # Emulate repeats by changing random_state each loop
                    iters = []
                    counter = 0
                    for rep in range(args.cv_repeats):
                        sgkf = StratifiedGroupKFold(n_splits=args.cv_splits, shuffle=True, random_state=42 + rep)
                        for tr, te in sgkf.split(X_prefiltered, y, groups):
                            iters.append((tr, te))
                            counter += 1
                    split_iter = enumerate(iters, 1)
                except Exception:
                    print('Warning: StratifiedGroupKFold unavailable; falling back to standard RepeatedStratifiedKFold')
            if split_iter is None:
                rskf = RepeatedStratifiedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=42)
                split_iter = enumerate(rskf.split(X_prefiltered, y), 1)

            for fold_id, (tr, te) in split_iter:
                X_tr, X_te = X_prefiltered[tr], X_prefiltered[te]
                y_tr, y_te = y[tr], y[te]

                # Optional train-only prefilter on LR CV branch
                local_feat_names = feat_names
                if args.prefilter and args.cv_prefilter_train_only:
                    X_tr_pf, fn_pf, pf_stats = _prefilter_unsupervised(
                        X_tr, local_feat_names,
                        missing_threshold=args.prefilter_missing_threshold,
                        variance_threshold=args.prefilter_variance_threshold,
                        correlation_threshold=args.prefilter_correlation_threshold,
                    )
                    # Map test to kept columns
                    keep_idx = [i for i, n in enumerate(local_feat_names) if n in set(fn_pf)]
                    if not keep_idx:
                        continue
                    X_tr, X_te = X_tr_pf, X_te[:, keep_idx]
                    local_feat_names = fn_pf

                # Determine Top-K based on events in training
                pos_tr = int((y_tr == 1).sum())
                top_k = int(min(args.max_topk, max(args.panel_size, args.topk_multiplier * max(1, pos_tr))))
                top_k = min(top_k, X_tr.shape[1])

                # Filter near-zero-variance and low-unique features on training only
                n_train = X_tr.shape[0]
                nzv_mask = []
                uniq_mask = []
                for j in range(X_tr.shape[1]):
                    col = X_tr[:, j]
                    col = col[~np.isnan(col)]
                    var_ok = np.nanvar(col) > args.nzv_var_threshold
                    uniq_ok = (len(np.unique(col)) / max(1, len(col))) >= args.min_unique_ratio
                    nzv_mask.append(var_ok)
                    uniq_mask.append(uniq_ok)
                valid_mask = np.array(nzv_mask) & np.array(uniq_mask)
                valid_idx_all = np.where(valid_mask)[0]
                if valid_idx_all.size == 0:
                    # If everything was filtered, fall back to all features
                    valid_idx_all = np.arange(X_tr.shape[1])

                # Compute directionless AUC and correlation tie-breaker on candidate features
                aucs = np.zeros(X_tr.shape[1], dtype=float)
                cors = np.zeros(X_tr.shape[1], dtype=float)
                for j in valid_idx_all:
                    col = X_tr[:, j]
                    aucs[j] = safe_univariate_auc(y_tr, col)
                    cors[j] = abs_pointbiserial(y_tr, col)
                # Rank by AUC, then correlation, then variance proxy (unique count)
                # Build a structured array for argsort with descending keys
                uniq_counts = np.zeros(X_tr.shape[1], dtype=float)
                for j in valid_idx_all:
                    col = X_tr[:, j]
                    col = col[~np.isnan(col)]
                    uniq_counts[j] = len(np.unique(col))

                # Apply minimum train AUC threshold if possible
                cand_idx = [j for j in valid_idx_all if aucs[j] >= args.min_train_auc]
                if len(cand_idx) < args.panel_size:
                    # relax threshold to ensure enough features
                    cand_idx = valid_idx_all.tolist()

                sort_keys = np.lexsort((
                    -uniq_counts[cand_idx],  # more unique values preferred
                    -cors[cand_idx],         # stronger correlation preferred
                    -aucs[cand_idx],         # higher AUC preferred (primary)
                ))
                ordered = np.array(cand_idx)[sort_keys]
                top_idx = ordered[:top_k]

                # Shrink to final panel size using L1-logistic on training fold (optional)
                if args.use_l1_shrink and len(top_idx) > 0 and len(np.unique(y_tr)) == 2:
                    cand_idx = top_idx
                    # Impute and scale on training only
                    imp_shrink = SimpleImputer(strategy='median')
                    X_tr_cand = imp_shrink.fit_transform(X_tr[:, cand_idx])
                    scaler = StandardScaler()
                    X_tr_cand = scaler.fit_transform(X_tr_cand)
                    try:
                        lr = LogisticRegression(penalty='l1', solver='liblinear',
                                                class_weight='balanced', max_iter=5000)
                        lr.fit(X_tr_cand, y_tr)
                        coefs = np.abs(lr.coef_.ravel())
                        nonzero = np.where(coefs > 1e-12)[0]
                        if nonzero.size == 0:
                            # fallback to top by AUC order
                            panel_idx = cand_idx[:max(1, args.panel_size)]
                        else:
                            # rank non-zero by coefficient magnitude
                            nz_order = nonzero[np.argsort(coefs[nonzero])[::-1]]
                            keep = nz_order[:max(1, args.panel_size)]
                            panel_idx = [cand_idx[i] for i in keep]
                            # if fewer than panel_size, pad with next best by AUC
                            if len(panel_idx) < args.panel_size:
                                for j in cand_idx:
                                    if j not in panel_idx:
                                        panel_idx.append(j)
                                        if len(panel_idx) >= args.panel_size:
                                            break
                    except Exception:
                        panel_idx = cand_idx[:max(1, args.panel_size)]
                else:
                    # Simple truncation to panel size
                    panel_idx = top_idx[:max(1, args.panel_size)]
                panel = [local_feat_names[i] for i in panel_idx]

                # Impute missing values using training only
                imp = SimpleImputer(strategy='median')
                X_tr_p = imp.fit_transform(X_tr[:, panel_idx])
                X_te_p = imp.transform(X_te[:, panel_idx])

                # Build train/dev split for early stopping
                use_es = args.xgb_early_stopping and args.xgb_early_stopping > 0
                if use_es and len(np.unique(y_tr)) == 2 and y_tr.shape[0] >= 10:
                    try:
                        X_tr_i, X_dev_i, y_tr_i, y_dev_i = train_test_split(
                            X_tr_p, y_tr, test_size=0.2, stratify=y_tr, random_state=42
                        )
                    except ValueError:
                        X_tr_i, X_dev_i, y_tr_i, y_dev_i = X_tr_p, None, y_tr, None
                        use_es = False
                else:
                    X_tr_i, X_dev_i, y_tr_i, y_dev_i = X_tr_p, None, y_tr, None
                    use_es = False

                # Class weight
                neg_tr = int((y_tr == 0).sum())
                spw = (neg_tr / max(1, pos_tr)) if pos_tr > 0 else 1.0

                clf = xgb.XGBClassifier(
                    max_depth=args.xgb_depth,
                    n_estimators=args.xgb_n_estimators,
                    learning_rate=args.xgb_lr,
                    subsample=args.xgb_subsample,
                    colsample_bytree=args.xgb_colsample,
                    reg_alpha=args.xgb_reg_alpha,
                    reg_lambda=args.xgb_reg_lambda,
                    min_child_weight=args.xgb_min_child_weight,
                    objective='binary:logistic',
                    eval_metric='auc',
                    tree_method='hist',
                    random_state=42 + fold_id,
                    n_jobs=args.n_jobs,
                    scale_pos_weight=spw,
                )

                if use_es and X_dev_i is not None:
                    # Prefer early_stopping_rounds when available; otherwise fallback to callbacks API
                    try:
                        clf.fit(
                            X_tr_i, y_tr_i,
                            eval_set=[(X_dev_i, y_dev_i)],
                            early_stopping_rounds=args.xgb_early_stopping,
                        )
                    except TypeError:
                        # Older/newer versions might not accept early_stopping_rounds; use callbacks
                        try:
                            cb = [xgb.callback.EarlyStopping(rounds=args.xgb_early_stopping, save_best=True)]
                            clf.fit(
                                X_tr_i, y_tr_i,
                                eval_set=[(X_dev_i, y_dev_i)],
                                callbacks=cb,
                            )
                        except Exception:
                            # Fallback: train without early stopping
                            clf.fit(X_tr_p, y_tr)
                else:
                    clf.fit(X_tr_p, y_tr)

                prob = clf.predict_proba(X_te_p)[:, 1]
                if len(np.unique(y_te)) == 2:
                    auc = roc_auc_score(y_te, prob)
                    fold_scores.append(float(auc))
                    # OOF accumulate
                    for idx_local, idx_global in enumerate(te):
                        oof_sum[idx_global] += prob[idx_local]
                        oof_count[idx_global] += 1
                    if args.save_fold_selections:
                        fold_panels.append({'fold': fold_id, 'features': panel,
                                            'best_ntree_limit': getattr(clf, 'best_ntree_limit', None)})

            # Summarize
            if fold_scores:
                ci = [float(np.percentile(fold_scores, 2.5)), float(np.percentile(fold_scores, 97.5))]
                # OOF AUC + CI (bootstrap on patients)
                mask = oof_count > 0
                oof_auc = None
                oof_ci = None
                if np.any(mask):
                    y_oof = y[mask]
                    p_oof = (oof_sum[mask] / np.maximum(oof_count[mask], 1)).astype(float)
                    if len(np.unique(y_oof)) == 2:
                        oof_auc = float(roc_auc_score(y_oof, p_oof))
                        # stratified bootstrap
                        rng = np.random.RandomState(42)
                        pos_idx = np.where(y_oof == 1)[0]
                        neg_idx = np.where(y_oof == 0)[0]
                        scores_bs = []
                        if len(pos_idx) > 0 and len(neg_idx) > 0:
                            for _ in range(2000):
                                bs_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=rng)
                                bs_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=rng)
                                idx = np.concatenate([bs_pos, bs_neg])
                                y_b = y_oof[idx]
                                if len(np.unique(y_b)) < 2:
                                    continue
                                scores_bs.append(roc_auc_score(y_b, p_oof[idx]))
                            if scores_bs:
                                oof_ci = [float(np.percentile(scores_bs, 2.5)), float(np.percentile(scores_bs, 97.5))]
                summary = {
                    'cv_model': 'xgb',
                    'cv_splits': args.cv_splits,
                    'cv_repeats': args.cv_repeats,
                    'cv_grouped': bool(args.cv_grouped),
                    'n_folds': len(fold_scores),
                    'auc_mean': float(np.mean(fold_scores)),
                    'auc_median': float(np.median(fold_scores)),
                    'auc_std': float(np.std(fold_scores)),
                    'auc_ci': ci,
                    'scores': fold_scores,
                    'oof_auc': oof_auc,
                    'oof_auc_ci': oof_ci
                }
            else:
                summary = {'error': 'No valid folds (possibly single-class splits).'}

            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output_dir / 'cv_eval.json', 'w') as f:
                if args.save_fold_selections:
                    summary['fold_selections'] = fold_panels
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
            return
        else:
            print(f"Running repeated stratified CV: {args.cv_splits}x{args.cv_repeats} with train-only STABL selection ...")
            fold_scores = []
            fold_panels = []
            thresh = (args.thresholds[0] if args.thresholds else 0.75)
            # OOF accumulators
            n_samples = X_prefiltered.shape[0]
            oof_sum = np.zeros(n_samples, dtype=float)
            oof_count = np.zeros(n_samples, dtype=int)
            # Build split iterator (grouped or not)
            split_iter = None
            if args.cv_grouped and groups is not None:
                try:
                    from sklearn.model_selection import StratifiedGroupKFold
                    iters = []
                    for rep in range(args.cv_repeats):
                        sgkf = StratifiedGroupKFold(n_splits=args.cv_splits, shuffle=True, random_state=42 + rep)
                        for tr, te in sgkf.split(X_prefiltered, y, groups):
                            iters.append((tr, te))
                    split_iter = enumerate(iters, 1)
                except Exception:
                    print('Warning: StratifiedGroupKFold unavailable; falling back to standard RepeatedStratifiedKFold')
            if split_iter is None:
                rskf = RepeatedStratifiedKFold(n_splits=args.cv_splits, n_repeats=args.cv_repeats, random_state=42)
                split_iter = enumerate(rskf.split(X_prefiltered, y), 1)

            for fold_id, (tr, te) in split_iter:
                X_tr, X_te = X_prefiltered[tr], X_prefiltered[te]
                y_tr, y_te = y[tr], y[te]

                # Optional train-only prefilter on LR CV branch
                local_feat_names = feat_names
                if args.prefilter and args.cv_prefilter_train_only:
                    X_tr_pf, fn_pf, pf_stats = _prefilter_unsupervised(
                        X_tr, local_feat_names,
                        missing_threshold=args.prefilter_missing_threshold,
                        variance_threshold=args.prefilter_variance_threshold,
                        correlation_threshold=args.prefilter_correlation_threshold,
                    )
                    keep_idx = [i for i, n in enumerate(local_feat_names) if n in set(fn_pf)]
                    if not keep_idx:
                        continue
                    X_tr, X_te = X_tr_pf, X_te[:, keep_idx]
                    local_feat_names = fn_pf

                # Train-only STABL selection
                ensemble = EnsembleSTABL(
                    n_runs=args.ensemble_runs,
                    threshold=thresh,
                    n_bootstraps=500,
                    regularization=args.stabl_regularization,
                    alpha=args.stabl_l1_ratio,
                    max_features=args.n_features if args.n_features else 10,
                    consensus_threshold=args.consensus_threshold,
                    n_jobs=args.n_jobs,
                    random_state=42 + fold_id,
                    C=args.stabl_c,
                    tol=args.stabl_tol,
                    max_iter=args.stabl_max_iter,
                    artificial_type=(None if args.stabl_artificial_type == 'none' else args.stabl_artificial_type),
                    lambda_grid=args.lambda_grid,
                    n_lambda=args.n_lambda,
                    fdr_start=args.fdr_start,
                    fdr_end=args.fdr_end,
                    fdr_step=args.fdr_step,
                    perc_corr_group_threshold=args.stabl_corr_group_threshold,
                    sample_fraction=args.stabl_sample_fraction
                )
                panel = ensemble.fit(pd.DataFrame(X_tr, columns=local_feat_names), pd.Series(y_tr))
                sel_idx = [i for i, n in enumerate(local_feat_names) if n in panel]
                if not sel_idx:
                    continue

                # Train-only LR with optional tuning
                solver = 'liblinear'
                base_pipe = SkPipe([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', StandardScaler()),
                    ('model', LogisticRegression(class_weight='balanced', solver=solver, max_iter=5000, penalty='l2'))
                ])
                pipe = base_pipe
                best_c = 1.0
                if len(np.unique(y_tr)) == 2:
                    class_counts = np.bincount(y_tr.astype(int))
                    min_count = int(class_counts.min()) if class_counts.size == 2 else 0
                    cv_splits = max(2, min(5, min_count))
                    pipe, best_c, best_cv_auc = _tune_lr_c_optuna(
                        X_tr[:, sel_idx],
                        y_tr,
                        base_pipe,
                        cv_splits=cv_splits,
                        trials=getattr(args, 'lr_optuna_trials', 30),
                        seed=getattr(args, 'lr_optuna_seed', 42),
                        c_min=getattr(args, 'lr_optuna_c_min', 1e-3),
                        c_max=getattr(args, 'lr_optuna_c_max', 1e2),
                        grid_values=getattr(args, 'lr_c_grid', None),
                    )

                pipe.fit(X_tr[:, sel_idx], y_tr)
                prob = pipe.predict_proba(X_te[:, sel_idx])[:, 1]
                if len(np.unique(y_te)) == 2:
                    auc = roc_auc_score(y_te, prob)
                    fold_scores.append(float(auc))
                    # OOF accumulate
                    for idx_local, idx_global in enumerate(te):
                        oof_sum[idx_global] += prob[idx_local]
                        oof_count[idx_global] += 1
                    if args.save_fold_selections:
                        fold_panels.append({'fold': fold_id, 'features': panel, 'best_c': float(best_c)})

            # Summarize
            if fold_scores:
                ci = [float(np.percentile(fold_scores, 2.5)), float(np.percentile(fold_scores, 97.5))]
                # OOF AUC + CI via patient bootstrap
                mask = oof_count > 0
                oof_auc = None
                oof_ci = None
                if np.any(mask):
                    y_oof = y[mask]
                    p_oof = (oof_sum[mask] / np.maximum(oof_count[mask], 1)).astype(float)
                    if len(np.unique(y_oof)) == 2:
                        oof_auc = float(roc_auc_score(y_oof, p_oof))
                        rng = np.random.RandomState(42)
                        pos_idx = np.where(y_oof == 1)[0]
                        neg_idx = np.where(y_oof == 0)[0]
                        scores_bs = []
                        if len(pos_idx) > 0 and len(neg_idx) > 0:
                            for _ in range(2000):
                                bs_pos = resample(pos_idx, replace=True, n_samples=len(pos_idx), random_state=rng)
                                bs_neg = resample(neg_idx, replace=True, n_samples=len(neg_idx), random_state=rng)
                                idx = np.concatenate([bs_pos, bs_neg])
                                y_b = y_oof[idx]
                                if len(np.unique(y_b)) < 2:
                                    continue
                                scores_bs.append(roc_auc_score(y_b, p_oof[idx]))
                            if scores_bs:
                                oof_ci = [float(np.percentile(scores_bs, 2.5)), float(np.percentile(scores_bs, 97.5))]
                summary = {
                    'cv_model': 'lr',
                    'cv_splits': args.cv_splits,
                    'cv_repeats': args.cv_repeats,
                    'cv_grouped': bool(args.cv_grouped),
                    'n_folds': len(fold_scores),
                    'auc_mean': float(np.mean(fold_scores)),
                    'auc_median': float(np.median(fold_scores)),
                    'auc_std': float(np.std(fold_scores)),
                    'auc_ci': ci,
                    'scores': fold_scores,
                    'oof_auc': oof_auc,
                    'oof_auc_ci': oof_ci
                }
            else:
                summary = {'error': 'No valid folds (possibly single-class splits).'}

            args.output_dir.mkdir(parents=True, exist_ok=True)
            with open(args.output_dir / 'cv_eval.json', 'w') as f:
                if args.save_fold_selections:
                    summary['fold_selections'] = fold_panels
                json.dump(summary, f, indent=2)
            print(json.dumps(summary, indent=2))
            return

    # Temporal holdout: train on oldest, test on newest
    if args.temporal_holdout:
        if not args.scanner_metadata_path:
            raise ValueError('--scanner-metadata-path is required for temporal holdout')
        meta = pd.read_csv(args.scanner_metadata_path)
        # Normalize header whitespace
        meta.columns = [str(c).strip() for c in meta.columns]
        id_col_req = args.scanner_id_col
        date_col_req = args.date_col
        id_col = id_col_req if id_col_req in meta.columns else None
        date_col = date_col_req if date_col_req in meta.columns else None
        if id_col is None or date_col is None:
            # Canonical fallback on column names (not values)
            import re
            def canon(s):
                s = '' if s is None else str(s)
                s = s.strip().lower().replace('-', '_').replace(' ', '_')
                s = re.sub(r'[^a-z0-9_]+', '_', s)
                s = re.sub(r'_+', '_', s).strip('_')
                return s
            meta_map = {canon(c): c for c in meta.columns}
            if id_col is None and canon(id_col_req) in meta_map:
                id_col = meta_map[canon(id_col_req)]
            if date_col is None and canon(date_col_req) in meta_map:
                date_col = meta_map[canon(date_col_req)]
        if id_col is None or date_col is None:
            raise ValueError(f"Scanner metadata must include specified id and date columns. Found columns: {list(meta.columns)}")
        meta_copy = meta[[id_col, date_col]].copy()
        meta_copy = meta_copy.rename(columns={id_col: 'scanner_patient_name'})

        raw_dates = meta_copy[date_col]
        parsed_dates = pd.to_datetime(raw_dates, errors='coerce')
        needs_numeric_parse = parsed_dates.isna() & raw_dates.notna()
        if needs_numeric_parse.any():
            as_str = raw_dates.astype(str).str.strip()
            mask_numeric = needs_numeric_parse & as_str.str.fullmatch(r'\d{8}')
            if mask_numeric.any():
                parsed_numeric = pd.to_datetime(as_str[mask_numeric], format='%Y%m%d', errors='coerce')
                parsed_dates.loc[mask_numeric] = parsed_numeric
        meta_copy[date_col] = parsed_dates
        # Build ordered ID list by StudyDate (global)
        valid_meta = meta_copy.dropna(subset=[date_col]).rename(columns={date_col: 'StudyDate'}).copy()
        valid_meta = valid_meta.sort_values('StudyDate').reset_index(drop=True)
        if valid_meta.empty:
            raise ValueError('No valid StudyDate to perform temporal split')
        n_all = len(valid_meta)
        n_train = int(np.floor((1.0 - args.holdout_fraction) * n_all))
        candidate_train_ids = valid_meta.iloc[:n_train]['scanner_patient_name'].astype(str).tolist()
        candidate_test_ids = valid_meta.iloc[n_train:]['scanner_patient_name'].astype(str).tolist()

        # Helper to load radiomics df with labels from path
        def _load_radiomics_with_labels(path_str: Optional[str]):
            if path_str is None:
                # Use already loaded df / feat_names / X_prefiltered / y
                return df.copy(), feat_names.copy(), X_prefiltered.copy(), y.copy()
            rdf = pd.read_csv(Path(path_str))
            # Normalize ID column
            if 'scanner_patient_name' in rdf.columns:
                pass
            elif 'patient_name' in rdf.columns:
                rdf = rdf.rename(columns={'patient_name': 'scanner_patient_name'})
            elif 'patient_id' in rdf.columns:
                rdf = rdf.rename(columns={'patient_id': 'scanner_patient_name'})
            else:
                raise ValueError(f'Radiomics CSV lacks an ID column: {path_str}')
            # Add labels via matches
            if 'cr_popf' not in rdf.columns:
                m = pd.read_csv(matches_path)
                if 'scanner_patient_name' not in m.columns:
                    raise ValueError('Matches CSV must contain scanner_patient_name')
                merged_l = rdf.merge(m[['scanner_patient_name', 'popf_grade']], on='scanner_patient_name', how='left')
                n_miss = int(merged_l['popf_grade'].isna().sum())
                if n_miss > 0 and args.allow_id_normalization:
                    from unicodedata import normalize, combining
                    import re
                    def canon(s):
                        s = '' if s is None else str(s)
                        s = ''.join(c for c in normalize('NFKD', s) if not combining(c)).lower()
                        s = s.replace('-', '_').replace(' ', '_')
                        s = re.sub(r'[^a-z0-9_]+', '_', s)
                        s = re.sub(r'_+', '_', s).strip('_')
                        return s
                    merged_l['_canon'] = merged_l['scanner_patient_name'].map(canon)
                    m['_canon'] = m['scanner_patient_name'].map(canon)
                    m2 = m.drop_duplicates('_canon')[['_canon', 'popf_grade']]
                    merged_l = merged_l.merge(m2.rename(columns={'popf_grade': 'popf_grade_canon'}), on='_canon', how='left')
                    merged_l['popf_grade'] = merged_l['popf_grade'].fillna(merged_l['popf_grade_canon'])
                    merged_l = merged_l.drop(columns=['_canon', 'popf_grade_canon'])
                if n_miss > 0:
                    # drop unmatched
                    merged_l = merged_l.dropna(subset=['popf_grade']).reset_index(drop=True)
                pos = set([t.strip().upper() for t in args.positive_grades.split(',') if t.strip()])
                merged_l['cr_popf'] = merged_l['popf_grade'].apply(lambda x: 1 if str(x).upper() in pos else 0)
                rdf = merged_l
            # Build features
            exclude = {'scanner_patient_name', 'patient_name', 'patient_id', 'cr_popf', 'popf_grade'}
            fns = [c for c in rdf.columns if (c not in exclude and pd.api.types.is_numeric_dtype(rdf[c]))]
            if args.texture_only:
                fams = ('_glcm_', '_glrlm_', '_glszm_', '_gldm_', '_ngtdm_')
                fns = [c for c in fns if any(f in c.lower() for f in fams)]
            X_ = rdf[fns].values
            y_ = rdf['cr_popf'].astype(int).values
            return rdf, fns, X_, y_

        # Load train/test datasets (train may default to df)
        df_train, feat_names_train, X_train_all, y_train_all = _load_radiomics_with_labels(args.train_radiomics_path)
        df_test, feat_names_test, X_test_all, y_test_all = _load_radiomics_with_labels(args.test_radiomics_path)

        # Restrict to intersection of features if using separate datasets
        if args.train_radiomics_path or args.test_radiomics_path:
            common_feats = [f for f in feat_names_train if f in set(feat_names_test)]
            if not common_feats:
                raise ValueError('No common features between training and test radiomics datasets')
            # Rebuild X arrays in the same feature order
            X_train_all = df_train[common_feats].values
            X_test_all = df_test[common_feats].values
            feat_names = common_feats  # override for modeling below
        else:
            # Use already built arrays/names
            feat_names = list(feat_names)

        # Build train/test id lists constrained by dataset availability
        train_ids = [i for i in candidate_train_ids if i in set(df_train['scanner_patient_name'].astype(str))]
        test_ids = [i for i in candidate_test_ids if i in set(df_test['scanner_patient_name'].astype(str))]
        if len(train_ids) == 0 or len(test_ids) == 0:
            raise ValueError('Temporal split mapping produced empty train or test IDs after dataset constraint. Check ID alignment and files.')

        # Map to row indices for each dataset
        id_to_idx_train = (
            df_train.reset_index()[['index', 'scanner_patient_name']]
                   .astype({'scanner_patient_name': str})
                   .set_index('scanner_patient_name')['index']
                   .to_dict()
        )
        id_to_idx_test = (
            df_test.reset_index()[['index', 'scanner_patient_name']]
                  .astype({'scanner_patient_name': str})
                  .set_index('scanner_patient_name')['index']
                  .to_dict()
        )
        tr = np.array([id_to_idx_train[i] for i in train_ids if i in id_to_idx_train])
        te = np.array([id_to_idx_test[i] for i in test_ids if i in id_to_idx_test])
        if len(tr) == 0 or len(te) == 0:
            raise ValueError('Temporal split mapping produced empty train or test indices. Check ID overlap.')
        X_tr, X_te = X_train_all[tr], X_test_all[te]
        y_tr, y_te = y_train_all[tr], y_test_all[te]
        # Print per-dataset summaries
        print(f"Temporal TRAIN set: n={len(tr)} | features={len(feat_names)} | pos={int((y_tr==1).sum())} neg={int((y_tr==0).sum())}")
        print(f"Temporal TEST set:  n={len(te)} | features={len(feat_names)} | pos={int((y_te==1).sum())} neg={int((y_te==0).sum())}")
        # Optional ComBat harmonization fit on training only
        if args.combat_train_only:
            meta_path_cb = args.combat_metadata_path or args.scanner_metadata_path
            if (meta_path_cb is None) or (args.combat_batch_col is None):
                raise ValueError('--combat-train-only requires --combat-batch-col and a metadata path (--combat-metadata-path or --scanner-metadata-path)')
            meta_cb = pd.read_csv(meta_path_cb)
            meta_cb.columns = [str(c).strip() for c in meta_cb.columns]
            id_col_cb = args.scanner_id_col if args.scanner_id_col in meta_cb.columns else None
            if id_col_cb is None:
                import re
                def canon(s):
                    s = '' if s is None else str(s)
                    s = s.strip().lower().replace('-', '_').replace(' ', '_')
                    s = re.sub(r'[^a-z0-9_]+', '_', s)
                    s = re.sub(r'_+', '_', s).strip('_')
                    return s
                cmap = {canon(c): c for c in meta_cb.columns}
                if canon(args.scanner_id_col) in cmap:
                    id_col_cb = cmap[canon(args.scanner_id_col)]
            if id_col_cb is None or args.combat_batch_col not in meta_cb.columns:
                raise ValueError('ComBat metadata must include id col and combat batch col')
            meta_cb = meta_cb[[id_col_cb, args.combat_batch_col]].rename(columns={id_col_cb: 'scanner_patient_name', args.combat_batch_col: 'combat_batch'})
            # Build batch arrays aligned to df_train/df_test indices
            train_batches_all = df_train[['scanner_patient_name']].merge(meta_cb, on='scanner_patient_name', how='left')['combat_batch'].astype(str).values
            test_batches_all = df_test[['scanner_patient_name']].merge(meta_cb, on='scanner_patient_name', how='left')['combat_batch'].astype(str).values
            train_batches = np.where(pd.isna(train_batches_all), 'UNKNOWN', train_batches_all)[tr]
            test_batches = np.where(pd.isna(test_batches_all), 'UNKNOWN', test_batches_all)[te]
            X_tr, X_te = _combat_train_only(X_tr, X_te, train_batches, test_batches, verbose=True)
        # Optional reference-based standardization (train-only)
        elif args.ref_standardize_train_only:
            meta_path_cb = args.combat_metadata_path or args.scanner_metadata_path
            ref_col = args.ref_batch_col or args.combat_batch_col
            if (meta_path_cb is None) or (ref_col is None) or (args.ref_batch_value is None):
                raise ValueError('--ref-standardize-train-only requires --ref-batch-value and a metadata path plus a batch column (--ref-batch-col or --combat-batch-col)')
            meta_cb = pd.read_csv(meta_path_cb)
            meta_cb.columns = [str(c).strip() for c in meta_cb.columns]
            id_col_cb = args.scanner_id_col if args.scanner_id_col in meta_cb.columns else None
            if id_col_cb is None:
                import re
                def canon(s):
                    s = '' if s is None else str(s)
                    s = s.strip().lower().replace('-', '_').replace(' ', '_')
                    s = re.sub(r'[^a-z0-9_]+', '_', s)
                    s = re.sub(r'_+', '_', s).strip('_')
                    return s
                cmap = {canon(c): c for c in meta_cb.columns}
                if canon(args.scanner_id_col) in cmap:
                    id_col_cb = cmap[canon(args.scanner_id_col)]
            if id_col_cb is None or ref_col not in meta_cb.columns:
                raise ValueError('Reference standardization metadata must include id col and reference batch col')
            meta_cb = meta_cb[[id_col_cb, ref_col]].rename(columns={id_col_cb: 'scanner_patient_name', ref_col: 'ref_batch'})
            train_batches_all = df_train[['scanner_patient_name']].merge(meta_cb, on='scanner_patient_name', how='left')['ref_batch'].astype(str).values
            test_batches_all = df_test[['scanner_patient_name']].merge(meta_cb, on='scanner_patient_name', how='left')['ref_batch'].astype(str).values
            train_batches = np.where(pd.isna(train_batches_all), 'UNKNOWN', train_batches_all)[tr]
            test_batches = np.where(pd.isna(test_batches_all), 'UNKNOWN', test_batches_all)[te]
            X_tr, X_te = _ref_standardize_train_only(X_tr, X_te, train_batches, test_batches, args.ref_batch_value, verbose=True)
        # Optional train-only prefilter in temporal mode
        if args.prefilter and args.temporal_prefilter_train_only:
            X_tr_pf, fn_pf, pf_stats = _prefilter_unsupervised(
                X_tr, feat_names,
                missing_threshold=args.prefilter_missing_threshold,
                variance_threshold=args.prefilter_variance_threshold,
                correlation_threshold=args.prefilter_correlation_threshold,
            )
            keep_idx = [i for i, n in enumerate(feat_names) if n in set(fn_pf)]
            if keep_idx:
                X_tr, X_te = X_tr_pf, X_te[:, keep_idx]
                feat_names = fn_pf
        # Train-only selection
        thresh = (args.thresholds[0] if args.thresholds else 0.75)
        ensemble = EnsembleSTABL(
            n_runs=args.ensemble_runs,
            threshold=thresh,
            n_bootstraps=500,
            regularization=args.stabl_regularization,
            alpha=args.stabl_l1_ratio,
            max_features=args.n_features,
            consensus_threshold=args.consensus_threshold,
            n_jobs=args.n_jobs,
            random_state=42,
            C=args.stabl_c,
            tol=args.stabl_tol,
            max_iter=args.stabl_max_iter,
            artificial_type=(None if args.stabl_artificial_type == 'none' else args.stabl_artificial_type),
            lambda_grid=args.lambda_grid,
            n_lambda=args.n_lambda,
            fdr_start=args.fdr_start,
            fdr_end=args.fdr_end,
            fdr_step=args.fdr_step,
            perc_corr_group_threshold=args.stabl_corr_group_threshold,
            sample_fraction=args.stabl_sample_fraction
        )
        panel = list(ensemble.fit(pd.DataFrame(X_tr, columns=feat_names), pd.Series(y_tr)))

        extra_panel = []
        if getattr(args, 'temporal_extra_features', 0) > 0 and hasattr(ensemble, 'feature_frequencies'):
            freq_items = sorted(
                getattr(ensemble, 'feature_frequencies', {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )
            for feat, _count in freq_items:
                if feat in panel:
                    continue
                if feat not in feat_names:
                    continue
                extra_panel.append(feat)
                if len(extra_panel) >= int(args.temporal_extra_features):
                    break
            if extra_panel:
                panel.extend(extra_panel)
                print(f"Augmenting temporal panel with {len(extra_panel)} high-frequency features: {extra_panel}")

        sel_idx = [i for i, n in enumerate(feat_names) if n in panel]
        if not sel_idx:
            raise ValueError('No features selected in temporal training set')
        from sklearn.pipeline import Pipeline as SkPipe
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, GridSearchCV
        
        # Build base pipeline
        solver = 'liblinear'  # supports l1 and l2, provides predict_proba
        base_pipe = SkPipe([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('model', LogisticRegression(class_weight='balanced', solver=solver, max_iter=5000, penalty=args.lr_penalty))
        ])
        
        if args.lr_tune and len(np.unique(y_tr)) == 2:
            # Train-only tuning with adaptive CV splits based on minority class size
            class_counts = np.bincount(y_tr.astype(int)) if y_tr.dtype != bool else np.bincount(y_tr.astype(int))
            min_count = int(class_counts.min()) if class_counts.size == 2 else 0
            cv_splits = max(2, min(5, min_count))
            pipe, best_c, best_cv_auc = _tune_lr_c_optuna(
                X_tr[:, sel_idx],
                y_tr,
                base_pipe,
                cv_splits=cv_splits,
                trials=getattr(args, 'lr_optuna_trials', 30),
                seed=getattr(args, 'lr_optuna_seed', 42),
                c_min=getattr(args, 'lr_optuna_c_min', 1e-3),
                c_max=getattr(args, 'lr_optuna_c_max', 1e2),
                grid_values=getattr(args, 'lr_c_grid', None),
            )
        else:
            # No tuning; default C=1.0
            pipe = base_pipe
            pipe.set_params(model__C=1.0)
            best_c = 1.0
            best_cv_auc = None

        # Fit on full training
        pipe.fit(X_tr[:, sel_idx], y_tr)
        prob = pipe.predict_proba(X_te[:, sel_idx])[:, 1]
        auc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) == 2 else None

        # Optional stratified test-set bootstrap CI
        auc_ci = None
        if auc is not None and args.test_bootstrap and args.test_bootstrap > 0:
            scores = []
            rng = np.random.RandomState(42)
            pos_idx = np.where(y_te == 1)[0]
            neg_idx = np.where(y_te == 0)[0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            if n_pos > 0 and n_neg > 0:
                for _ in range(int(args.test_bootstrap)):
                    bs_pos = resample(pos_idx, replace=True, n_samples=n_pos, random_state=rng)
                    bs_neg = resample(neg_idx, replace=True, n_samples=n_neg, random_state=rng)
                    idx = np.concatenate([bs_pos, bs_neg])
                    y_b = y_te[idx]
                    if len(np.unique(y_b)) < 2:
                        continue
                    scores.append(roc_auc_score(y_b, prob[idx]))
                if scores:
                    auc_ci = [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # Map IDs to study dates for reporting/plotting context
        id_to_date = dict(zip(valid_meta['scanner_patient_name'].astype(str), valid_meta['StudyDate']))
        def _date_bounds(ids):
            dates = [id_to_date.get(i) for i in ids]
            dates = [d for d in dates if d is not None and not pd.isna(d)]
            if not dates:
                return None, None
            return min(dates), max(dates)

        train_start, train_end = _date_bounds(train_ids)
        test_start, test_end = _date_bounds(test_ids)

        def _ts_iso(ts: pd.Timestamp | None):
            if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
                return ts.date().isoformat()
            return None

        temporal_payload = {
            'train_size': int(len(tr)),
            'test_size': int(len(te)),
            'holdout_fraction': args.holdout_fraction,
            'train_radiomics_path': str(args.train_radiomics_path) if args.train_radiomics_path else str(radiomics_path),
            'test_radiomics_path': str(args.test_radiomics_path) if args.test_radiomics_path else str(radiomics_path),
            'common_features_count': int(len(feat_names)),
            'selected_features': panel,
            'extra_features': extra_panel,
            'auc': float(auc) if auc is not None else None,
            'auc_ci': auc_ci,
            'lr_penalty': args.lr_penalty,
            'best_c': float(best_c) if best_c is not None else None,
            'best_cv_auc_on_train': best_cv_auc,
            'train_pos': int((y_tr == 1).sum()),
            'train_neg': int((y_tr == 0).sum()),
            'test_pos': int((y_te == 1).sum()),
            'test_neg': int((y_te == 0).sum()),
            'train_date_start': _ts_iso(train_start),
            'train_date_end': _ts_iso(train_end),
            'test_date_start': _ts_iso(test_start),
            'test_date_end': _ts_iso(test_end)
        }

        with open(args.output_dir / 'temporal_eval.json', 'w') as f:
            json.dump(temporal_payload, f, indent=2)

        if PLOTTING_AVAILABLE:
            try:
                fig, ax = create_beautiful_figure('wide')
                groups = ['Training (oldest)', f"Holdout (newest {args.holdout_fraction:.0%})"]
                idx = np.arange(len(groups))
                train_neg = int((y_tr == 0).sum())
                train_pos = int((y_tr == 1).sum())
                test_neg = int((y_te == 0).sum())
                test_pos = int((y_te == 1).sum())
                neg_counts = [train_neg, test_neg]
                pos_counts = [train_pos, test_pos]

                bar_width = 0.55
                ax.bar(idx, neg_counts, bar_width, color=NORD_COLORS.get('nord8', '#88C0D0'), label='Non-event')
                ax.bar(idx, pos_counts, bar_width, bottom=neg_counts, color=NORD_COLORS.get('nord11', '#BF616A'), label='CR-POPF')

                for i, (neg, pos) in enumerate(zip(neg_counts, pos_counts)):
                    total = neg + pos
                    ax.text(i, total + 1, f"n={total}", ha='center', va='bottom', fontsize=16)
                    ax.text(i, neg / 2 if neg > 0 else 0.5, f"{neg} neg", ha='center', va='center', fontsize=14, color='white' if neg > 0 else NORD_COLORS.get('nord3', '#4C566A'))
                    ax.text(i, neg + pos / 2 if pos > 0 else neg + 0.5, f"{pos} pos", ha='center', va='center', fontsize=14, color='white' if pos > 0 else NORD_COLORS.get('nord3', '#4C566A'))

                ax.set_xticks(idx)
                ax.set_xticklabels(groups, rotation=0)
                ax.set_ylabel('Patients')
                ax.set_ylim(0, max(neg_counts[i] + pos_counts[i] for i in range(len(groups))) * 1.2 + 2)
                ax.legend(loc='upper left')

                auc_text = "AUROC: NA"
                if auc is not None:
                    if auc_ci:
                        auc_text = f"AUROC = {auc:.3f} (95% CI {auc_ci[0]:.3f}–{auc_ci[1]:.3f})"
                    else:
                        auc_text = f"AUROC = {auc:.3f}"
                date_text = []
                if train_start and train_end:
                    date_text.append(f"Train window: {train_start.date()} → {train_end.date()}")
                if test_start and test_end:
                    date_text.append(f"Holdout window: {test_start.date()} → {test_end.date()}")

                annotation = "\n".join(filter(None, [auc_text, f"Holdout fraction: {args.holdout_fraction:.0%}"] + date_text))
                fig.text(0.65, 0.6, annotation, ha='left', va='center')
                ax.set_title('Temporal Holdout Split (Train vs Holdout)')

                plot_base = Path(args.output_dir) / 'temporal_holdout_split'
                save_beautiful_figure(fig, plot_base)
                plt.close(fig)
            except Exception as plot_err:
                print(f"[WARN] Temporal holdout plot skipped: {plot_err}")

        msg = f"Temporal holdout AUC: {auc:.3f}" if auc is not None else "Temporal holdout could not compute AUC (single class)"
        if auc_ci:
            msg += f" [95% CI {auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
        if best_cv_auc is not None:
            msg += f" | Tuned C={best_c} (train CV AUC={best_cv_auc:.3f})"
        print(msg)
        return

    # Default: run full exploratory pipeline (selection + internal validation; for diagnostics only)
    pipeline = UltraOptimizedPipeline(
        ensemble_runs=args.ensemble_runs,
        thresholds=args.thresholds if args.thresholds else [0.7, 0.75, 0.8, 0.9],
        max_features=args.n_features,
        regularizations=[args.stabl_regularization],
        validation_method=args.evaluation,
        n_permutations=args.permutations,
        n_jobs=args.n_jobs,
        random_state=42,
        output_dir=args.output_dir
    )
    results = pipeline.run_pipeline(pd.DataFrame(X_prefiltered, columns=feat_names), pd.Series(y))
    print("\nNOTE: Ultra validation is exploratory/optimistic. Use V3 fixed-panel CV for publication.")


def generate_roc_from_existing_results(results_path: str):
    """Generate ROC plots from existing ultra-optimized results pickle file"""
    
    # Load results using joblib
    results = joblib.load(results_path)
    
    # Create output directory for ROC plots
    results_dir = Path(results_path).parent
    
    # Handle different result formats
    if isinstance(results, list):
        # This is the full config_results list
        # Find the best configuration
        best_config = max(results, key=lambda x: x.get('best_auc', 0))
    elif isinstance(results, dict):
        # This might be a single configuration
        best_config = results
    else:
        print("Unexpected results format")
        return
    
    if 'model_results' in best_config:
        # Initialize pipeline just for plotting
        pipeline = UltraOptimizedPipeline()
        
        # Generate ROC plots
        pipeline._create_roc_curves_plot(best_config['model_results'], results_dir.parent if results_dir.name == 'incremental_results' else results_dir)
        print(f"ROC curves saved to: {results_dir.parent if results_dir.name == 'incremental_results' else results_dir} / roc_curves")
    else:
        print("No model results found in the pickle file")


if __name__ == "__main__":
    # Check if we're just generating ROC plots from existing results
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-roc':
        if len(sys.argv) < 3:
            print("Usage: python script.py --generate-roc <path_to_results.pkl>")
            sys.exit(1)
        generate_roc_from_existing_results(sys.argv[2])
    else:
        main()
