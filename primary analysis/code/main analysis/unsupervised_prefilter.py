#!/usr/bin/env python3
"""
Unsupervised Pre-filtering for Radiomics Features
=================================================
Reduces feature space without using outcome labels to avoid leakage.

Methods:
1. Near-zero variance removal
2. Highly correlated feature removal (keep one from each pair)
3. Missing value filtering
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Optional BeautifulFigures-style plotting utilities
try:
    from utils.plotting_utils import (
        setup_plotting,
        create_beautiful_figure,
        save_beautiful_figure,
        NORD_COLORS,
    )
    HAS_BEAUTIFUL = True
except Exception:
    HAS_BEAUTIFUL = False


class UnsupervisedRadiomicsFilter:
    """
    Unsupervised filtering of radiomics features.
    No outcome information used - prevents data leakage.
    """
    
    def __init__(self, variance_threshold=0.01, correlation_threshold=0.95, 
                 missing_threshold=0.3):
        """
        Parameters:
        -----------
        variance_threshold : float
            Minimum variance to keep feature (default 0.01)
        correlation_threshold : float  
            Maximum correlation allowed between features (default 0.95)
        missing_threshold : float
            Maximum proportion of missing values allowed (default 0.3)
        """
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.missing_threshold = missing_threshold
        
        # Track filtering results
        self.filtering_stats = {}
        self.removed_features = {
            'missing': [],
            'variance': [],
            'correlation': []
        }
        
    def fit_transform(self, X, feature_names):
        """
        Apply all unsupervised filters sequentially.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix (can contain NaN)
        feature_names : list
            Names of features
            
        Returns:
        --------
        X_filtered : array-like
            Filtered feature matrix
        feature_names_filtered : list
            Names of remaining features
        """
        print("="*60)
        print("UNSUPERVISED RADIOMICS PRE-FILTERING")
        print("="*60)
        
        n_original = X.shape[1]
        self.filtering_stats['n_original'] = n_original
        
        # Step 1: Remove features with too many missing values
        print(f"\n1. Missing value filtering (>{self.missing_threshold*100:.0f}% missing)...")
        X, feature_names = self._filter_missing(X, feature_names)
        
        # Step 2: Remove near-zero variance features
        print(f"\n2. Variance filtering (variance < {self.variance_threshold})...")
        X, feature_names = self._filter_variance(X, feature_names)
        
        # Step 3: Remove highly correlated features
        print(f"\n3. Correlation filtering (|r| > {self.correlation_threshold})...")
        X, feature_names = self._filter_correlation(X, feature_names)
        
        # Final stats
        self.filtering_stats['n_final'] = len(feature_names)
        self.filtering_stats['n_removed'] = n_original - len(feature_names)
        self.filtering_stats['reduction_percent'] = (
            (n_original - len(feature_names)) / n_original * 100
        )
        
        print(f"\n{'='*60}")
        print(f"FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Original features: {n_original}")
        print(f"Remaining features: {len(feature_names)}")
        print(f"Removed: {self.filtering_stats['n_removed']} ({self.filtering_stats['reduction_percent']:.1f}%)")
        
        return X, feature_names
    
    def _filter_missing(self, X, feature_names):
        """Remove features with too many missing values."""
        missing_prop = np.isnan(X).mean(axis=0)
        keep_mask = missing_prop <= self.missing_threshold
        
        removed = [feat for feat, keep in zip(feature_names, keep_mask) if not keep]
        self.removed_features['missing'] = removed
        self.filtering_stats['n_missing_removed'] = len(removed)
        
        print(f"   Removed {len(removed)} features")
        
        return X[:, keep_mask], [f for f, k in zip(feature_names, keep_mask) if k]
    
    def _filter_variance(self, X, feature_names):
        """Remove near-zero variance features."""
        # Impute for variance calculation
        X_imputed = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        
        # Calculate variance
        variances = np.var(X_imputed, axis=0)
        keep_mask = variances > self.variance_threshold
        
        removed = [feat for feat, keep in zip(feature_names, keep_mask) if not keep]
        self.removed_features['variance'] = removed
        self.filtering_stats['n_variance_removed'] = len(removed)
        
        print(f"   Removed {len(removed)} features")
        
        return X[:, keep_mask], [f for f, k in zip(feature_names, keep_mask) if k]
    
    def _filter_correlation(self, X, feature_names):
        """Remove highly correlated features (keep one with higher variance)."""
        # Impute for correlation calculation
        X_imputed = np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        
        # Calculate correlation matrix using Spearman correlation (IBSI standard)
        print("   Calculating Spearman correlation matrix...")
        corr_matrix = np.abs(spearmanr(X_imputed)[0])
        
        # Find highly correlated pairs
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((corr_matrix > self.correlation_threshold) & upper_tri)
        
        print(f"   Found {len(high_corr_pairs[0])} highly correlated pairs")
        
        # Decide which features to remove
        to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            if i not in to_remove and j not in to_remove:
                # Keep the one with higher variance
                var_i = np.nanvar(X[:, i])
                var_j = np.nanvar(X[:, j])
                if var_i < var_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        
        # Create keep mask
        keep_mask = np.ones(X.shape[1], dtype=bool)
        keep_mask[list(to_remove)] = False
        
        removed = [feature_names[i] for i in to_remove]
        self.removed_features['correlation'] = removed
        self.filtering_stats['n_correlation_removed'] = len(removed)
        
        print(f"   Removed {len(removed)} features")
        
        return X[:, keep_mask], [f for f, k in zip(feature_names, keep_mask) if k]
    
    def plot_filtering_summary(self, save_path='results/unsupervised_filtering_summary'):
        """Create visualization of filtering results."""
        # Ensure consistent styling
        if HAS_BEAUTIFUL:
            setup_plotting()
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Features removed by stage
        ax1 = axes[0, 0]
        stages = ['Missing\nValues', 'Low\nVariance', 'High\nCorrelation']
        counts = [
            self.filtering_stats['n_missing_removed'],
            self.filtering_stats['n_variance_removed'],
            self.filtering_stats['n_correlation_removed']
        ]
        # Use Nord palette if available
        if HAS_BEAUTIFUL:
            colors = [NORD_COLORS['nord11'], NORD_COLORS['nord12'], NORD_COLORS['nord13']]
            edge = NORD_COLORS['nord3']
            bars = ax1.bar(stages, counts, color=colors, edgecolor=edge, linewidth=2)
        else:
            bars = ax1.bar(stages, counts, color=['red', 'orange', 'yellow'])
        ax1.set_ylabel('Number of Features Removed')
        ax1.set_title('A. Features Removed by Filter Type')
        
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    str(count), ha='center', va='bottom')
        
        # 2. Cumulative feature reduction
        ax2 = axes[0, 1]
        n_orig = self.filtering_stats['n_original']
        cumulative = [
            n_orig,
            n_orig - self.filtering_stats['n_missing_removed'],
            n_orig - self.filtering_stats['n_missing_removed'] - 
                    self.filtering_stats['n_variance_removed'],
            self.filtering_stats['n_final']
        ]
        
        if HAS_BEAUTIFUL:
            ax2.plot(['Original', 'After\nMissing', 'After\nVariance', 'Final'],
                     cumulative, 'o-', linewidth=3, markersize=8, color=NORD_COLORS['nord10'])
        else:
            ax2.plot(['Original', 'After\nMissing', 'After\nVariance', 'Final'],
                     cumulative, 'o-', linewidth=2, markersize=8)
        ax2.set_ylabel('Number of Features')
        ax2.set_title('B. Cumulative Feature Reduction')
        ax2.grid(True, alpha=0.3)
        
        # Add annotations
        for i, n in enumerate(cumulative):
            ax2.text(i, n + 10, str(n), ha='center', va='bottom')
        
        # 3. Summary statistics
        ax3 = axes[1, 0]
        ax3.axis('off')
        
        summary_text = f"""Filtering Summary
=================
Original features: {self.filtering_stats['n_original']}
Final features: {self.filtering_stats['n_final']}
Total removed: {self.filtering_stats['n_removed']}
Reduction: {self.filtering_stats['reduction_percent']:.1f}%

Thresholds Used:
• Missing values: >{self.missing_threshold*100:.0f}%
• Variance: <{self.variance_threshold}
• Correlation: >{self.correlation_threshold}

Filter Results:
• Missing: {self.filtering_stats['n_missing_removed']} removed
• Variance: {self.filtering_stats['n_variance_removed']} removed  
• Correlation: {self.filtering_stats['n_correlation_removed']} removed"""
        
        ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 4. Feature type breakdown (if available)
        ax4 = axes[1, 1]
        
        # Pie chart of reduction percentage
        sizes = [self.filtering_stats['n_final'], self.filtering_stats['n_removed']]
        labels = ['Retained', 'Removed']
        if HAS_BEAUTIFUL:
            colors = [NORD_COLORS['nord14'], NORD_COLORS['nord11']]
        else:
            colors = ['green', 'red']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('D. Overall Feature Retention')
        
        plt.suptitle('Unsupervised Radiomics Pre-filtering Results', fontsize=16, fontweight='bold')
        plt.tight_layout()
        # Save plots (multi-format if Beautiful utils available)
        base = Path(save_path)
        # If user passed a filename with extension, drop it for multi-export
        if base.suffix:
            base = base.with_suffix('')
        if HAS_BEAUTIFUL:
            save_beautiful_figure(fig, base)
            plt.close(fig)
            print(f"\nSummary plot saved to {base.with_suffix('.png')}, .svg, .pdf")
        else:
            plt.savefig(base.with_suffix('.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"\nSummary plot saved to {base.with_suffix('.png')}")


HONORIFICS = {
    "mr", "mrs", "ms", "mme", "mlle", "mle", "dr", "prof", "monsieur", "madame",
    "m", "mme.", "mlle.", "mr.", "mrs.", "dr.", "prof."
}

def _strip_accents(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def _canonicalize_id(s) -> str:
    import re
    s = "" if s is None else str(s)
    s = _strip_accents(s).lower()
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    tokens = [t for t in s.split("_") if t and t not in HONORIFICS]
    s = "_".join(tokens)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def apply_unsupervised_filtering(radiomics_path, matches_path, output_path,
                               variance_threshold=0.01, correlation_threshold=0.95,
                               missing_threshold=0.3, positive_grades=("B","C"),
                               allow_id_normalization=False):
    """Apply unsupervised filtering to radiomics data."""
    print("Loading data...")
    
    # Load data
    radiomics_df = pd.read_csv(radiomics_path)
    matches_df = pd.read_csv(matches_path)
    
    # Get patient IDs from matches
    valid_patients = matches_df['scanner_patient_name'].unique()
    
    # Filter radiomics to valid patients
    # Check which column name is used in radiomics
    if 'scanner_patient_name' in radiomics_df.columns:
        patient_col = 'scanner_patient_name'
    elif 'patient_name' in radiomics_df.columns:
        patient_col = 'patient_name'
    elif 'patient_id' in radiomics_df.columns:
        # Some exports use 'patient_id' with the same content as scanner_patient_name
        patient_col = 'patient_id'
    else:
        raise KeyError("No patient identifier column found (expected one of: 'scanner_patient_name', 'patient_name', 'patient_id')")
    radiomics_df = radiomics_df[radiomics_df[patient_col].isin(valid_patients)]
    
    # Get radiomics features only (exclude patient ID)
    exclude_cols = ['scanner_patient_name', 'patient_name']
    
    radiomics_cols = [col for col in radiomics_df.columns if col not in exclude_cols]
    numeric_cols = radiomics_df.select_dtypes(include=[np.number]).columns
    radiomics_cols = [col for col in radiomics_cols if col in numeric_cols]
    
    print(f"\nOriginal radiomics features: {len(radiomics_cols)}")
    
    # Extract feature matrix
    X = radiomics_df[radiomics_cols].values
    
    # Apply filtering
    filter_obj = UnsupervisedRadiomicsFilter(
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        missing_threshold=missing_threshold
    )
    
    X_filtered, features_filtered = filter_obj.fit_transform(X, radiomics_cols)
    print(f"\nRetained features: {len(features_filtered)}")
    
    # Create visualization
    filter_obj.plot_filtering_summary()
    
    # Save filtered data with patient IDs
    filtered_df = pd.DataFrame(X_filtered, columns=features_filtered)
    filtered_df.insert(0, 'scanner_patient_name', radiomics_df[patient_col].values)

    # Add POPF grade from matches (exact, then canonical fallback)
    filtered_df = filtered_df.merge(
        matches_df[['scanner_patient_name', 'popf_grade']],
        on='scanner_patient_name', how='left'
    )

    n_missing = int(filtered_df['popf_grade'].isna().sum())
    if n_missing > 0 and allow_id_normalization:
        # Try normalized join to recover additional matches
        tmp = filtered_df.copy()
        tmp['_canon'] = tmp['scanner_patient_name'].map(_canonicalize_id)
        m2 = matches_df.copy()
        m2['_canon'] = m2['scanner_patient_name'].map(_canonicalize_id)
        m2 = m2.drop_duplicates('_canon')[['_canon', 'popf_grade']]
        tmp = tmp.merge(m2.rename(columns={'popf_grade': 'popf_grade_canon'}), on='_canon', how='left')
        tmp['popf_grade'] = tmp['popf_grade'].fillna(tmp['popf_grade_canon'])
        filtered_df = tmp.drop(columns=['_canon', 'popf_grade_canon'])
        n_missing = int(filtered_df['popf_grade'].isna().sum())

    # Drop rows without outcomes (policy)
    if n_missing > 0:
        out_path = Path(output_path)
        unmatched_path = out_path.with_name(out_path.stem + '_unmatched.csv')
        filtered_df.loc[filtered_df['popf_grade'].isna(), ['scanner_patient_name']].to_csv(unmatched_path, index=False)
        print(f"Dropping {n_missing} radiomics rows without POPF grade. Unmatched IDs saved to {unmatched_path}")
        filtered_df = filtered_df.dropna(subset=['popf_grade']).reset_index(drop=True)

    pos = set([str(g).upper() for g in positive_grades])
    filtered_df['cr_popf'] = filtered_df['popf_grade'].apply(lambda x: 1 if str(x).upper() in pos else 0)

    filtered_df.to_csv(output_path, index=False)
    print(f"\nFiltered data saved to {output_path}")
    
    # Save feature list
    feature_list_path = output_path.replace('.csv', '_features.txt')
    with open(feature_list_path, 'w') as f:
        for feat in features_filtered:
            f.write(f"{feat}\n")
    
    return filter_obj, X_filtered, features_filtered


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Apply unsupervised filtering to radiomics features')
    parser.add_argument('--input', required=True, help='Path to radiomics data CSV')
    parser.add_argument('--matches', required=True, help='Path to outcome matches CSV (e.g., data/outcome_matches.csv)')
    parser.add_argument('--output', default='data/radiomics_filtered_unsupervised.csv',
                       help='Output path for filtered radiomics')
    parser.add_argument('--variance-threshold', type=float, default=0.01,
                       help='Minimum variance threshold')
    parser.add_argument('--correlation-threshold', type=float, default=0.95,
                       help='Maximum correlation threshold')
    parser.add_argument('--missing-threshold', type=float, default=0.3,
                       help='Maximum missing value threshold')
    parser.add_argument('--positive-grades', type=str, default='B,C',
                        help="Comma-separated POPF grades considered positive (default: 'B,C'). E.g., 'B,C,BL'")
    parser.add_argument('--allow-id-normalization', action='store_true', default=False,
                        help='Attempt a normalized ID fallback join when exact match fails (default: off)')
    
    args = parser.parse_args()
    
    posg = tuple([t.strip() for t in args.positive_grades.split(',') if t.strip()])
    filter_obj, X_filtered, features_filtered = apply_unsupervised_filtering(
        args.input, args.matches, args.output,
        args.variance_threshold, args.correlation_threshold, args.missing_threshold,
        positive_grades=posg, allow_id_normalization=args.allow_id_normalization
    )
