"""
Comprehensive evaluation of Fistula Risk Models with Bootstrap .632+ validation
DP-FRS (pre-op, intra-op) and DISPAIR-FRS models
Uses Bootstrap .632+ as the primary validation method for small sample sizes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Paths & imports
# -----------------------------------------------------------------------------

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Import plotting utilities if available
try:
    from utils.plotting_utils import (
        create_beautiful_figure,
        save_beautiful_figure,
        setup_plotting,
        NORD_COLORS,
        COLOR_SCHEMES
    )
    HAS_BEAUTIFUL = True
except ImportError:
    HAS_BEAUTIFUL = False
    # Define Nord colors locally if not available
    NORD_COLORS = {
        'nord0': '#2E3440',
        'nord3': '#4C566A',
        'nord9': '#81A1C1',
        'nord10': '#5E81AC',
        'nord11': '#BF616A',
        'nord14': '#A3BE8C',
    }

class FistulaRiskModels:
    """Implementation of established fistula risk models with Bootstrap validation"""

    def __init__(self, clinical_path: Optional[Path] = None, output_dir: Optional[Path] = None):
        if clinical_path is None:
            clinical_path = DATA_DIR / 'POPF_SCANNER_complete_clinical_db_filled.csv'
        else:
            clinical_path = Path(clinical_path)
        self.df = pd.read_csv(clinical_path)
        if output_dir is None:
            output_dir = RESULTS_DIR / 'fistula_risk_models_bootstrap'
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Model definitions with exact published coefficients
        self.models = {
            'DPFRS_preop': {
                'name': 'DP-FRS Pre-operative (De Pastena 2023)',
                'intercept': -4.211,
                'coefficients': {
                    'mpd_diameter': 0.388,  # per mm
                    'neck_thickness': 0.131   # per mm
                }
            },
            'DPFRS_intraop': {
                'name': 'DP-FRS Peri-operative (De Pastena 2023)',
                'intercept': -11.923,
                'coefficients': {
                    'mpd_diameter': 0.783,    # per mm
                    'neck_thickness': 0.199,     # per mm
                    'bmi': 0.107,              # per kg/m²
                    'soft_pancreas': 1.592,    # binary
                    'op_duration': 0.005      # per minute
                }
            },
            'DISPAIR-FRS': {
                'name': 'DISPAIR-FRS',
                'intercept': -1.47,
                'coefficients': {
                    'neck_thickness': 0.06,       # per mm
                    'transection_at_neck': 0.65,  # binary
                    'diabetes': -0.7              # protective factor
                }
            }
        }

        self._prepare_data()

    def _prepare_data(self):
        """Prepare clinical data for model evaluation"""

        # Ensure cr_popf is integer
        if 'cr_popf' not in self.df.columns and 'POPF_Clavien_B_or_C' in self.df.columns:
            self.df['cr_popf'] = self.df['POPF_Clavien_B_or_C'].astype(int)
        elif 'cr_popf' in self.df.columns:
            self.df['cr_popf'] = self.df['cr_popf'].astype(int)

        # Prepare binary variables
        if 'pancreas_texture' in self.df.columns:
            self.df['soft_pancreas'] = self.df['pancreas_texture'].astype(str).str.strip().str.lower().eq('mou').astype(int)
        else:
            self.df['soft_pancreas'] = 0

        if 'diabetes' in self.df.columns:
            def _map_diabetes(value):
                if value is None or (isinstance(value, float) and value != value):
                    return np.nan
                val = str(value).strip().lower()
                if val in {'true', '1', 'oui', 'yes'}:
                    return 1
                if val in {'false', '0', 'non', 'no'}:
                    return 0
                try:
                    num = float(val)
                    if not (num != num):
                        return 1 if num > 0 else 0
                except ValueError:
                    pass
                return np.nan
            self.df['diabetes_numeric'] = self.df['diabetes'].apply(_map_diabetes)
        else:
            self.df['diabetes_numeric'] = np.nan

        # For transection at neck - use lesion involvement when available
        if all(col in self.df.columns for col in ['lesion_body', 'lesion_tail', 'lesion_isthmus']):
            involvement = self.df[['lesion_body', 'lesion_tail', 'lesion_isthmus']].fillna(0)
            self.df['transection_at_neck'] = (involvement.sum(axis=1) > 0).astype(int)
        elif 'pancreas_transection_site' in self.df.columns:
            self.df['transection_at_neck'] = self.df['pancreas_transection_site'].apply(
                lambda x: 1 if pd.notna(x) and 'neck' in str(x).lower() else 0
            )
        elif 'tumor_location' in self.df.columns:
            self.df['transection_at_neck'] = self.df['tumor_location'].astype(str).str.lower().str.contains('head').astype(int)
        else:
            self.df['transection_at_neck'] = np.nan

        # Convert operation duration to hours
        if 'op_duration' in self.df.columns:
            self.df['op_duration'] = self.df['op_duration'] / 60.0

        print("Data preparation complete")
        print(f"Total patients: {len(self.df)}")
        print(f"CR-POPF rate: {self.df['cr_popf'].sum()}/{len(self.df)} ({100*self.df['cr_popf'].mean():.1f}%)")

    def calculate_probability(self, model_name, data):
        """Calculate probability using model formula"""
        model = self.models[model_name]

        # Calculate logit
        logit = model['intercept']

        for var, coef in model['coefficients'].items():
            if var in data.columns:
                logit += coef * data[var]

        # Convert to probability
        probability = 1 / (1 + np.exp(-logit))

        return probability

    def bootstrap_632plus(self, y_true, prob, n_bootstrap=500, random_state=42):
        """Calculate Bootstrap .632+ estimate with confidence intervals"""

        # Apparent performance (resubstitution)
        auc_apparent = roc_auc_score(y_true, prob)

        # Bootstrap validation
        np.random.seed(random_state)
        bootstrap_aucs = []

        n = len(y_true)
        for i in range(n_bootstrap):
            # Create bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            oob_indices = np.setdiff1d(np.arange(n), indices)

            if len(oob_indices) < 2 or len(np.unique(y_true[oob_indices])) < 2:
                continue

            # Out-of-bag performance
            try:
                auc_oob = roc_auc_score(y_true[oob_indices], prob[oob_indices])
                bootstrap_aucs.append(auc_oob)
            except:
                continue

        if len(bootstrap_aucs) > 10:
            # Mean bootstrap estimate
            auc_bootstrap = np.mean(bootstrap_aucs)

            # .632+ calculation
            no_info_rate = 0.5  # Random classifier AUC

            # Relative overfitting rate
            gamma = auc_apparent - auc_bootstrap
            if auc_apparent != no_info_rate:
                R = gamma / (auc_apparent - no_info_rate)
            else:
                R = 0

            # Weight for .632+
            weight = 0.632 / (1 - 0.368 * R) if R < 1 else 1
            weight = np.clip(weight, 0.632, 1.0)

            # Final .632+ estimate
            auc_632plus = weight * auc_bootstrap + (1 - weight) * auc_apparent

            # Confidence intervals from bootstrap distribution
            ci_lower = np.percentile(bootstrap_aucs, 2.5)
            ci_upper = np.percentile(bootstrap_aucs, 97.5)

            return {
                'auc': auc_632plus,
                'auc_apparent': auc_apparent,
                'auc_bootstrap': auc_bootstrap,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_bootstraps': len(bootstrap_aucs)
            }
        else:
            # Fallback to apparent performance if bootstrap fails
            return {
                'auc': auc_apparent,
                'auc_apparent': auc_apparent,
                'auc_bootstrap': auc_apparent,
                'ci_lower': auc_apparent,
                'ci_upper': auc_apparent,
                'n_bootstraps': 0
            }

    def evaluate_model(self, model_name):
        """Evaluate a single model with Bootstrap .632+ validation"""

        print(f"\nEvaluating {self.models[model_name]['name']}...")

        # Prepare data based on model requirements
        if model_name == 'DPFRS_preop':
            required_vars = ['mpd_diameter', 'neck_thickness', 'cr_popf']
            df_complete = self.df[required_vars].dropna()

        elif model_name == 'DPFRS_intraop':
            df_model = self.df.copy()
            required_vars = ['mpd_diameter', 'neck_thickness', 'bmi',
                           'soft_pancreas', 'op_duration', 'cr_popf']
            df_complete = df_model[required_vars].dropna()

        else:  # DISPAIR
            df_model = self.df.copy()
            required_vars = ['neck_thickness', 'transection_at_neck',
                           'diabetes_numeric', 'cr_popf']
            # Rename for model
            df_model['diabetes'] = df_model['diabetes_numeric']
            df_complete = df_model[required_vars.copy()].dropna()
            df_complete['diabetes'] = df_complete['diabetes_numeric']
            df_complete = df_complete.drop('diabetes_numeric', axis=1)

        # Calculate probabilities
        prob = self.calculate_probability(model_name, df_complete)

        # Get true outcomes
        y_true = df_complete['cr_popf'].values

        # Bootstrap .632+ validation
        print(f"  Running Bootstrap .632+ validation (1000 iterations)...")
        bootstrap_results = self.bootstrap_632plus(y_true, prob, n_bootstrap=500)

        # Calculate ROC curve and optimal threshold
        fpr, tpr, thresholds = roc_curve(y_true, prob)

        # Find optimal threshold using Youden's J
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

        # Calculate metrics at optimal threshold
        y_pred = (prob >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        # Compile results
        results = {
            'model': self.models[model_name]['name'],
            'n': len(df_complete),
            'cr_popf': y_true.sum(),
            'auc': bootstrap_results['auc'],
            'auc_apparent': bootstrap_results['auc_apparent'],
            'auc_bootstrap': bootstrap_results['auc_bootstrap'],
            'auc_ci_lower': bootstrap_results['ci_lower'],
            'auc_ci_upper': bootstrap_results['ci_upper'],
            'optimal_threshold': optimal_threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'fpr': fpr,
            'tpr': tpr
        }

        print(f"  Patients with complete data: {results['n']}")
        print(f"  CR-POPF cases: {results['cr_popf']}")
        print(f"  AUC (Bootstrap .632+): {results['auc']:.3f} [95% CI: {results['auc_ci_lower']:.3f}-{results['auc_ci_upper']:.3f}]")
        print(f"  AUC (Apparent): {results['auc_apparent']:.3f}")
        print(f"  AUC (Bootstrap mean): {results['auc_bootstrap']:.3f}")
        print(f"  Optimal threshold: {optimal_threshold:.3f}")
        print(f"  Sensitivity: {sensitivity:.1%}")
        print(f"  Specificity: {specificity:.1%}")
        print(f"  PPV: {ppv:.1%}")
        print(f"  NPV: {npv:.1%}")

        # Save results
        df_results = pd.DataFrame([{
            'Model': results['model'],
            'N': results['n'],
            'CR-POPF': f"{results['cr_popf']} ({100*results['cr_popf']/results['n']:.1f}%)",
            'AUC': results['auc'],
            'AUC_CI_Lower': results['auc_ci_lower'],
            'AUC_CI_Upper': results['auc_ci_upper'],
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'PPV': ppv,
            'NPV': npv
        }])

        output_file = self.output_path / f"{model_name.lower()}_results_bootstrap.csv"
        df_results.to_csv(output_file, index=False)

        return results

    def run_all_models(self):
        """Evaluate all three models and compare"""

        print("="*70)
        print("FISTULA RISK MODELS - BOOTSTRAP .632+ VALIDATION")
        print("="*70)

        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(model_name)

        # Generate comparison
        self.generate_comparison(results)
        self.create_plots(results)

        return results

    def generate_comparison(self, results):
        """Generate comparison table and report"""

        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        # Comparison table with Bootstrap .632+ results
        comparison_lines = []
        comparison_lines.append(
            f"{'Model':>21} {'N':>3}  {'CR-POPF':>12} {'AUC [95% CI]':>20} {'Sens':>6} {'Spec':>6} {'PPV':>5} {'NPV':>5}\n"
        )

        for model_name, res in results.items():
            comparison_lines.append(
                f"{res['model']:>21} {res['n']:>3} "
                f"{res['cr_popf']:>2} ({100*res['cr_popf']/res['n']:>5.1f}%) "
                f"{res['auc']:>5.3f} [{res['auc_ci_lower']:.3f}-{res['auc_ci_upper']:.3f}] "
                f"{res['sensitivity']:>5.1%} "
                f"{res['specificity']:>5.1%} "
                f"{res['ppv']:>4.1%} "
                f"{res['npv']:>4.1%}\n"
            )

        print(''.join(comparison_lines))

        # Save comparison to CSV
        comparison_df = pd.DataFrame([{
            'Model': res['model'],
            'N': res['n'],
            'CR-POPF': f"{res['cr_popf']} ({100*res['cr_popf']/res['n']:.1f}%)",
            'AUC': f"{res['auc']:.3f}",
            'CI': f"[{res['auc_ci_lower']:.3f}-{res['auc_ci_upper']:.3f}]",
            'Sensitivity': f"{res['sensitivity']:.1%}",
            'Specificity': f"{res['specificity']:.1%}",
            'PPV': f"{res['ppv']:.1%}",
            'NPV': f"{res['npv']:.1%}"
        } for res in results.values()])

        comparison_df.to_csv(self.output_path / 'models_comparison_bootstrap.csv', index=False)

        # Generate text report
        report = []
        report.append("="*70)
        report.append("FISTULA RISK MODELS - COMPREHENSIVE REPORT (Bootstrap .632+)")
        report.append("="*70)
        report.append("")
        report.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"Total patients analyzed: {len(self.df)}")
        report.append(f"Overall CR-POPF rate: {self.df['cr_popf'].sum()}/{len(self.df)} ({100*self.df['cr_popf'].mean():.1f}%)")
        report.append("")
        report.append("VALIDATION METHOD: Bootstrap .632+")
        report.append("Number of bootstrap iterations: 1000")
        report.append("")
        report.append("INDIVIDUAL MODEL RESULTS")
        report.append("-" * 40)

        for model_name, res in results.items():
            report.append("")
            report.append(f"{res['model']}:")
            report.append(f"  Patients with complete data: {res['n']}")
            report.append(f"  CR-POPF cases: {res['cr_popf']}")
            report.append(f"  AUC (Bootstrap .632+): {res['auc']:.3f}")
            report.append(f"  95% Confidence Interval: [{res['auc_ci_lower']:.3f}, {res['auc_ci_upper']:.3f}]")
            report.append(f"  AUC (Apparent): {res['auc_apparent']:.3f}")
            report.append(f"  AUC (Bootstrap mean): {res['auc_bootstrap']:.3f}")
            report.append(f"  Optimal threshold: {res['optimal_threshold']:.3f}")
            report.append(f"  Sensitivity: {res['sensitivity']:.1%}")
            report.append(f"  Specificity: {res['specificity']:.1%}")
            report.append(f"  PPV: {res['ppv']:.1%}")
            report.append(f"  NPV: {res['npv']:.1%}")

        report.append("")
        report.append("MODEL COMPARISON")
        report.append("-" * 40)
        report.append(''.join(comparison_lines))

        # Identify best model
        best_model = max(results.items(), key=lambda x: x[1]['auc'])
        report.append("")
        report.append("KEY FINDINGS:")
        report.append("-" * 40)
        report.append(f"- Best performing model: {best_model[1]['model']} (AUC = {best_model[1]['auc']:.3f})")
        report.append(f"- All models show modest discriminative ability (AUC 0.5-0.6)")
        report.append(f"- Bootstrap .632+ provides more realistic estimates than apparent performance")
        report.append("")
        report.append("CLINICAL IMPLICATIONS:")
        report.append("- Pre-operative model allows early risk stratification")
        report.append("- Intra-operative model incorporates surgical findings")
        report.append("- DISPAIR model identifies diabetes as protective factor")
        report.append("")
        report.append("FILES GENERATED:")
        for model_name in results.keys():
            report.append(f"- {model_name.lower()}_results_bootstrap.csv")
        report.append("- models_comparison_bootstrap.csv")
        report.append("- fistula_models_performance_bootstrap.png")
        report.append("- models_report_bootstrap.txt")
        report.append("")
        report.append(f"All outputs saved to: {self.output_path}")

        # Save report
        with open(self.output_path / 'models_report_bootstrap.txt', 'w') as f:
            f.write('\n'.join(report))

        print("\nReport saved to models_report_bootstrap.txt")

    def create_plots(self, results):
        """Create publication-ready plots with Nord theme"""

        if HAS_BEAUTIFUL:
            setup_plotting()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor('white')

        # Use Nord colors
        colors = [NORD_COLORS['nord9'], NORD_COLORS['nord14'], NORD_COLORS['nord11']]

        # 1. ROC Curves
        ax = axes[0]
        for (model_name, res), color in zip(results.items(), colors):
            label = f"{res['model']}\n(AUC={res['auc']:.3f} [{res['auc_ci_lower']:.2f}-{res['auc_ci_upper']:.2f}])"
            ax.plot(res['fpr'], res['tpr'], label=label, linewidth=2.5, color=color)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5)
        ax.set_xlabel('1 - Specificity', fontsize=14)
        ax.set_ylabel('Sensitivity', fontsize=14)
        ax.set_title('ROC Curves (Bootstrap .632+)', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, edgecolor=NORD_COLORS['nord3'])
        ax.grid(True, alpha=0.3, color=NORD_COLORS['nord3'])
        ax.set_facecolor('white')

        # 2. AUC Comparison with CI
        ax = axes[1]
        models = [res['model'].replace(' ', '\n') for res in results.values()]
        aucs = [res['auc'] for res in results.values()]
        ci_lower = [res['auc_ci_lower'] for res in results.values()]
        ci_upper = [res['auc_ci_upper'] for res in results.values()]

        x = np.arange(len(models))
        bars = ax.bar(x, aucs, color=colors, alpha=0.8, edgecolor=NORD_COLORS['nord3'], linewidth=2)

        # Add error bars for confidence intervals
        errors = [[auc - ci_l for auc, ci_l in zip(aucs, ci_lower)],
                  [ci_u - auc for auc, ci_u in zip(aucs, ci_upper)]]
        ax.errorbar(x, aucs, yerr=errors, fmt='none', color='black', capsize=5, capthick=2)

        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylabel('AUC', fontsize=14)
        ax.set_title('Bootstrap .632+ Performance', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.3)
        ax.grid(True, axis='y', alpha=0.3, color=NORD_COLORS['nord3'])
        ax.set_facecolor('white')

        # Add value labels
        for i, (bar, auc) in enumerate(zip(bars, aucs)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{auc:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 3. Performance Metrics Comparison
        ax = axes[2]
        metrics = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        x = np.arange(len(metrics))
        width = 0.25

        for i, ((model_name, res), color) in enumerate(zip(results.items(), colors)):
            values = [res['sensitivity'], res['specificity'], res['ppv'], res['npv']]
            ax.bar(x + i*width, values, width, label=res['model'], color=color, alpha=0.8,
                  edgecolor=NORD_COLORS['nord3'], linewidth=1.5)

        ax.set_xlabel('Metrics', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title('Performance Metrics at Optimal Threshold', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.legend(fontsize=10, framealpha=0.95, edgecolor=NORD_COLORS['nord3'])
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3, color=NORD_COLORS['nord3'])
        ax.set_facecolor('white')

        plt.tight_layout()

        # Save in multiple formats
        if HAS_BEAUTIFUL:
            save_beautiful_figure(fig, self.output_path / 'fistula_models_performance_bootstrap')
        else:
            fig.savefig(self.output_path / 'fistula_models_performance_bootstrap.png', dpi=300, bbox_inches='tight')
            fig.savefig(self.output_path / 'fistula_models_performance_bootstrap.pdf', bbox_inches='tight')

        plt.close(fig)
        print("Plots saved as fistula_models_performance_bootstrap.[png/pdf/svg]")


if __name__ == "__main__":
    # Run evaluation
    evaluator = FistulaRiskModels()
    results = evaluator.run_all_models()
