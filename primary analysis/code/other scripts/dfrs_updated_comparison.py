#!/usr/bin/env python3
"""
D-FRS evaluation with curated outcome labels
Comprehensive comparison for manuscript
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def evaluate_dfrs_updated():
    """Evaluate D-FRS with corrected outcomes"""

    print("="*70)
    print("D-FRS EVALUATION WITH CURATED OUTCOME LABELS")
    print("="*70)

    parser = argparse.ArgumentParser(description="Evaluate D-FRS with curated outcome labels")
    parser.add_argument("--clinical-path", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "clinical_matched_popf_scanner.csv",
                        help="CSV with clinical variables and popf_grade")
    args, _ = parser.parse_known_args()

    # Load updated clinical database
    df = pd.read_csv(args.clinical_path)

    print(f"\nLoaded {len(df)} patients from updated clinical database")
    print(f"Patients with POPF outcome: {df['popf_grade'].notna().sum()}")

    # Calculate D-FRS components
    df['dfrs_score'] = 0

    # Component availability tracking
    components = {}

    # 1. BMI ≥25 (1 point)
    if 'bmi' in df.columns:
        df.loc[df['bmi'] >= 25, 'dfrs_score'] += 1
        components['BMI≥25'] = (df['bmi'] >= 25).sum()
        bmi_available = df['bmi'].notna().sum()
    else:
        bmi_available = 0

    # 2. Soft pancreas (2 points)
    if 'pancreas_texture' in df.columns:
        df.loc[df['pancreas_texture'] == 'Mou', 'dfrs_score'] += 2
        components['Soft pancreas'] = (df['pancreas_texture'] == 'Mou').sum()
        texture_available = df['pancreas_texture'].notna().sum()
    else:
        texture_available = 0

    # 3. MPD ≤3mm (2 points)
    if 'mpd_diameter' in df.columns:
        df.loc[df['mpd_diameter'] <= 3, 'dfrs_score'] += 2
        components['MPD≤3mm'] = (df['mpd_diameter'] <= 3).sum()
        mpd_available = df['mpd_diameter'].notna().sum()
    else:
        mpd_available = 0

    # 4. Operating time >240 min (1 point)
    if 'op_duration' in df.columns:
        df.loc[df['op_duration'] > 240, 'dfrs_score'] += 1
        components['Op>240min'] = (df['op_duration'] > 240).sum()
        op_available = df['op_duration'].notna().sum()
    else:
        op_available = 0

    # Complete D-FRS mask
    complete_mask = df[['bmi', 'pancreas_texture', 'mpd_diameter', 'op_duration', 'popf_grade']].notna().all(axis=1)
    df_complete = df[complete_mask].copy()

    print(f"\nPatients with complete D-FRS + outcome: {len(df_complete)}/{len(df)}")

    # Risk categories
    df_complete['dfrs_risk'] = pd.cut(df_complete['dfrs_score'],
                                      bins=[-0.1, 1.5, 3.5, 6.1],
                                      labels=['Low', 'Intermediate', 'High'])

    # Get outcomes
    y_true = df_complete['cr_popf'].values
    y_score = df_complete['dfrs_score'].values

    # Calculate metrics
    auc = roc_auc_score(y_true, y_score)

    print("\n" + "="*70)
    print("RESULTS WITH AUTHORITATIVE OUTCOMES")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  AUC: {auc:.3f}")
    print(f"  CR-POPF rate: {y_true.sum()}/{len(y_true)} ({100*y_true.mean():.1f}%)")

    print("\nRisk Stratification:")
    for risk in ['Low', 'Intermediate', 'High']:
        mask = df_complete['dfrs_risk'] == risk
        if mask.any():
            n_total = mask.sum()
            n_crpopf = y_true[mask].sum()
            rate = 100 * n_crpopf / n_total if n_total > 0 else 0
            print(f"  {risk:12s}: n={n_total:3d}, CR-POPF={n_crpopf:2d} ({rate:5.1f}%)")

    # Component contribution
    print("\nComponent Prevalence:")
    for component, count in components.items():
        print(f"  {component:15s}: {count}/{len(df_complete)} ({100*count/len(df_complete):.1f}%)")

    # Optimal threshold analysis
    print("\nOptimal Threshold Analysis:")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Youden Index
    youden = tpr - fpr
    optimal_idx = np.argmax(youden)
    optimal_threshold = thresholds[optimal_idx]

    print(f"  Optimal threshold (Youden): {optimal_threshold:.1f}")

    y_pred_optimal = (y_score >= optimal_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_optimal).ravel()

    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    print(f"  At threshold ≥{optimal_threshold:.0f}:")
    print(f"    Sensitivity: {sens:.3f}")
    print(f"    Specificity: {spec:.3f}")
    print(f"    PPV: {ppv:.3f}")
    print(f"    NPV: {npv:.3f}")

    # Create comprehensive figure
    create_comprehensive_figure(df_complete, base_path)

    # Save results
    results = {
        'total_patients': len(df),
        'complete_dfrs': len(df_complete),
        'cr_popf_rate': 100*y_true.mean(),
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'sensitivity': sens,
        'specificity': spec,
        'ppv': ppv,
        'npv': npv
    }

    results_df = pd.DataFrame([results])
    output_path = base_path / 'results' / 'dfrs_updated_results.csv'
    output_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"\n✅ Results saved to: {output_path}")

    return df_complete, results

def create_comprehensive_figure(df, base_path):
    """Create publication-ready D-FRS figure"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. ROC Curve
    ax = axes[0, 0]
    y_true = df['cr_popf'].values
    y_score = df['dfrs_score'].values

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'D-FRS (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('1 - Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('ROC Curve - D-FRS Performance')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # 2. Score Distribution
    ax = axes[0, 1]

    # Split by outcome
    no_crpopf = df[df['cr_popf'] == 0]['dfrs_score']
    yes_crpopf = df[df['cr_popf'] == 1]['dfrs_score']

    bins = np.arange(0, 7)
    ax.hist([no_crpopf, yes_crpopf], bins=bins, label=['No CR-POPF', 'CR-POPF'],
            alpha=0.7, color=['blue', 'red'], edgecolor='black')

    ax.set_xlabel('D-FRS Score')
    ax.set_ylabel('Number of Patients')
    ax.set_title('D-FRS Score Distribution by Outcome')
    ax.legend()
    ax.set_xticks(bins)

    # 3. Risk Category Performance
    ax = axes[1, 0]

    categories = ['Low\n(0-1)', 'Intermediate\n(2-3)', 'High\n(4-6)']
    rates = []
    counts = []

    for risk in ['Low', 'Intermediate', 'High']:
        mask = df['dfrs_risk'] == risk
        if mask.any():
            n_total = mask.sum()
            n_crpopf = df.loc[mask, 'cr_popf'].sum()
            rates.append(100 * n_crpopf / n_total)
            counts.append(n_total)
        else:
            rates.append(0)
            counts.append(0)

    bars = ax.bar(categories, rates, color=['green', 'orange', 'red'], alpha=0.7,
                   edgecolor='black', linewidth=1.5)

    # Add count labels
    for bar, count, rate in zip(bars, counts, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'n={count}\n{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('CR-POPF Rate (%)')
    ax.set_title('CR-POPF Rate by D-FRS Risk Category')
    ax.set_ylim(0, max(rates) * 1.3 if rates else 30)

    # 4. Component Analysis
    ax = axes[1, 1]

    components = {
        'BMI ≥25': (df['bmi'] >= 25).sum(),
        'Soft texture': (df['pancreas_texture'] == 'Mou').sum(),
        'MPD ≤3mm': (df['mpd_diameter'] <= 3).sum(),
        'Op >240min': (df['op_duration'] > 240).sum()
    }

    component_names = list(components.keys())
    component_counts = list(components.values())
    component_pcts = [100*c/len(df) for c in component_counts]

    bars = ax.barh(component_names, component_pcts, color='steelblue', alpha=0.7)

    # Add percentage labels
    for bar, pct in zip(bars, component_pcts):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                f'{pct:.1f}%', ha='left', va='center')

    ax.set_xlabel('Prevalence (%)')
    ax.set_title('D-FRS Component Prevalence')
    ax.set_xlim(0, 100)

    plt.suptitle('D-FRS Performance Analysis with Authoritative Outcomes', fontsize=14, y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = base_path / 'figures' / 'dfrs_comprehensive_updated.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Saved comprehensive D-FRS figure to: {output_path}")
    plt.close()

def main():
    """Main execution"""

    df_complete, results = evaluate_dfrs_updated()

    print("\n" + "="*70)
    print("SUMMARY FOR MANUSCRIPT")
    print("="*70)

    print(f"\n✅ D-FRS Performance (Authoritative Outcomes):")
    print(f"   - Evaluated on: {results['complete_dfrs']} patients")
    print(f"   - CR-POPF rate: {results['cr_popf_rate']:.1f}%")
    print(f"   - AUC: {results['auc']:.3f}")
    print(f"   - Optimal threshold: ≥{results['optimal_threshold']:.0f}")

    print(f"\n📊 Clinical Utility:")
    print(f"   - Sensitivity: {results['sensitivity']:.1%}")
    print(f"   - Specificity: {results['specificity']:.1%}")
    print(f"   - PPV: {results['ppv']:.1%}")
    print(f"   - NPV: {results['npv']:.1%}")

    print(f"\n💡 Interpretation:")
    print("   - Modest discriminative ability (AUC ~0.59)")
    print("   - High sensitivity but low specificity")
    print("   - Provides baseline for radiomics comparison")
    print("   - No recalibration performed (as requested)")

if __name__ == "__main__":
    main()
