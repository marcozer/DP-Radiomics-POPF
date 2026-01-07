"""
Lightweight BeautifulFigures-style plotting utils used across scripts.

Provides a consistent Nord-themed palette, convenient figure factories,
and multi-format export (PNG/SVG/PDF).

These helpers are intentionally dependency-light (matplotlib only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

from cycler import cycler
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Nord theme palette (subset)
NORD_COLORS = {
    'nord0': '#2E3440',  # black-ish
    'nord1': '#3B4252',
    'nord2': '#434C5E',
    'nord3': '#4C566A',
    'nord4': '#D8DEE9',
    'nord5': '#E5E9F0',
    'nord6': '#ECEFF4',
    'nord7': '#8FBCBB',
    'nord8': '#88C0D0',
    'nord9': '#81A1C1',  # blue
    'nord10': '#5E81AC', # dark blue
    'nord11': '#BF616A', # red
    'nord12': '#D08770', # orange
    'nord13': '#EBCB8B', # yellow
    'nord14': '#A3BE8C', # green
    'nord15': '#B48EAD', # purple
}


# Simple color schemes used in summary plots
COLOR_SCHEMES = {
    'models': [
        NORD_COLORS['nord9'],
        NORD_COLORS['nord14'],
        NORD_COLORS['nord11'],
        NORD_COLORS['nord10'],
        NORD_COLORS['nord13'],
        NORD_COLORS['nord15'],
    ]
}


def setup_plotting() -> None:
    """Apply BeautifulFigures-style rcParams (fonts, sizes, export policy)."""
    # Ensure Courier New (or a close mono fallback) mirrors historical figures.
    try:
        from matplotlib import font_manager as _fm

        _ = _fm.findfont('Courier New', fallback_to_default=False)
        font_family = 'Courier New'
    except Exception:
        font_family = 'DejaVu Sans Mono'

    mpl.rcParams.update({
        # Fonts (scaled ×1.5 vs previous defaults)
        'font.family': font_family,
        'font.size': 27,  # 18 * 1.5
        'mathtext.default': 'regular',
        # Axes
        'axes.titlesize': 33,  # 22 * 1.5
        'axes.titleweight': 'semibold',
        'axes.titlepad': 30,
        'axes.labelsize': 27,  # 18 * 1.5
        'axes.labelweight': 'semibold',
        'axes.labelpad': 18,
        'axes.edgecolor': NORD_COLORS['nord3'],
        'axes.facecolor': 'white',
        'axes.linewidth': 1.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.8,
        'grid.color': NORD_COLORS['nord3'],
        # Tick styling (major + minor)
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'xtick.major.size': 9,
        'ytick.major.size': 9,
        'xtick.major.width': 2.4,
        'ytick.major.width': 2.4,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.size': 6,
        'ytick.minor.size': 6,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        # Legend
        'legend.fontsize': 24,
        'legend.frameon': False,
        # Figure/export
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,   # keep fonts, avoid type 3
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        # Colour cycle aligned with Nord palette (cool-to-warm harmony)
        'axes.prop_cycle': cycler('color', [
            NORD_COLORS['nord9'],
            NORD_COLORS['nord14'],
            NORD_COLORS['nord11'],
            NORD_COLORS['nord10'],
            NORD_COLORS['nord13'],
            NORD_COLORS['nord15'],
        ]),
    })


def _figsize(kind: str) -> Tuple[float, float]:
    kind = (kind or '').lower()
    if kind == 'wide':
        return (12, 6)
    if kind == 'tall':
        return (8, 10)
    if kind == 'square':
        return (8, 8)
    return (10, 6)


def create_beautiful_figure(kind: str = 'wide'):
    """Return (fig, ax) with consistent styling and size."""
    setup_plotting()
    fig, ax = plt.subplots(figsize=_figsize(kind))
    ax.set_axisbelow(True)
    return fig, ax


def save_beautiful_figure(fig, path_base: Path | str):
    """Save the figure to PNG, SVG, and PDF given a base path (no extension)."""
    path_base = Path(path_base)
    fig.savefig(path_base.with_suffix('.png'))
    fig.savefig(path_base.with_suffix('.svg'))
    fig.savefig(path_base.with_suffix('.pdf'))


def plot_model_comparison_with_ci(models: Sequence[str],
                                  aucs: Sequence[float],
                                  cis: Sequence[Sequence[float]] | None = None,
                                  title: str | None = None):
    """Generate a Nord-themed horizontal bar plot comparing model AUROCs.

    Parameters
    ----------
    models : iterable of str
        Model names (display order matches provided order).
    aucs : iterable of float
        Point-estimate AUROC per model.
    cis : iterable of (lo, hi) sequences, optional
        Confidence intervals aligned with models; if absent, bars show point estimates only.
    title : str, optional
        Plot title to annotate at the top of the figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    setup_plotting()
    fig, ax = create_beautiful_figure('tall')

    models = list(models)
    aucs = list(aucs)
    cis = list(cis) if cis is not None else [None] * len(models)

    y_pos = np.arange(len(models))[::-1]
    palette = COLOR_SCHEMES.get('models', [NORD_COLORS['nord9']])

    for idx, (model, auc, ci) in enumerate(zip(models, aucs, cis)):
        color = palette[idx % len(palette)]
        ax.barh(y_pos[idx], auc, color=color, edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)

        if ci and len(ci) == 2 and ci[1] >= ci[0]:
            lo, hi = float(ci[0]), float(ci[1])
            ax.errorbar([auc], [y_pos[idx]],
                        xerr=[[max(0.0, auc - lo)], [max(0.0, hi - auc)]],
                        fmt='none', color=NORD_COLORS['nord0'], capsize=10, capthick=2)
            label = f"{auc:.3f}\n[{lo:.3f}, {hi:.3f}]"
        else:
            label = f"{auc:.3f}"

        ax.text(auc + 0.01, y_pos[idx], label, va='center', ha='left', fontsize=14)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlim(0.4, 1.0)
    ax.axvline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5)
    ax.set_xlabel('AUROC')
    if title:
        ax.set_title(title, pad=18)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_roc_with_confidence_band(fpr: Iterable[float],
                                  tpr: Iterable[float],
                                  ci_band: dict | None = None,
                                  label: str | None = None):
    """Plot ROC curve with optional confidence band.

    Parameters
    ----------
    fpr, tpr : iterable of float
        False-positive and true-positive rates (monotonic increasing FPR expected).
    ci_band : dict, optional
        Mapping with keys 'lower', 'upper', and 'grid' describing percentile bands.
    label : str, optional
        Legend label for the ROC curve.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    setup_plotting()
    fig, ax = create_beautiful_figure('square')

    if ci_band:
        grid = np.asarray(ci_band.get('grid'))
        lower = np.asarray(ci_band.get('lower'))
        upper = np.asarray(ci_band.get('upper'))
        if grid.size and lower.size and upper.size:
            ax.fill_between(grid, lower, upper, color=NORD_COLORS['nord9'], alpha=0.2, label='95% CI')

    ax.plot(list(fpr), list(tpr), color=NORD_COLORS['nord9'], linewidth=3,
            label=label or 'ROC curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color=NORD_COLORS['nord11'], alpha=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig, ax


def plot_validation_comparison(method_results: dict, path_base: Path | str):
    """Render a validation method comparison bar chart matching legacy style.

    method_results: mapping name -> { 'auc': float, 'ci': [lo, hi] or 'std': float }
    path_base: base filepath (no extension) for PNG/SVG/PDF.
    """
    setup_plotting()
    fig, ax = create_beautiful_figure('wide')

    methods = []
    aucs = []
    errors = []

    # Preserve a conventional order where possible
    order = ['bootstrap_632_plus', 'repeated_stratified_cv', 'loocv', 'simple_bootstrap']
    for key in order:
        if key in method_results:
            methods.append(key.replace('_', ' ').title())
            aucs.append(method_results[key].get('auc', 0.0))
            if 'ci' in method_results[key] and method_results[key]['ci']:
                lo, hi = method_results[key]['ci']
                err_low = max(0.0, method_results[key]['auc'] - lo)
                err_high = max(0.0, hi - method_results[key]['auc'])
                errors.append([err_low, err_high])
            elif 'std' in method_results[key]:
                s = float(method_results[key]['std'])
                errors.append([s, s])
            else:
                errors.append([0.0, 0.0])

    x = np.arange(len(methods))
    best_idx = int(np.argmax(aucs)) if aucs else 0
    colors = [NORD_COLORS['nord14'] if i == best_idx else NORD_COLORS['nord9'] for i in range(len(methods))]
    bars = ax.bar(x, aucs, color=colors, edgecolor=NORD_COLORS['nord3'], linewidth=2, alpha=0.9)
    if errors:
        import numpy as _np
        e = _np.array(errors).T if hasattr(errors, '__len__') else None
        if e is not None and e.shape[0] == 2:
            ax.errorbar(x, aucs, yerr=e, fmt='none', color=NORD_COLORS['nord0'], capsize=8, capthick=2)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=14)
    ax.set_ylabel('AUC', fontsize=16)
    ax.set_ylim([0.4, 1.0])
    ax.axhline(0.5, color=NORD_COLORS['nord11'], linestyle='--', alpha=0.5, label='Random')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend(fontsize=12)

    # Value labels with CI when present
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        if i < len(errors) and errors[i][0] > 0 and errors[i][1] > 0:
            ci_text = f"{auc:.3f}\n[{auc - errors[i][0]:.3f}, {auc + errors[i][1]:.3f}]"
        else:
            ci_text = f"{auc:.3f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, ci_text,
                ha='center', va='bottom', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    from pathlib import Path as _P
    save_beautiful_figure(fig, _P(path_base))
    plt.close(fig)


def plot_single_roc(y_true, y_score, path_base: Path | str, n_boot: int = 1000,
                    label_auc: float | None = None,
                    label_ci: tuple[float, float] | list[float] | None = None):
    """Plot ROC with bootstrap confidence band (legacy-style aesthetics).

    - y_true: array-like of 0/1
    - y_score: array-like of probabilities/scores
    - path_base: base filepath (no extension) for PNG/SVG/PDF
    - n_boot: number of bootstrap resamples for CI band
    """
    import numpy as _np
    from sklearn.metrics import roc_curve, auc

    setup_plotting()

    y_true = _np.asarray(y_true).astype(int)
    y_score = _np.asarray(y_score).astype(float)

    # Base ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    base_auc = auc(fpr, tpr)

    # Bootstrap band on a fixed grid
    rng = _np.random.RandomState(42)
    grid = _np.linspace(0, 1, 101)
    tprs = []
    aucs = []
    idx = _np.arange(len(y_true))
    for _ in range(int(n_boot)):
        bs = rng.choice(idx, size=len(idx), replace=True)
        y_b = y_true[bs]
        s_b = y_score[bs]
        if _np.unique(y_b).size < 2:
            continue
        f_b, t_b, _ = roc_curve(y_b, s_b)
        # Interpolate TPR at base grid
        tpr_i = _np.interp(grid, f_b, t_b)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)
        aucs.append(auc(f_b, t_b))

    tprs = _np.array(tprs) if len(tprs) else _np.zeros((1, grid.size))
    aucs = _np.array(aucs) if len(aucs) else _np.array([base_auc])
    tpr_med = _np.median(tprs, axis=0)
    tpr_lo = _np.percentile(tprs, 2.5, axis=0)
    tpr_hi = _np.percentile(tprs, 97.5, axis=0)
    auc_lo, auc_hi = float(_np.percentile(aucs, 2.5)), float(_np.percentile(aucs, 97.5))

    fig, ax = create_beautiful_figure('square')
    # Confidence band
    ax.fill_between(grid, tpr_lo, tpr_hi, color=NORD_COLORS['nord9'], alpha=0.2, label='95% CI')
    # Median ROC
    if label_auc is not None:
        if label_ci is not None and len(label_ci) == 2:
            lo, hi = float(label_ci[0]), float(label_ci[1])
            label_txt = f'ROC (AUC = {label_auc:.3f} [{lo:.3f}-{hi:.3f}])'
        else:
            label_txt = f'ROC (AUC = {label_auc:.3f})'
    else:
        label_txt = f'ROC (AUC = {base_auc:.3f} [{auc_lo:.3f}-{auc_hi:.3f}])'
    ax.plot(grid, tpr_med, color=NORD_COLORS['nord9'], linewidth=3, label=label_txt)
    # Diagonal
    ax.plot([0, 1], [0, 1], color=NORD_COLORS['nord11'], linestyle='--', alpha=0.6)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    save_beautiful_figure(fig, Path(path_base))
    plt.close(fig)
