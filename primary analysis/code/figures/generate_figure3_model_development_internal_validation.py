#!/usr/bin/env python3
"""Generate manuscript Figure 3 from local R0_v2 outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


NORD = {
    "night": "#2E3440",
    "slate": "#4C566A",
    "blue": "#5E81AC",
    "frost": "#88C0D0",
    "pale": "#D8DEE9",
    "green": "#A3BE8C",
    "red": "#BF616A",
    "orange": "#D08770",
    "white": "#FFFFFF",
}


DEFAULT_ANALYSIS_DIR = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "r0_v2_elasticnet_7rad_mpd_thickness"
)

DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "results_reference"
    / "manuscript_figures"
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "monospace",
            "font.monospace": ["Courier New", "DejaVu Sans Mono", "Menlo"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.edgecolor": NORD["slate"],
            "axes.linewidth": 1.6,
            "axes.labelcolor": NORD["night"],
            "axes.titlecolor": NORD["night"],
            "xtick.color": NORD["night"],
            "ytick.color": NORD["night"],
            "grid.color": NORD["slate"],
            "grid.alpha": 0.25,
            "figure.facecolor": NORD["white"],
            "axes.facecolor": NORD["white"],
        }
    )


def read_bootstrap_metrics(analysis_dir: Path) -> dict[str, float]:
    path = analysis_dir / "bootstrap_632plus_metrics.csv"
    if not path.exists():
        return {
            "auc": 0.787178,
            "ci_low": 0.685255,
            "ci_high": 0.854118,
        }

    df = pd.read_csv(path)
    row = df.loc[df["slug"].eq("radiomics_7rad")].iloc[0]
    return {
        "auc": float(row["auc_632plus"]),
        "ci_low": float(row["auc_632plus_ci_low"]),
        "ci_high": float(row["auc_632plus_ci_high"]),
    }


def read_apparent_roc(analysis_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    path = analysis_dir / "predictions_anonymized.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the R0_v2 analysis locally, or pass --analysis-dir."
        )

    df = pd.read_csv(path)
    rad = df.loc[df["slug"].eq("radiomics_7rad")].copy()
    y_true = rad["cr_popf"].astype(int).to_numpy()
    y_prob = rad["apparent_probability"].astype(float).to_numpy()
    fpr, tpr = roc_curve_numpy(y_true, y_prob)
    return fpr, tpr, float(np.trapz(tpr, fpr))


def roc_curve_numpy(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    score_sorted = y_score[order]

    distinct = np.where(np.diff(score_sorted))[0]
    threshold_idxs = np.r_[distinct, y_sorted.size - 1]
    tps = np.cumsum(y_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    positives = y_true.sum()
    negatives = y_true.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC requires at least one event and one nonevent.")

    tpr = np.r_[0.0, tps / positives]
    fpr = np.r_[0.0, fps / negatives]
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.r_[fpr, 1.0]
        tpr = np.r_[tpr, 1.0]
    return fpr, tpr


def validation_summary(primary_auc: dict[str, float]) -> list[dict[str, float | str]]:
    """Validation-method values for the retained EN model."""
    return [
        {
            "label": "Bootstrap .632+",
            "auc": primary_auc["auc"],
            "ci_low": primary_auc["ci_low"],
            "ci_high": primary_auc["ci_high"],
        },
        {
            "label": "Stratified CV\n(repeated)",
            "auc": 0.780093,
            "ci_low": 0.674861,
            "ci_high": 0.908403,
        },
        {
            "label": "Leave-One-Out",
            "auc": 0.781796,
            "ci_low": 0.697296,
            "ci_high": 0.859901,
        },
        {
            "label": "Simple\nBootstrap",
            "auc": 0.766477,
            "ci_low": 0.643912,
            "ci_high": 0.882369,
        },
    ]


def model_development_summary(primary_auc: dict[str, float]) -> list[tuple[str, float]]:
    return [
        ("EN", primary_auc["auc"]),
        ("LR", 0.778),
        ("SVM", 0.725),
        ("RF", 0.720),
        ("LGB", 0.687),
        ("XGB", 0.677),
    ]


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        color=NORD["night"],
        va="top",
        ha="left",
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color(NORD["slate"])
        spine.set_linewidth(1.6)
    ax.tick_params(labelsize=11, width=1.4, length=5)


def plot_model_heatmap(
    ax: plt.Axes, cax: plt.Axes, model_rows: Iterable[tuple[str, float]]
) -> None:
    rows = list(model_rows)
    labels = [r[0] for r in rows]
    values = np.array([r[1] for r in rows], dtype=float).reshape(-1, 1)

    im = ax.imshow(values, cmap="Blues", vmin=0.60, vmax=0.85, aspect="auto")
    ax.set_xticks([0])
    ax.set_xticklabels(["Bootstrap .632+"], fontsize=11)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_title("AUC by model (Bootstrap .632+)", fontsize=13, pad=12)
    ax.tick_params(axis="x", rotation=0)

    for i, value in enumerate(values[:, 0]):
        text_color = NORD["white"] if value >= 0.70 else NORD["night"]
        ax.text(
            0,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=12,
            color=text_color,
        )

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    add_panel_label(ax, "A")

    cb = plt.colorbar(im, cax=cax)
    cb.set_label("AUC", fontsize=11)
    cb.ax.tick_params(labelsize=10, width=1.2)
    cb.outline.set_linewidth(1.2)


def plot_validation_comparison(
    ax: plt.Axes, rows: list[dict[str, float | str]]
) -> None:
    x = np.arange(len(rows))
    aucs = np.array([float(r["auc"]) for r in rows])
    low = np.array([float(r["ci_low"]) for r in rows])
    high = np.array([float(r["ci_high"]) for r in rows])
    yerr = np.vstack([aucs - low, high - aucs])
    colors = [NORD["blue"], NORD["frost"], NORD["frost"], NORD["frost"]]

    ax.bar(
        x,
        aucs,
        yerr=yerr,
        capsize=4,
        color=colors,
        edgecolor=NORD["night"],
        linewidth=1.2,
        error_kw={"elinewidth": 1.4, "ecolor": NORD["night"]},
    )
    ax.axhline(0.5, color=NORD["red"], linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.75)
    ax.set_ylim(0.55, 0.95)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r["label"]) for r in rows], rotation=30, ha="right")
    ax.set_title("Validation method comparison (EN)", fontsize=13, pad=12)
    for xi, auc in zip(x, aucs):
        ax.text(
            xi,
            min(auc + 0.03, 0.935),
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NORD["night"],
        )
    style_axis(ax)
    add_panel_label(ax, "B")


def plot_roc(
    ax: plt.Axes,
    fpr: np.ndarray,
    tpr: np.ndarray,
    apparent_auc: float,
    primary_auc: dict[str, float],
) -> None:
    ax.plot([0, 1], [0, 1], color=NORD["red"], linestyle=(0, (5, 3)), linewidth=1.4, alpha=0.75)
    ax.step(
        fpr,
        tpr,
        where="post",
        color=NORD["blue"],
        linewidth=2.8,
        label=(
            "Bootstrap .632+ AUC\n"
            f"{primary_auc['auc']:.3f} "
            f"[{primary_auc['ci_low']:.3f}-{primary_auc['ci_high']:.3f}]"
        ),
    )
    ax.text(
        0.04,
        0.96,
        "Curve: apparent final fit",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color=NORD["slate"],
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC curve (EN, apparent fit)", fontsize=13, pad=12)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=1.0,
        facecolor=NORD["white"],
        edgecolor=NORD["pale"],
        fontsize=9,
    )
    style_axis(ax)
    add_panel_label(ax, "C")


def build_figure(analysis_dir: Path, output_dir: Path, stem: str) -> None:
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_auc = read_bootstrap_metrics(analysis_dir)
    fpr, tpr, apparent_auc = read_apparent_roc(analysis_dir)

    fig = plt.figure(figsize=(11.225, 7.937), dpi=100)
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 0.08, 1.38],
        height_ratios=[1.0, 1.0],
        wspace=0.42,
        hspace=0.58,
    )

    ax_a = fig.add_subplot(gs[:, 0])
    cax = fig.add_subplot(gs[:, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 2])

    plot_model_heatmap(ax_a, cax, model_development_summary(primary_auc))
    plot_validation_comparison(ax_b, validation_summary(primary_auc))
    plot_roc(ax_c, fpr, tpr, apparent_auc, primary_auc)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.10)
    svg_path = output_dir / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    strip_svg_trailing_whitespace(svg_path)
    plt.close(fig)
    print(f"Wrote {svg_path}")


def strip_svg_trailing_whitespace(svg_path: Path) -> None:
    text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(re.sub(r"[ \t]+\n", "\n", text), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--stem",
        default="figure3_model_development_internal_validation",
        help="Output filename stem for SVG.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_figure(args.analysis_dir, args.output_dir, args.stem)


if __name__ == "__main__":
    main()
