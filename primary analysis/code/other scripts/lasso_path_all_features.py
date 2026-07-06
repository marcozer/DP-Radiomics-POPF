#!/usr/bin/env python3
"""LASSO path plotting for a complete V3 STABL run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import re
import unicodedata

from sklearn.linear_model import lasso_path
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.plotting_utils import (  # noqa: E402
    create_beautiful_figure,
    save_beautiful_figure,
    setup_plotting,
    NORD_COLORS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot LASSO path with all features and highlight STABL selections")
    parser.add_argument("--results-dir", required=True, help="Path to V3 results directory")
    parser.add_argument("--output-dir", default=None, help="Optional output directory (defaults to <results-dir>/plots)")
    parser.add_argument("--top-k", type=int, default=None, help="Highlight top-K features by STABL frequency")
    parser.add_argument("--alphas", type=float, nargs="*", default=None, help="Custom lambda grid for the path")
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    results_path = results_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {results_dir}")
    with open(results_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_radiomics_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Radiomics file not found: {path}")
    return pd.read_csv(path)


HONORIFICS = {"mr", "mrs", "ms", "mme", "mlle", "mle", "dr", "prof", "monsieur", "madame", "m", "mme.", "mlle.", "mr.", "mrs.", "dr.", "prof."}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _canonicalize(text: str) -> str:
    text = "" if text is None else str(text)
    text = _strip_accents(text).lower()
    text = text.replace("-", "_").replace(" ", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    tokens = [tok for tok in text.split("_") if tok and tok not in HONORIFICS]
    return re.sub(r"_+", "_", "_".join(tokens)).strip("_")


def ensure_labels(df: pd.DataFrame, run_args: dict, results_dir: Path) -> pd.DataFrame:
    if "cr_popf" in df.columns:
        return df

    matches_path = run_args.get("matches_path") or "data/outcome_matches.csv"
    matches_path = Path(matches_path)
    if not matches_path.is_absolute():
        candidate = (results_dir / matches_path).resolve()
        matches_path = candidate if candidate.exists() else (Path.cwd() / matches_path).resolve()

    if not matches_path.exists():
        raise FileNotFoundError(f"Matches file not found: {matches_path}")

    matches_df = pd.read_csv(matches_path)

    def pick_id(frame: pd.DataFrame) -> str:
        for col in ("scanner_patient_name", "patient_name", "patient_id"):
            if col in frame.columns:
                return col
        raise ValueError("No identifier column found (expected scanner_patient_name/patient_name/patient_id)")

    df_id = pick_id(df)
    matches_id = pick_id(matches_df)
    if matches_id != "scanner_patient_name":
        matches_df = matches_df.rename(columns={matches_id: "scanner_patient_name"})

    merged = df.merge(matches_df[["scanner_patient_name", "popf_grade"]], left_on=df_id, right_on="scanner_patient_name", how="left")

    if merged["popf_grade"].isna().any() and run_args.get("allow_id_normalization"):
        merged["_canon_left"] = merged[df_id].map(_canonicalize)
        matches_df["_canon_right"] = matches_df["scanner_patient_name"].map(_canonicalize)
        matches_canon = matches_df.drop_duplicates("_canon_right")[["_canon_right", "popf_grade"]]
        merged = merged.merge(matches_canon, left_on="_canon_left", right_on="_canon_right", how="left", suffixes=("", "_canon"))
        merged["popf_grade"] = merged["popf_grade"].fillna(merged["popf_grade_canon"])
        merged = merged.drop(columns=["_canon_left", "_canon_right", "popf_grade_canon"], errors="ignore")

    merged = merged.drop(columns=["scanner_patient_name"], errors="ignore")
    merged = merged.dropna(subset=["popf_grade"]).copy()
    if merged.empty:
        raise ValueError("Unable to align radiomics rows with POPF outcomes; check matches file")

    positive_grades = run_args.get("positive_grades", "B,C")
    positives = {g.strip().upper() for g in str(positive_grades).split(",") if g.strip()}
    merged["cr_popf"] = merged["popf_grade"].apply(lambda g: 1 if str(g).upper() in positives else 0)
    merged = merged.drop(columns=["popf_grade"])
    return merged


def build_design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    exclude = {"cr_popf", "popf_grade", "scanner_patient_name", "patient_name", "patient_id"}
    candidate_cols = [col for col in df.columns if col not in exclude]
    numeric_df = df[candidate_cols].apply(pd.to_numeric, errors="coerce")
    numeric_cols = [col for col in numeric_df.columns if numeric_df[col].notna().sum() > 0]
    if not numeric_cols:
        raise ValueError("No numeric radiomics features available for LASSO path plotting.")

    numeric_df = numeric_df[numeric_cols].fillna(numeric_df[numeric_cols].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df.values)
    y = df["cr_popf"].astype(int).values
    return X_scaled, y, numeric_cols


def plot_lasso_path(
    alphas: np.ndarray,
    coefs: np.ndarray,
    feature_names: Sequence[str],
    highlight_features: Sequence[str],
    freq_dict: dict[str, float],
    output_path: Path,
) -> None:
    setup_plotting()
    fig, ax = create_beautiful_figure("wide")

    highlight_list = list(dict.fromkeys(highlight_features))
    highlight_set = set(highlight_list)
    palette = [
        NORD_COLORS["nord8"],
        NORD_COLORS["nord9"],
        NORD_COLORS["nord10"],
        NORD_COLORS["nord11"],
        NORD_COLORS["nord12"],
        NORD_COLORS["nord13"],
    ]
    grey_color = "#C7CDD6"

    for idx, feat in enumerate(feature_names):
        if feat in highlight_set:
            continue
        ax.plot(alphas, coefs[idx, :], color=grey_color, linewidth=0.6, alpha=0.35, zorder=1)

    for order, feat in enumerate(highlight_list):
        feature_idx = feature_names.index(feat)
        color = palette[order % len(palette)]
        coef_line = coefs[feature_idx, :]
        ax.plot(alphas, coef_line, color=color, linewidth=3.2, alpha=0.95, label=f"{feat} (freq {freq_dict.get(feat, 0.0):.2f})", zorder=3)
        ax.text(
            alphas[-1],
            coef_line[-1],
            feat,
            color=color,
            fontsize=12,
            ha="left",
            va="center",
            clip_on=True,
        )

    highlight_idx = [feature_names.index(feat) for feat in highlight_list]
    highlight_max = np.max(np.abs(coefs[highlight_idx, :])) if highlight_idx else np.max(np.abs(coefs))
    max_abs = highlight_max if highlight_max > 0 else np.max(np.abs(coefs))
    if max_abs > 0:
        ax.set_ylim(-1.1 * max_abs, 1.1 * max_abs)
    ax.set_xscale("log")
    ax.set_xlim(alphas.min(), alphas.max())
    ax.margins(x=0.02)
    ax.set_xlabel("Regularisation parameter λ")
    ax.set_ylabel("Coefficient")
    ax.set_title("LASSO coefficient paths (STABL panel highlighted)")
    ax.axhline(0.0, color="#5B6573", linewidth=1.2, linestyle="--", alpha=0.6, zorder=2)
    tick_levels = np.logspace(np.floor(np.log10(alphas.min())), np.ceil(np.log10(alphas.max())), num=8)
    for level in tick_levels:
        if alphas.min() <= level <= alphas.max():
            ax.axvline(level, color="#E5E9F0", linewidth=0.5, linestyle="-", alpha=0.4, zorder=0)
    if highlight_list:
        ax.legend(loc="best", fontsize=12)
    ax.grid(True, which="both", alpha=0.25, linewidth=0.6)

    save_beautiful_figure(fig, output_path)
    print(f"Saved LASSO path plot to {output_path.with_suffix('.png')}")


def save_coefficient_table(
    alphas: np.ndarray,
    coefs: np.ndarray,
    feature_names: Sequence[str],
    highlight_features: Sequence[str],
    output_path: Path,
) -> None:
    highlight_list = list(dict.fromkeys(highlight_features))
    if highlight_list:
        indices = [feature_names.index(feat) for feat in highlight_list]
        table = pd.DataFrame(coefs[indices, :].T, columns=highlight_list)
    else:
        table = pd.DataFrame(coefs.T, columns=feature_names)
    table.insert(0, "lambda", alphas)
    table.to_csv(output_path, index=False)
    print(f"Saved coefficient evolution table to {output_path}")


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    selected = results.get("selected_features") or []
    freq_dict = results.get("feature_frequencies", {})

    run_args = results.get("args", {})
    radiomics_path = run_args.get("radiomics_path") or "data/pancreatic_head_radiomics.csv"
    radiomics_path = Path(radiomics_path)
    if not radiomics_path.is_absolute():
        candidate = (results_dir / radiomics_path).resolve()
        radiomics_path = candidate if candidate.exists() else (Path.cwd() / radiomics_path).resolve()

    df = load_radiomics_dataframe(radiomics_path)
    df = ensure_labels(df, run_args, results_dir)
    X_scaled, y, all_features = build_design_matrix(df)

    missing_selected = sorted(set(selected) - set(all_features))
    if missing_selected:
        print(f"Warning: {len(missing_selected)} selected features missing/non-numeric: {missing_selected[:10]}" + ("…" if len(missing_selected) > 10 else ""))

    highlight = [feat for feat in selected if feat in all_features]
    if args.top_k and highlight:
        highlight = sorted(highlight, key=lambda f: freq_dict.get(f, 0.0), reverse=True)[: args.top_k]
    if not highlight:
        highlight = all_features[: min(5, len(all_features))]

    alphas = np.array(args.alphas) if args.alphas else np.logspace(-4, 1, 200)
    alphas_path, coefs_full, _ = lasso_path(X_scaled, y, alphas=alphas)
    order = np.argsort(alphas_path)
    alphas_plot = alphas_path[order]
    coefs_plot = coefs_full[:, order]

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_lasso_path(alphas_plot, coefs_plot, all_features, highlight, freq_dict, output_dir / "lasso_path_all")
    save_coefficient_table(alphas_plot, coefs_plot, all_features, highlight, output_dir / "lasso_path_all_coefficients.csv")


if __name__ == "__main__":
    main()
