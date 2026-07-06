#!/usr/bin/env python3
"""Create comparative_risk_stratification_v2-style figures from clean OOF models.

This is the portable public-repo counterpart of the R0 regeneration wrapper. It
does not bundle any database. Provide local data paths at runtime, and the
script writes the same figure family produced by
``comparative_risk_stratification_v2.py`` while using unweighted fixed-L2
out-of-fold probabilities for radiomics/refit models.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_PATH = Path(__file__).resolve()
MODELS_DIR = SCRIPT_PATH.parent
CODE_DIR = MODELS_DIR.parent
REPO_DIR = CODE_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"

for path in (str(MODELS_DIR), str(CODE_DIR)):
    if path not in sys.path:
        sys.path.append(path)

import comparative_risk_stratification_v2 as base  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate v2-style figures with clean unweighted OOF probabilities."
    )
    parser.add_argument("--radiomics-path", type=Path, default=DATA_DIR / "HF3.csv")
    parser.add_argument(
        "--clinical-path",
        type=Path,
        default=DATA_DIR / "final_clinical_db.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "v2_style_figures_fixed_l2",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument(
        "--write-patient-predictions",
        action="store_true",
        help="Write patient-level OOF predictions locally. Keep disabled for public exports.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def build_unweighted_l2() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    class_weight=None,
                    penalty="l2",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base.RAD_PATH = args.radiomics_path
    base.TRUSTABLE_PATH = args.clinical_path
    records = base.load_data(max_cases=args.max_cases)
    base.compute_clinical_scores(records)
    if not records:
        raise RuntimeError("No records loaded. Check --radiomics-path and --clinical-path.")
    return records


def build_dataset(records: Sequence[Dict[str, Any]], indices: Sequence[int], features: Sequence[str]):
    eligible = [
        idx
        for idx in indices
        if base._is_valid(records[idx].get("cr_popf"))
        and all(base._is_valid(records[idx].get(feature)) for feature in features)
    ]
    x = np.asarray([[float(records[idx][feature]) for feature in features] for idx in eligible], dtype=float)
    y = np.asarray([int(records[idx]["cr_popf"]) for idx in eligible], dtype=int)
    patients = [records[idx]["scanner_patient_name"] for idx in eligible]
    return x, y, patients


def fit_oof_unweighted_l2(x: np.ndarray, y: np.ndarray, n_splits: int) -> Tuple[np.ndarray, np.ndarray]:
    min_class = int(np.bincount(y).min())
    folds = min(n_splits, min_class)
    if folds < 2:
        raise RuntimeError("Not enough events/non-events for stratified OOF evaluation.")
    pred = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y)):
        model = clone(build_unweighted_l2())
        model.fit(x[train_idx], y[train_idx])
        pred[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        fold_ids[test_idx] = fold
    if not np.all(np.isfinite(pred)):
        raise RuntimeError("OOF prediction generation failed.")
    return pred, fold_ids


def logit(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1 - eps)
    return np.log(prob / (1 - prob))


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -50, 50)))


def intercept_only_apply(prob: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    x_train = logit(prob[train])
    y_train = y[train].astype(int)
    intercept = 0.0
    for _ in range(100):
        mu = sigmoid(intercept + x_train)
        grad = float(np.sum(y_train - mu))
        hess = -float(np.sum(mu * (1 - mu)))
        if abs(hess) < 1e-12:
            break
        step = grad / hess
        intercept -= step
        if abs(step) < 1e-8:
            break
    return sigmoid(intercept + logit(prob[test]))


def calibrate_intercept_only_cv(prob: np.ndarray, y: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    calibrated = np.full_like(prob, np.nan, dtype=float)
    for fold in np.unique(fold_ids):
        test = fold_ids == fold
        train = ~test
        calibrated[test] = intercept_only_apply(prob, y, train, test)
    if not np.all(np.isfinite(calibrated)):
        raise RuntimeError("Intercept-only calibration failed.")
    return np.clip(calibrated, 0, 1)


def make_score_fold_ids(y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    fold_ids = np.full(len(y), -1, dtype=int)
    cv = StratifiedKFold(n_splits=min(n_splits, int(np.bincount(y).min())), shuffle=True, random_state=42)
    for fold, (_, test_idx) in enumerate(cv.split(np.zeros((len(y), 1)), y)):
        fold_ids[test_idx] = fold
    if np.any(fold_ids < 0):
        raise RuntimeError("Could not assign calibration folds.")
    return fold_ids


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wilson_ci(events: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = events / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def quantile_reliability_bins(prob: np.ndarray, y: np.ndarray, n_bins: int = 6) -> List[Dict[str, float]]:
    order = np.argsort(prob)
    chunks = np.array_split(order, n_bins)
    rows: List[Dict[str, float]] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        events = int(y[chunk].sum())
        n = int(len(chunk))
        low, high = wilson_ci(events, n)
        rows.append(
            {
                "n": n,
                "events": events,
                "mean_pred": float(prob[chunk].mean()),
                "event_rate": float(events / n),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def overwrite_performance_figure_direct_calibration(
    output_dir: Path,
    slug: str,
    name: str,
    y: np.ndarray,
    raw_prob: np.ndarray,
    calibrated_prob: np.ndarray,
    metrics: Dict[str, Any],
    *,
    probability_label: str,
) -> None:
    """Render the primary OOF reliability diagram.

    Recalibrated probabilities are evaluated upstream but are not plotted as a
    second curve because they did not improve probability estimates in the
    locked analysis.
    """
    raw_bins = quantile_reliability_bins(raw_prob, y, n_bins=6)
    fpr_raw, tpr_raw, _ = roc_curve(y, raw_prob)
    raw_color = base.NORD_COLORS["nord10"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].plot(fpr_raw, tpr_raw, color=raw_color, linewidth=2.4, label=f"Out-of-fold AUC = {metrics['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=base.NORD_COLORS["nord3"], alpha=0.6)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{name} ROC curve")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3, color=base.NORD_COLORS["nord3"])

    axes[1].plot([0, 1], [0, 1], linestyle="--", color=base.NORD_COLORS["nord3"], alpha=0.65, label="Ideal")
    xs = np.asarray([row["mean_pred"] for row in raw_bins], dtype=float)
    ys = np.asarray([row["event_rate"] for row in raw_bins], dtype=float)
    yerr = np.asarray(
        [
            [row["event_rate"] - row["ci_low"] for row in raw_bins],
            [row["ci_high"] - row["event_rate"] for row in raw_bins],
        ],
        dtype=float,
    )
    axes[1].plot(xs, ys, color=raw_color, linestyle="-", linewidth=1.9, alpha=0.9, label="Out-of-fold reliability curve")
    axes[1].errorbar(
        xs,
        ys,
        yerr=yerr,
        fmt="o",
        markersize=6,
        mfc="white",
        mec=raw_color,
        mew=1.4,
        ecolor=raw_color,
        elinewidth=1.0,
        capsize=2.5,
    )
    for row in raw_bins:
        axes[1].annotate(
            f"n={row['n']}",
            (row["mean_pred"], row["event_rate"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            color=base.NORD_COLORS["nord3"],
        )
    max_axis = max(0.5, min(1.0, float(np.quantile(raw_prob, 0.99)) + 0.08))
    max_y = max([row["event_rate"] for row in raw_bins])
    axes[1].set_xlim(-0.02, max_axis)
    axes[1].set_ylim(-0.02, max(0.65, min(1.0, float(max_y + 0.12))))
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed CR-POPF rate")
    axes[1].set_title(f"{name} out-of-fold calibration plot")
    axes[1].grid(True, alpha=0.3, color=base.NORD_COLORS["nord3"])
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].text(
        0.02,
        0.02,
        "\n".join(
            [
                f"AUC={metrics['auc']:.3f}",
                f"Brier={metrics['brier']:.3f}",
                f"ECE={metrics['ece']:.3f}",
                "Recalibration evaluated, not retained",
            ]
        ),
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=base.NORD_COLORS["nord3"], alpha=0.9),
    )

    axes[2].hist(raw_prob, bins=20, color=raw_color, alpha=0.55, label="Out-of-fold probabilities")
    axes[2].axvline(float(y.mean()), color=base.NORD_COLORS["nord3"], linestyle="--", linewidth=1.2, label="Observed prevalence")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Patients")
    axes[2].set_title(f"{name} out-of-fold probability distribution")
    axes[2].grid(True, alpha=0.3, color=base.NORD_COLORS["nord3"])
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    base.save_beautiful_figure(fig, output_dir / f"{slug}_performance")
    plt.close(fig)


def logistic_specs(records: Sequence[Dict[str, Any]], common_indices: Sequence[int]):
    stabl = list(base.STABL_FEATURES)
    all_indices = list(range(len(records)))
    clinical_features = [
        "mpd_diameter",
        "neck_thickness",
        "bmi",
        "blood_loss",
        "op_duration",
        "soft_pancreas_flag",
        "diabetes_numeric",
        "transection_at_neck",
    ]
    return [
        ("Radiomics signature", "radiomics", stabl, common_indices),
        ("Clinical refit", "clinical_refit", clinical_features, all_indices),
        ("Radiomics + clinical refit", "radiomics_clinical_refit", stabl + clinical_features, all_indices),
        ("Radiomics + D-FRS pre-operative", "radiomics_dfrs_preop", stabl + ["dfrs_preop_logit"], common_indices),
        ("Radiomics + D-FRS intra-operative", "radiomics_dfrs_intra", stabl + ["dfrs_intraop_logit"], common_indices),
        ("Radiomics + DISPAIR", "radiomics_dispair", stabl + ["dispair_legacy_logit"], common_indices),
        ("Radiomics + DISPAIR refined", "radiomics_dispair_refined", stabl + ["dispair_refined_logit"], common_indices),
        (
            "Radiomics + all scores",
            "radiomics_all_scores",
            stabl + ["dfrs_preop_logit", "dfrs_intraop_logit", "dispair_legacy_logit"],
            common_indices,
        ),
    ]


def score_specs():
    return [
        ("D-FRS pre-operative", "dfrs_preop", "dfrs_preop_prob"),
        ("D-FRS intra-operative", "dfrs_intra", "dfrs_intraop_prob"),
        ("DISPAIR-FRS", "dispair", "dispair_legacy_prob"),
        ("DISPAIR-FRS refined", "dispair_refined", "dispair_refined_prob"),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)

    common_indices = [
        idx
        for idx, rec in enumerate(records)
        if all(
            base._is_valid(rec.get(key))
            for key in ("dfrs_preop_prob", "dfrs_intraop_prob", "dispair_legacy_prob")
        )
        and all(base._is_valid(rec.get(feature)) for feature in base.STABL_FEATURES)
    ]

    metrics_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    for name, slug, features, indices in logistic_specs(records, common_indices):
        x, y, patients = build_dataset(records, indices, features)
        if len(y) < 10 or np.unique(y).size < 2:
            print(f"[skip] {name}: insufficient data")
            continue
        prob, fold_ids = fit_oof_unweighted_l2(x, y, args.outer_folds)
        calibrated_prob = calibrate_intercept_only_cv(prob, y, fold_ids)
        raw_metrics, _ = base.evaluate_predictions(
            f"{name} (identity raw)",
            f"{slug}_raw_identity_tmp",
            y,
            prob,
            args.output_dir,
            make_plots=False,
        )
        metrics, risk = base.evaluate_predictions(
            name,
            slug,
            y,
            prob,
            args.output_dir,
            prob_raw=prob,
            metrics_raw=raw_metrics,
            prob_for_auc=prob,
        )
        brier = float(brier_score_loss(y, prob))
        metrics.update(
            {
                "calibration_method": "identity",
                "calibration": "identity",
                "calibration_brier": brier,
                "brier_raw": brier,
                "calibration_brier_delta": 0.0,
                "raw_calibration_slope": raw_metrics.get("calibration_slope"),
                "raw_calibration_intercept": raw_metrics.get("calibration_intercept"),
                "raw_ece": raw_metrics.get("ece"),
                "calibration_brier_full": brier,
                "probability_model": "fixed_unweighted_l2_C1",
            }
        )
        metrics_rows.append(metrics)
        risk_rows.extend(risk)
        overwrite_performance_figure_direct_calibration(
            args.output_dir,
            slug,
            name,
            y,
            prob,
            calibrated_prob,
            metrics,
            probability_label="CV intercept-only calibrated bins",
        )
        (args.output_dir / f"{slug}_calibration.json").write_text(
            json.dumps(
                {
                    "method": "identity",
                    "probability_model": "fixed_unweighted_l2_C1",
                    "brier": brier,
                    "auc": metrics.get("auc"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        for row_idx, (patient, label, probability, fold_id) in enumerate(zip(patients, y, prob, fold_ids)):
            prediction_rows.append(
                {
                    "model": name,
                    "slug": slug,
                    "patient": patient,
                    "cr_popf": int(label),
                    "probability": float(probability),
                    "calibrated_probability": float(calibrated_prob[row_idx]),
                    "fold_id": int(fold_id),
                    "probability_model": "fixed_unweighted_l2_C1",
                }
            )

    for name, slug, prob_key in score_specs():
        values = [
            (records[idx]["scanner_patient_name"], int(records[idx]["cr_popf"]), float(records[idx][prob_key]))
            for idx in range(len(records))
            if base._is_valid(records[idx].get("cr_popf")) and base._is_valid(records[idx].get(prob_key))
        ]
        if len(values) < 10:
            print(f"[skip] {name}: insufficient data")
            continue
        y = np.asarray([row[1] for row in values], dtype=int)
        prob = np.asarray([row[2] for row in values], dtype=float)
        score_fold_ids = make_score_fold_ids(y)
        calibrated_prob = calibrate_intercept_only_cv(prob, y, score_fold_ids)
        metrics, risk = base.evaluate_predictions(name, slug, y, prob, args.output_dir)
        brier = float(brier_score_loss(y, prob))
        metrics.update(
            {
                "calibration_method": "precomputed",
                "calibration": "precomputed",
                "calibration_brier": brier,
                "brier_raw": brier,
                "calibration_brier_delta": 0.0,
                "calibration_brier_full": brier,
                "auc_calibrated": metrics["auc"],
                "auc_calibrated_ci_low": metrics["auc_ci_low"],
                "auc_calibrated_ci_high": metrics["auc_ci_high"],
                "probability_model": "published_score_no_refit",
            }
        )
        metrics_rows.append(metrics)
        risk_rows.extend(risk)
        overwrite_performance_figure_direct_calibration(
            args.output_dir,
            slug,
            name,
            y,
            prob,
            calibrated_prob,
            metrics,
            probability_label="CV intercept-only calibrated bins",
        )

    write_csv(args.output_dir / "model_metrics.csv", metrics_rows)
    write_csv(args.output_dir / "risk_group_summary.csv", risk_rows)
    if args.write_patient_predictions:
        write_csv(args.output_dir / "oof_predictions_v2_style.csv", prediction_rows)
    (args.output_dir / "model_metrics.json").write_text(
        json.dumps({"models": metrics_rows}, indent=2),
        encoding="utf-8",
    )

    base.plot_risk_group_comparison(risk_rows, args.output_dir / "risk_group_event_rates")
    base.plot_risk_group_comparison(
        risk_rows,
        args.output_dir / "risk_group_event_rates_radiomics_vs_dfrs",
        models_filter=["Radiomics signature", "Radiomics + D-FRS pre-operative"],
    )

    valid_auc = [row for row in metrics_rows if np.isfinite(row.get("auc", np.nan))]
    ordered = sorted(valid_auc, key=lambda row: row["auc"], reverse=True)
    fig, _ = base.plot_model_comparison_with_ci(
        [row["model"] for row in ordered],
        [row["auc"] for row in ordered],
        [(row["auc_ci_low"], row["auc_ci_high"]) for row in ordered],
        title="Comparative AUROC (OOF predictions)",
    )
    base.save_beautiful_figure(fig, args.output_dir / "auc_comparison")
    base.plt.close(fig)

    print(f"V2-style clean figures written to: {args.output_dir}")


if __name__ == "__main__":
    main()
