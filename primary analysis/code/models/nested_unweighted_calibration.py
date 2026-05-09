#!/usr/bin/env python3
"""Clean nested probability audit for the DP radiomics POPF models.

This script is intentionally separate from the deployment inference code. It
recomputes manuscript-facing out-of-fold estimates with unweighted logistic
regression, checks calibration, and optionally runs a nested Optuna sensitivity
analysis. It never stores patient data in the repository; data paths are passed
at runtime and outputs are written under ``primary analysis/results`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
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


DEFAULT_RAD_PATH = DATA_DIR / "HF3.csv"
DEFAULT_CLINICAL_PATH = DATA_DIR / "POPF_SCANNER_complete_clinical_db_filled.csv"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "nested_unweighted_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean OOF calibration audit with unweighted logistic regression."
    )
    parser.add_argument("--radiomics-path", type=Path, default=DEFAULT_RAD_PATH)
    parser.add_argument("--clinical-path", type=Path, default=DEFAULT_CLINICAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--optuna-trials", type=int, default=150)
    parser.add_argument("--optuna-timeout", type=int, default=None)
    parser.add_argument("--include-nested-optuna", action="store_true")
    parser.add_argument(
        "--write-patient-predictions",
        action="store_true",
        help="Write patient-level OOF predictions locally. Keep disabled for public exports.",
    )
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def _is_valid(value: Any) -> bool:
    return base._is_valid(value)


def _finite(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _slug(text: str) -> str:
    return (
        text.lower()
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_ready(v) for v in value]
    if isinstance(value, tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    base.RAD_PATH = args.radiomics_path
    base.TRUSTABLE_PATH = args.clinical_path
    records = base.load_data(max_cases=args.max_cases)
    base.compute_clinical_scores(records)
    if not records:
        raise RuntimeError("No analyzable records were loaded.")
    return records


def build_pipeline(
    *,
    penalty: str = "l2",
    c_value: float = 1.0,
    l1_ratio: Optional[float] = None,
) -> Pipeline:
    kwargs: Dict[str, Any] = {
        "C": c_value,
        "class_weight": None,
        "max_iter": 5000,
        "random_state": 42,
    }
    if penalty == "elasticnet":
        kwargs.update({"penalty": "elasticnet", "solver": "saga", "l1_ratio": l1_ratio})
    elif penalty == "l2":
        kwargs.update({"penalty": "l2", "solver": "lbfgs"})
    else:
        raise ValueError(f"Unsupported penalty: {penalty}")

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(**kwargs)),
        ]
    )


def eligible_indices(
    records: Sequence[Dict[str, Any]],
    features: Sequence[str],
    *,
    subset: Optional[Sequence[int]] = None,
) -> List[int]:
    candidates = subset if subset is not None else range(len(records))
    return [
        idx
        for idx in candidates
        if _is_valid(records[idx].get("cr_popf"))
        and all(_is_valid(records[idx].get(feature)) for feature in features)
    ]


def tune_inner_optuna(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    inner_folds: int,
    trials: int,
    timeout: Optional[int],
) -> Dict[str, float]:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is required for --include-nested-optuna") from exc

    n_splits = min(inner_folds, int(np.bincount(y_train).min()))
    if n_splits < 2:
        return {"C": 1.0, "l1_ratio": 0.0}

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    def objective(trial: "optuna.Trial") -> float:
        c_value = trial.suggest_float("C", 1e-3, 1e2, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 0.8)
        model = build_pipeline(penalty="elasticnet", c_value=c_value, l1_ratio=l1_ratio)
        scores: List[float] = []
        for train_idx, val_idx in inner_cv.split(x_train, y_train):
            fold_model = clone(model)
            fold_model.fit(x_train[train_idx], y_train[train_idx])
            prob = fold_model.predict_proba(x_train[val_idx])[:, 1]
            if np.unique(y_train[val_idx]).size == 2:
                scores.append(float(roc_auc_score(y_train[val_idx], prob)))
        if not scores:
            raise optuna.TrialPruned()
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)
    return {
        "C": float(study.best_params.get("C", 1.0)),
        "l1_ratio": float(study.best_params.get("l1_ratio", 0.0)),
        "inner_auc": float(study.best_value),
    }


def fit_oof(
    records: Sequence[Dict[str, Any]],
    features: Sequence[str],
    *,
    subset: Optional[Sequence[int]],
    outer_folds: int,
    nested_optuna: bool,
    inner_folds: int,
    optuna_trials: int,
    optuna_timeout: Optional[int],
) -> Optional[Dict[str, Any]]:
    indices = eligible_indices(records, features, subset=subset)
    if len(indices) < 10:
        return None

    x = np.asarray([[float(records[idx][feature]) for feature in features] for idx in indices])
    y = np.asarray([int(records[idx]["cr_popf"]) for idx in indices], dtype=int)
    if np.unique(y).size < 2:
        return None

    min_class = int(np.bincount(y).min())
    n_splits = min(outer_folds, min_class)
    if n_splits < 2:
        return None

    oof = np.full(len(indices), np.nan, dtype=float)
    fold_ids = np.full(len(indices), -1, dtype=int)
    fold_details: List[Dict[str, Any]] = []
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(x, y), start=1):
        if nested_optuna:
            params = tune_inner_optuna(
                x[train_idx],
                y[train_idx],
                inner_folds=inner_folds,
                trials=optuna_trials,
                timeout=optuna_timeout,
            )
            model = build_pipeline(
                penalty="elasticnet",
                c_value=float(params["C"]),
                l1_ratio=float(params["l1_ratio"]),
            )
        else:
            params = {"C": 1.0, "l1_ratio": None, "inner_auc": None}
            model = build_pipeline(penalty="l2", c_value=1.0)

        model.fit(x[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(x[test_idx])[:, 1]
        fold_ids[test_idx] = fold
        fold_details.append(
            {
                "fold": fold,
                "train_n": int(train_idx.size),
                "train_events": int(y[train_idx].sum()),
                "test_n": int(test_idx.size),
                "test_events": int(y[test_idx].sum()),
                "C": _finite(params.get("C")),
                "l1_ratio": _finite(params.get("l1_ratio")),
                "inner_auc": _finite(params.get("inner_auc")),
            }
        )

    if not np.all(np.isfinite(oof)):
        raise RuntimeError("OOF prediction generation failed for at least one sample.")

    return {
        "indices": indices,
        "patient_ids": [records[idx]["scanner_patient_name"] for idx in indices],
        "y": y,
        "prob": oof,
        "fold_ids": fold_ids,
        "fold_details": fold_details,
    }


def calibration_error_summary(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    max_error = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        err = abs(float(y_true[mask].mean()) - float(prob[mask].mean()))
        ece += float(mask.mean()) * err
        max_error = max(max_error, err)
    return float(ece), float(max_error)


def metrics_for(y_true: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    auc = float(roc_auc_score(y_true, prob))
    ci_low, ci_high = base.bootstrap_auc_ci(y_true, prob, n_boot=2000, seed=42)
    ece, max_error = calibration_error_summary(y_true, prob)
    intercept, slope = base.calibration_intercept_slope(y_true, prob)
    return {
        "n": int(y_true.size),
        "events": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auc": auc,
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "brier": float(brier_score_loss(y_true, prob)),
        "ece": ece,
        "max_calibration_error": max_error,
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "mean_predicted_risk": float(prob.mean()),
        "median_predicted_risk": float(np.median(prob)),
        "q05_predicted_risk": float(np.quantile(prob, 0.05)),
        "q95_predicted_risk": float(np.quantile(prob, 0.95)),
    }


def score_model(
    records: Sequence[Dict[str, Any]],
    name: str,
    prob_key: str,
) -> Optional[Dict[str, Any]]:
    values = [
        (idx, int(records[idx]["cr_popf"]), float(records[idx][prob_key]))
        for idx in range(len(records))
        if _is_valid(records[idx].get("cr_popf")) and _is_valid(records[idx].get(prob_key))
    ]
    if len(values) < 10:
        return None
    y = np.asarray([row[1] for row in values], dtype=int)
    prob = np.asarray([row[2] for row in values], dtype=float)
    if np.unique(y).size < 2:
        return None
    return {
        "name": name,
        "slug": _slug(name),
        "patient_ids": [records[row[0]]["scanner_patient_name"] for row in values],
        "y": y,
        "prob": prob,
        "metrics": metrics_for(y, prob),
        "kind": "published_score_no_refit",
    }


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_ready(row.get(key)) for key in fieldnames})


def write_oof_predictions(path: Path, models: Sequence[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []
    for model in models:
        for pid, label, prob in zip(model["patient_ids"], model["y"], model["prob"]):
            rows.append(
                {
                    "model": model["name"],
                    "slug": model["slug"],
                    "scanner_patient_name": pid,
                    "cr_popf": int(label),
                    "probability": float(prob),
                    "kind": model["kind"],
                }
            )
    write_csv(path, rows)


def paired_delong_rows(models: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_slug = {model["slug"]: model for model in models}
    if "radiomics_signature" not in by_slug:
        return []
    reference = by_slug["radiomics_signature"]
    ref_map = {
        pid: (int(label), float(prob))
        for pid, label, prob in zip(reference["patient_ids"], reference["y"], reference["prob"])
    }

    rows: List[Dict[str, Any]] = []
    for model in models:
        if model["slug"] == "radiomics_signature":
            continue
        candidate_map = {
            pid: (int(label), float(prob))
            for pid, label, prob in zip(model["patient_ids"], model["y"], model["prob"])
        }
        common_ids = [pid for pid in reference["patient_ids"] if pid in candidate_map]
        labels: List[int] = []
        ref_prob: List[float] = []
        cand_prob: List[float] = []
        for pid in common_ids:
            ref_label, rp = ref_map[pid]
            cand_label, cp = candidate_map[pid]
            if ref_label != cand_label:
                continue
            labels.append(ref_label)
            ref_prob.append(rp)
            cand_prob.append(cp)
        if len(set(labels)) != 2:
            continue
        result = base.delong_test(
            np.asarray(labels, dtype=float),
            np.asarray(ref_prob, dtype=float),
            np.asarray(cand_prob, dtype=float),
        )
        rows.append(
            {
                "reference_model": reference["name"],
                "comparison_model": model["name"],
                "paired_n": len(labels),
                "paired_events": int(sum(labels)),
                "auc_reference": float(result["auc1"]),
                "auc_comparison": float(result["auc2"]),
                "auc_delta_reference_minus_comparison": float(result["auc1"] - result["auc2"]),
                "p_value": float(result["p_value"]),
            }
        )
    return rows


def plot_probability_audit(model: Dict[str, Any], output_dir: Path) -> None:
    y = np.asarray(model["y"], dtype=int)
    prob = np.asarray(model["prob"], dtype=float)
    metrics = model["metrics"]
    fpr, tpr, _ = roc_curve(y, prob)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.2)
    axes[0].plot([0, 1], [0, 1], color="#666666", ls="--", lw=1)
    axes[0].set_title(f"ROC, AUC {metrics['auc']:.3f}")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")

    bins = np.linspace(0.0, 1.0, 7)
    bin_ids = np.clip(np.digitize(prob, bins) - 1, 0, len(bins) - 2)
    mean_pred: List[float] = []
    event_rate: List[float] = []
    sizes: List[int] = []
    for bin_id in range(len(bins) - 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        mean_pred.append(float(prob[mask].mean()))
        event_rate.append(float(y[mask].mean()))
        sizes.append(int(mask.sum()))
    axes[1].plot([0, 1], [0, 1], color="#666666", ls="--", lw=1)
    axes[1].scatter(mean_pred, event_rate, s=np.asarray(sizes) * 7, color="#d62728", alpha=0.75)
    axes[1].set_title(f"Calibration, Brier {metrics['brier']:.3f}")
    axes[1].set_xlabel("Mean predicted risk")
    axes[1].set_ylabel("Observed event rate")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    axes[2].hist(prob[y == 0], bins=20, alpha=0.65, label="No CR-POPF", color="#9ecae1")
    axes[2].hist(prob[y == 1], bins=20, alpha=0.65, label="CR-POPF", color="#e7969c")
    axes[2].axvline(y.mean(), color="#333333", ls="--", lw=1, label="Observed prevalence")
    axes[2].set_title("OOF probability distribution")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Patients")
    axes[2].legend(fontsize=8)

    fig.suptitle(model["name"])
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"{model['slug']}_probability_calibration_audit.{ext}", dpi=300)
    plt.close(fig)


def model_specs() -> List[Dict[str, Any]]:
    stabl = list(base.STABL_FEATURES)
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
        {"name": "Radiomics signature", "features": stabl},
        {"name": "Radiomics + D-FRS pre-operative", "features": stabl + ["dfrs_preop_logit"]},
        {"name": "Radiomics + D-FRS intra-operative", "features": stabl + ["dfrs_intraop_logit"]},
        {"name": "Radiomics + DISPAIR", "features": stabl + ["dispair_legacy_logit"]},
        {
            "name": "Radiomics + all scores",
            "features": stabl + ["dfrs_preop_logit", "dfrs_intraop_logit", "dispair_legacy_logit"],
        },
        {"name": "Clinical refit", "features": clinical_features, "subset": None},
        {"name": "Radiomics + clinical refit", "features": stabl + clinical_features, "subset": None},
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(args)
    common_score_indices = [
        idx
        for idx, rec in enumerate(records)
        if all(_is_valid(rec.get(feature)) for feature in base.STABL_FEATURES)
        and all(
            _is_valid(rec.get(score))
            for score in ("dfrs_preop_logit", "dfrs_intraop_logit", "dispair_legacy_logit")
        )
    ]

    availability = {
        "total_loaded": len(records),
        "events_loaded": int(sum(int(row["cr_popf"]) for row in records)),
        "radiomics_feature_complete": len(eligible_indices(records, base.STABL_FEATURES)),
        "common_radiomics_score_complete": len(common_score_indices),
        "dfrs_preop_available": sum(_is_valid(row.get("dfrs_preop_prob")) for row in records),
        "dfrs_intraop_available": sum(_is_valid(row.get("dfrs_intraop_prob")) for row in records),
        "dispair_available": sum(_is_valid(row.get("dispair_legacy_prob")) for row in records),
    }

    models: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []

    for spec in model_specs():
        subset = spec.get("subset", common_score_indices)
        result = fit_oof(
            records,
            spec["features"],
            subset=subset,
            outer_folds=args.outer_folds,
            nested_optuna=False,
            inner_folds=args.inner_folds,
            optuna_trials=args.optuna_trials,
            optuna_timeout=args.optuna_timeout,
        )
        if result is None:
            continue
        model = {
            "name": spec["name"],
            "slug": _slug(spec["name"]),
            "patient_ids": result["patient_ids"],
            "y": result["y"],
            "prob": result["prob"],
            "metrics": metrics_for(result["y"], result["prob"]),
            "kind": "fixed_unweighted_l2_oof",
            "features": spec["features"],
        }
        models.append(model)
        for row in result["fold_details"]:
            fold_rows.append({"model": spec["name"], "kind": model["kind"], **row})

    if args.include_nested_optuna:
        for spec in model_specs():
            if spec.get("subset") is None:
                continue
            result = fit_oof(
                records,
                spec["features"],
                subset=common_score_indices,
                outer_folds=args.outer_folds,
                nested_optuna=True,
                inner_folds=args.inner_folds,
                optuna_trials=args.optuna_trials,
                optuna_timeout=args.optuna_timeout,
            )
            if result is None:
                continue
            model = {
                "name": f"{spec['name']} (nested Optuna sensitivity)",
                "slug": f"{_slug(spec['name'])}_nested_optuna",
                "patient_ids": result["patient_ids"],
                "y": result["y"],
                "prob": result["prob"],
                "metrics": metrics_for(result["y"], result["prob"]),
                "kind": "nested_unweighted_elasticnet_sensitivity",
                "features": spec["features"],
            }
            models.append(model)
            for row in result["fold_details"]:
                fold_rows.append({"model": spec["name"], "kind": model["kind"], **row})

    for name, key in (
        ("D-FRS pre-operative", "dfrs_preop_prob"),
        ("D-FRS intra-operative", "dfrs_intraop_prob"),
        ("DISPAIR", "dispair_legacy_prob"),
    ):
        score = score_model(records, name, key)
        if score is not None:
            models.append(score)

    if not models:
        raise RuntimeError("No model produced evaluable OOF predictions.")

    metrics_rows = [
        {
            "model": model["name"],
            "slug": model["slug"],
            "kind": model["kind"],
            **model["metrics"],
        }
        for model in models
    ]
    write_csv(args.output_dir / "model_metrics_selected.csv", metrics_rows)
    write_csv(args.output_dir / "fold_details.csv", fold_rows)
    if args.write_patient_predictions:
        write_oof_predictions(args.output_dir / "oof_predictions.csv", models)
    delong_rows = paired_delong_rows(models)
    write_csv(args.output_dir / "delong_comparisons.csv", delong_rows)

    for slug in ("radiomics_signature", "radiomics_plus_d_frs_pre_operative"):
        model = next((item for item in models if item["slug"] == slug), None)
        if model is not None:
            plot_probability_audit(model, args.output_dir)

    report = {
        "inputs": {
            "radiomics_path": str(args.radiomics_path),
            "clinical_path": str(args.clinical_path),
        },
        "availability": availability,
        "primary_interpretation": (
            "Use fixed_unweighted_l2_oof as the primary probability estimate. "
            "Nested Optuna, when enabled, is a sensitivity analysis only."
        ),
        "models": metrics_rows,
        "delong": delong_rows,
    }
    with (args.output_dir / "analysis_report.json").open("w", encoding="utf-8") as fh:
        json.dump(_json_ready(report), fh, indent=2)

    with (args.output_dir / "calibration_deployment_audit.md").open("w", encoding="utf-8") as fh:
        fh.write("# Clean Calibration / Deployment Audit\n\n")
        fh.write("Primary estimates use unweighted fixed L2 logistic regression with nested OOF evaluation.\n\n")
        fh.write("## Availability\n\n")
        for key, value in availability.items():
            fh.write(f"- `{key}`: {value}\n")
        fh.write("\n## Primary Metrics\n\n")
        for row in metrics_rows:
            if row["kind"] != "fixed_unweighted_l2_oof":
                continue
            fh.write(
                f"- {row['model']}: AUC {row['auc']:.3f}, "
                f"Brier {row['brier']:.3f}, ECE {row['ece']:.3f}, "
                f"mean risk {row['mean_predicted_risk']:.3f}\n"
            )
        if delong_rows:
            fh.write("\n## Paired DeLong vs Radiomics Signature\n\n")
            for row in delong_rows:
                fh.write(
                    f"- {row['comparison_model']}: delta "
                    f"{row['auc_delta_reference_minus_comparison']:.3f}, "
                    f"p={row['p_value']:.4f}, paired n={row['paired_n']}\n"
                )

    print(f"Clean calibration audit written to: {args.output_dir}")


if __name__ == "__main__":
    main()
