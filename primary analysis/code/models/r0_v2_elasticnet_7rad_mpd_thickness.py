#!/usr/bin/env python3
"""R0_v2 elastic-net analysis for the locked 7-rad signature ± MPD/thickness.

This public script intentionally does not bundle data. It expects local,
non-committed radiomics and clinical CSV files and writes only aggregate
metrics, anonymized row-index predictions, and manuscript-style figures.

Final manuscript policy:
- the locked 7 radiomics features are refitted with one estimator family:
  standardized, unweighted elastic-net logistic regression;
- the comparative radioclinical model uses the same 7 features plus
  `mpd_diameter` and `neck_thickness`;
- DP-FRS and DISPAIR are reported as standalone published-score benchmarks,
  not as no-refit fused radiomics-score models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent.parent
REPO_DIR = CODE_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"

if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))

try:
    from statistical_tests import delong_test
except Exception:  # pragma: no cover - optional in minimal environments
    delong_test = None


STABL_FEATURES = [
    "log-sigma-3-0-mm-3D_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_glcm_ClusterShade",
    "log-sigma-3-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis",
    "log-sigma-7-0-mm-3D_ngtdm_Strength",
    "original_shape_MinorAxisLength",
    "wavelet-HLH_firstorder_Median",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
]

CLINICAL_FEATURES = ["mpd_diameter", "neck_thickness"]
RANDOM_SEED = 20260616

NORD = {
    "nord0": "#2E3440",
    "nord3": "#4C566A",
    "nord9": "#81A1C1",
    "nord10": "#5E81AC",
    "nord11": "#BF616A",
    "nord13": "#EBCB8B",
    "nord14": "#A3BE8C",
}

MODEL_COLORS = {
    "7-rad": "nord10",
    "7-rad + MPD/thickness": "nord11",
    "DP-FRS preoperative": "nord14",
    "DP-FRS intraoperative": "nord13",
    "DISPAIR": "nord9",
}

HONORIFICS = {
    "mr",
    "mrs",
    "ms",
    "mme",
    "mlle",
    "dr",
    "prof",
    "m",
    "monsieur",
    "madame",
    "doctor",
    "docteur",
    "docteure",
    "professeur",
}


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    label: str
    features: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics-path", type=Path, default=DATA_DIR / "HF3.csv")
    parser.add_argument("--clinical-path", type=Path, default=DATA_DIR / "final_clinical_db.csv")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "r0_v2_elasticnet_7rad_mpd_thickness")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-n", type=int, default=632)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--id-col", type=str, default="scanner_patient_name")
    parser.add_argument("--expected-n", type=int, default=195)
    parser.add_argument("--expected-events", type=int, default=36)
    parser.add_argument(
        "--export-model-pkl",
        type=Path,
        default=None,
        help="Optional path for a non-patient-level deployable 7-rad elastic-net model bundle.",
    )
    return parser.parse_args()


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def canonicalize(value: Any) -> str:
    if value is None:
        return ""
    text = strip_accents(str(value).lower().strip())
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok and tok not in HONORIFICS]
    return "".join(tokens)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return None if math.isnan(out) else out


def is_valid(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def yes_no_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return 1.0 if float(value) > 0 else 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "oui", "yes", "y"}:
        return 1.0
    if text in {"false", "0", "non", "no", "n"}:
        return 0.0
    try:
        numeric = float(text)
        return 1.0 if numeric > 0 else 0.0
    except ValueError:
        pass
    return None


def male_indicator(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"male", "m", "1", "true", "oui", "yes"}:
        return 1.0
    if text in {"female", "f", "0", "false", "non", "no"}:
        return 0.0
    return safe_float(value)


def texture_to_indicator(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"mou", "soft", "souple", "molle"}:
        return 1.0
    if text in {"dur", "dure", "firm", "ferme", "hard"}:
        return 0.0
    return None


def truncated_cube(age: Optional[float], knot: float) -> Optional[float]:
    if age is None or math.isnan(age):
        return None
    diff = age - knot
    return 0.0 if diff <= 0 else diff**3


def detect_id(row: Dict[str, Any], preferred: str) -> str:
    for candidate in [preferred, "scanner_patient_name", "patient_id", "PatientName", "name"]:
        value = row.get(candidate)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def load_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if not args.radiomics_path.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {args.radiomics_path}")
    if not args.clinical_path.exists():
        raise FileNotFoundError(f"Clinical CSV not found: {args.clinical_path}")

    radiomics_map: Dict[str, Dict[str, Optional[float]]] = {}
    order: List[str] = []
    with args.radiomics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = detect_id(row, args.id_col)
            if not pid or pid in radiomics_map:
                continue
            radiomics_map[pid] = {feature: safe_float(row.get(feature)) for feature in STABL_FEATURES}
            order.append(pid)
            if args.max_cases and len(order) >= args.max_cases:
                break

    clinical_map: Dict[str, Dict[str, Any]] = {}
    canonical_clinical: Dict[str, Dict[str, Any]] = {}
    with args.clinical_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = detect_id(row, args.id_col)
            if not pid:
                continue
            entry = {
                "cr_popf": safe_float(row.get("cr_popf")),
                "age": safe_float(row.get("age")),
                "sex": row.get("sex"),
                "bmi": safe_float(row.get("bmi")),
                "mpd_diameter": safe_float(row.get("mpd_diameter")),
                "neck_thickness": safe_float(row.get("neck_thickness")),
                "blood_loss": safe_float(row.get("blood_loss")),
                "op_duration": safe_float(row.get("op_duration")),
                "pancreas_texture": row.get("pancreas_texture"),
                "pancreas_transection_site": row.get("pancreas_transection_site"),
                "tumor_location": row.get("tumor_location"),
                "lesion_head": safe_float(row.get("lesion_head")),
                "lesion_body": safe_float(row.get("lesion_body")),
                "lesion_tail": safe_float(row.get("lesion_tail")),
                "lesion_isthmus": safe_float(row.get("lesion_isthmus")),
                "diabetes": row.get("diabetes"),
            }
            clinical_map[pid] = entry
            canon = canonicalize(pid)
            if canon and canon not in canonical_clinical:
                canonical_clinical[canon] = entry

    records: List[Dict[str, Any]] = []
    missing_clinical = 0
    missing_outcomes = 0
    canonical_matches = 0
    for pid in order:
        clinical = clinical_map.get(pid)
        if clinical is None:
            clinical = canonical_clinical.get(canonicalize(pid))
            if clinical is not None:
                canonical_matches += 1
        if clinical is None:
            missing_clinical += 1
            continue
        if clinical.get("cr_popf") is None:
            missing_outcomes += 1
            continue
        rec = {"row_index": len(records), "cr_popf": int(round(float(clinical["cr_popf"])))}
        rec.update(clinical)
        rec.update(radiomics_map[pid])
        records.append(rec)

    print(f"[load_records] radiomics IDs: {len(order)}")
    print(f"[load_records] missing clinical rows: {missing_clinical}")
    print(f"[load_records] missing outcomes: {missing_outcomes}")
    print(f"[load_records] canonical matches used: {canonical_matches}")
    return records


def compute_clinical_scores(records: List[Dict[str, Any]]) -> None:
    for rec in records:
        mpd = rec.get("mpd_diameter")
        neck = rec.get("neck_thickness")
        bmi = rec.get("bmi")
        op_duration = rec.get("op_duration")
        soft = texture_to_indicator(rec.get("pancreas_texture"))
        diabetes = yes_no_numeric(rec.get("diabetes"))
        if diabetes is None:
            diabetes = 0.0

        lesion_head = is_valid(rec.get("lesion_head")) and float(rec["lesion_head"]) > 0
        lesion_body = is_valid(rec.get("lesion_body")) and float(rec["lesion_body"]) > 0
        lesion_tail = is_valid(rec.get("lesion_tail")) and float(rec["lesion_tail"]) > 0
        lesion_isthmus = is_valid(rec.get("lesion_isthmus")) and float(rec["lesion_isthmus"]) > 0

        if lesion_head or lesion_isthmus:
            transection = "head"
        elif lesion_body:
            transection = "isthmus"
        elif lesion_tail:
            transection = "body"
        else:
            transection = "isthmus"
        rec["transection_at_neck"] = 1.0 if transection in {"head", "isthmus"} else 0.0

        if mpd is not None and neck is not None:
            logit = -4.211 + 0.388 * mpd + 0.131 * neck
            rec["dfrs_preop_logit"] = logit
            rec["dfrs_preop_prob"] = 1.0 / (1.0 + math.exp(-logit))
        else:
            rec["dfrs_preop_logit"] = None
            rec["dfrs_preop_prob"] = None

        if None not in (mpd, neck, bmi, soft, op_duration):
            logit = -11.923 + 0.783 * mpd + 0.199 * neck + 0.107 * bmi + 1.592 * soft + 0.005 * op_duration
            rec["dfrs_intraop_logit"] = logit
            rec["dfrs_intraop_prob"] = 1.0 / (1.0 + math.exp(-logit))
        else:
            rec["dfrs_intraop_logit"] = None
            rec["dfrs_intraop_prob"] = None

        if is_valid(neck):
            logit = -8.322 + 0.384 * neck + 0.545 * rec["transection_at_neck"] - 1.116 * diabetes
            rec["dispair_legacy_logit"] = logit
            rec["dispair_legacy_prob"] = 1.0 / (1.0 + math.exp(-logit))
        else:
            rec["dispair_legacy_logit"] = None
            rec["dispair_legacy_prob"] = None


def param_grid() -> List[Dict[str, float]]:
    return [
        {"C": c, "l1_ratio": l1}
        for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        for l1 in [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    ]


def make_model(params: Optional[Dict[str, float]] = None, seed: int = RANDOM_SEED) -> Pipeline:
    params = params or {"C": 1.0, "l1_ratio": 0.5}
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=float(params["C"]),
                    l1_ratio=float(params["l1_ratio"]),
                    class_weight=None,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_dataset(records: List[Dict[str, Any]], features: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    eligible = [
        idx
        for idx, rec in enumerate(records)
        if is_valid(rec.get("cr_popf")) and all(is_valid(rec.get(feature)) for feature in features)
    ]
    X = np.asarray([[float(records[idx][feature]) for feature in features] for idx in eligible], dtype=float)
    y = np.asarray([int(records[idx]["cr_popf"]) for idx in eligible], dtype=int)
    return X, y, eligible


def tune_params(X: np.ndarray, y: np.ndarray, inner_folds: int, seed: int) -> Tuple[Dict[str, float], float]:
    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_params: Optional[Dict[str, float]] = None
    best_auc = -np.inf
    for params in param_grid():
        aucs: List[float] = []
        for train_idx, val_idx in cv.split(X, y):
            model = make_model(params, seed=seed)
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[val_idx])[:, 1]
            if np.unique(y[val_idx]).size > 1:
                aucs.append(float(roc_auc_score(y[val_idx], prob)))
        if aucs and float(np.mean(aucs)) > best_auc:
            best_auc = float(np.mean(aucs))
            best_params = dict(params)
    if best_params is None:
        raise RuntimeError("No valid elastic-net hyperparameter setting found.")
    return best_params, best_auc


def repeated_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    outer_folds: int,
    repeats: int,
    inner_folds: int,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    pred_sum = np.zeros(len(y), dtype=float)
    pred_count = np.zeros(len(y), dtype=float)
    details: List[Dict[str, Any]] = []
    for repeat in range(repeats):
        outer = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=RANDOM_SEED + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y)):
            params, inner_auc = tune_params(X[train_idx], y[train_idx], inner_folds, RANDOM_SEED + repeat * 1000 + fold)
            model = make_model(params, seed=RANDOM_SEED + repeat * 1000 + fold)
            model.fit(X[train_idx], y[train_idx])
            pred_sum[test_idx] += model.predict_proba(X[test_idx])[:, 1]
            pred_count[test_idx] += 1
            details.append({"repeat": repeat, "fold": fold, "inner_auc": inner_auc, **params})
    return pred_sum / np.maximum(pred_count, 1), details


def bootstrap_632plus(
    X: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    inner_folds: int,
    fixed_params: Dict[str, float],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED + 17)
    rows: List[Dict[str, float]] = []
    skipped = 0
    n = len(y)
    for iteration in range(n_bootstrap):
        train_idx = rng.integers(0, n, size=n)
        inbag = np.zeros(n, dtype=bool)
        inbag[np.unique(train_idx)] = True
        oob_idx = np.flatnonzero(~inbag)
        if len(oob_idx) < 10 or np.unique(y[oob_idx]).size < 2 or np.unique(y[train_idx]).size < 2:
            skipped += 1
            continue
        model = make_model(fixed_params, seed=RANDOM_SEED + 10_000 + iteration)
        model.fit(X[train_idx], y[train_idx])
        auc_train = float(roc_auc_score(y[train_idx], model.predict_proba(X[train_idx])[:, 1]))
        auc_oob = float(roc_auc_score(y[oob_idx], model.predict_proba(X[oob_idx])[:, 1]))
        err_train = 1.0 - auc_train
        err_oob = 1.0 - auc_oob
        denom = 0.5 - err_train
        relative_overfit = 0.0 if denom <= 1e-12 else float(np.clip((err_oob - err_train) / denom, 0.0, 1.0))
        weight = 0.632 / (1.0 - 0.368 * relative_overfit)
        rows.append(
            {
                "iteration": float(iteration),
                "auc_train": auc_train,
                "auc_oob": auc_oob,
                "auc_632plus": 1.0 - ((1.0 - weight) * err_train + weight * err_oob),
                "weight_632plus": float(weight),
                "relative_overfit": relative_overfit,
                "C": float(fixed_params["C"]),
                "l1_ratio": float(fixed_params["l1_ratio"]),
                "oob_n": float(len(oob_idx)),
            }
        )
    df = pd.DataFrame(rows)
    values = df["auc_632plus"].to_numpy(dtype=float)
    return (
        {
            "auc_632plus": float(np.mean(values)),
            "auc_632plus_ci_low": float(np.percentile(values, 2.5)),
            "auc_632plus_ci_high": float(np.percentile(values, 97.5)),
            "auc_oob_mean": float(df["auc_oob"].mean()),
            "auc_train_mean": float(df["auc_train"].mean()),
            "n_iterations": int(len(df)),
            "n_skipped": int(skipped),
            "mean_weight_632plus": float(df["weight_632plus"].mean()),
        },
        df,
    )


def calibration_slope_intercept(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    eps = 1e-8
    logits = np.log(np.clip(pred, eps, 1 - eps) / np.clip(1 - pred, eps, 1 - eps)).reshape(-1, 1)
    try:
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=5000)
    except TypeError:
        model = LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000)
    model.fit(logits, y)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_error(y: np.ndarray, pred: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.clip(np.digitize(pred, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    max_error = 0.0
    for idx in range(n_bins):
        mask = ids == idx
        if not np.any(mask):
            continue
        err = abs(float(y[mask].mean()) - float(pred[mask].mean()))
        ece += float(mask.mean()) * err
        max_error = max(max_error, err)
    return float(ece), float(max_error)


def summarize(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    ece, max_cal_error = calibration_error(y, pred)
    intercept, slope = calibration_slope_intercept(y, pred)
    return {
        "auc": float(roc_auc_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "logloss": float(log_loss(y, np.clip(pred, 1e-8, 1 - 1e-8), labels=[0, 1])),
        "ece": ece,
        "max_calibration_error": max_cal_error,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "mean_predicted_risk": float(np.mean(pred)),
        "median_predicted_risk": float(np.median(pred)),
    }


def bootstrap_auc_ci(y: np.ndarray, pred: np.ndarray, n_bootstrap: int = 5000) -> Tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED + 55)
    aucs: List[float] = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size > 1:
            aucs.append(float(roc_auc_score(y[idx], pred[idx])))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_auc_delta(y: np.ndarray, pred_ref: np.ndarray, pred_alt: np.ndarray) -> Dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED + 33)
    deltas: List[float] = []
    n = len(y)
    for _ in range(5000):
        idx = rng.integers(0, n, size=n)
        if np.unique(y[idx]).size > 1:
            deltas.append(float(roc_auc_score(y[idx], pred_alt[idx]) - roc_auc_score(y[idx], pred_ref[idx])))
    arr = np.asarray(deltas, dtype=float)
    observed = float(roc_auc_score(y, pred_alt) - roc_auc_score(y, pred_ref))
    p_two = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
    return {
        "delta_auc": observed,
        "delta_auc_ci_low": float(np.percentile(arr, 2.5)),
        "delta_auc_ci_high": float(np.percentile(arr, 97.5)),
        "paired_bootstrap_p": float(min(1.0, p_two)),
    }


def delong_compare(y: np.ndarray, pred_ref: np.ndarray, pred_alt: np.ndarray) -> Dict[str, float]:
    if delong_test is None:
        return {"delong_z": math.nan, "delong_p": math.nan}
    try:
        result = delong_test(y, pred_ref, pred_alt)
        if isinstance(result, dict):
            return {
                "delong_z": float(result.get("z", result.get("z_score", math.nan))),
                "delong_p": float(result.get("p_value", result.get("p", math.nan))),
            }
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return {"delong_z": float(result[0]), "delong_p": float(result[1])}
    except Exception:
        pass
    return {"delong_z": math.nan, "delong_p": math.nan}


def wilson_ci(events: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = events / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def quantile_reliability_bins(prob: np.ndarray, y: np.ndarray, n_bins: int = 6) -> pd.DataFrame:
    order = np.argsort(prob)
    rows: List[Dict[str, float]] = []
    for idx, chunk in enumerate(np.array_split(order, n_bins), start=1):
        events = int(y[chunk].sum())
        n = int(len(chunk))
        ci_low, ci_high = wilson_ci(events, n)
        rows.append(
            {
                "bin": idx,
                "n": n,
                "events": events,
                "mean_predicted": float(prob[chunk].mean()),
                "observed_rate": float(events / n),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")


def plot_figure(
    output_dir: Path,
    y: np.ndarray,
    apparent_predictions: Dict[str, np.ndarray],
    boot_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> None:
    plt.rcParams.update({"svg.fonttype": "none", "font.family": "DejaVu Sans Mono", "font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.6), dpi=300)

    for slug, label in [("radiomics_7rad", "7-rad"), ("radioclinical_mpd_thickness", "7-rad + MPD/thickness")]:
        color = NORD[MODEL_COLORS[label]]
        fpr, tpr, _ = roc_curve(y, apparent_predictions[slug])
        boot = boot_df.loc[boot_df["slug"] == slug].iloc[0]
        axes[0].plot(
            fpr,
            tpr,
            color=color,
            linewidth=3.0,
            label=f"{label}: .632+ AUC {boot['auc_632plus']:.3f} [{boot['auc_632plus_ci_low']:.3f}-{boot['auc_632plus_ci_high']:.3f}]",
        )
    axes[0].plot([0, 1], [0, 1], "--", color=NORD["nord3"], alpha=0.6)
    axes[0].set_title("Apparent ROC curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(True, alpha=0.3, color=NORD["nord3"])
    axes[0].legend(loc="lower right", fontsize=11)

    for slug, label in [("radiomics_7rad", "7-rad"), ("radioclinical_mpd_thickness", "7-rad + MPD/thickness")]:
        color = NORD[MODEL_COLORS[label]]
        bins = quantile_reliability_bins(apparent_predictions[slug], y, n_bins=6)
        xs = bins["mean_predicted"].to_numpy()
        ys = bins["observed_rate"].to_numpy()
        yerr = [ys - bins["ci_low"].to_numpy(), bins["ci_high"].to_numpy() - ys]
        axes[1].plot(xs, ys, color=color, linewidth=2.6, label=label)
        axes[1].errorbar(xs, ys, yerr=yerr, fmt="o", markersize=7, mfc="white", mec=color, mew=1.8, ecolor=color)
    axes[1].plot([0, 1], [0, 1], "--", color=NORD["nord3"], alpha=0.65, label="Ideal")
    axes[1].set_title("Apparent calibration")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Observed CR-POPF rate")
    axes[1].set_xlim(-0.02, 0.62)
    axes[1].set_ylim(-0.02, 0.75)
    axes[1].grid(True, alpha=0.3, color=NORD["nord3"])
    axes[1].legend(loc="upper left", fontsize=11)

    order = ["DP-FRS preoperative", "DP-FRS intraoperative", "DISPAIR", "7-rad", "7-rad + MPD/thickness"]
    plot_rows = benchmark_df.copy()
    plot_rows["order"] = plot_rows["model"].map({name: i for i, name in enumerate(order)})
    plot_rows = plot_rows.sort_values("order")
    y_pos = np.arange(len(plot_rows))
    aucs = plot_rows["auc"].to_numpy(dtype=float)
    low = plot_rows["auc_ci_low"].to_numpy(dtype=float)
    high = plot_rows["auc_ci_high"].to_numpy(dtype=float)
    colors = [NORD[MODEL_COLORS.get(label, "nord10")] for label in plot_rows["model"]]
    axes[2].barh(y_pos, aucs, color=colors, alpha=0.86)
    axes[2].errorbar(aucs, y_pos, xerr=[aucs - low, high - aucs], fmt="none", color=NORD["nord0"], capsize=4)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(plot_rows["model"].tolist())
    axes[2].set_xlim(0.35, 0.9)
    axes[2].set_xlabel("AUC")
    axes[2].set_title("Published benchmarks and refit models")
    axes[2].grid(True, axis="x", alpha=0.3, color=NORD["nord3"])
    for x, y_i in zip(aucs, y_pos):
        axes[2].text(x + 0.012, y_i, f"{x:.3f}", va="center", fontsize=12)

    if not comparison_df.empty:
        comp = comparison_df.iloc[0]
        fig.text(
            0.01,
            0.01,
            f"Paired validated comparison, 7-rad + MPD/thickness vs 7-rad: "
            f"Delta AUC {comp['delta_auc_oof']:.3f} [{comp['delta_auc_oof_ci_low']:.3f} to {comp['delta_auc_oof_ci_high']:.3f}], "
            f"bootstrap P={comp['paired_bootstrap_p_oof']:.3f}.",
            fontsize=12,
            color=NORD["nord3"],
        )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    save_figure(fig, output_dir / "figure_r0_v2_elasticnet_7rad_mpd_thickness")
    plt.close(fig)


def score_predictions(records: List[Dict[str, Any]], eligible_indices: Sequence[int], key: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = [i for i in eligible_indices if is_valid(records[i].get(key))]
    return (
        np.asarray([int(records[i]["cr_popf"]) for i in idx], dtype=int),
        np.asarray([float(records[i][key]) for i in idx], dtype=float),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)
    compute_clinical_scores(records)

    combined_features = STABL_FEATURES + CLINICAL_FEATURES
    X_common, y, eligible_indices = build_dataset(records, combined_features)
    if args.expected_n and len(y) != args.expected_n:
        raise RuntimeError(f"Unexpected cohort n={len(y)}; expected {args.expected_n}.")
    if args.expected_events is not None and int(y.sum()) != args.expected_events:
        raise RuntimeError(f"Unexpected event count={int(y.sum())}; expected {args.expected_events}.")

    feature_index = {feature: idx for idx, feature in enumerate(combined_features)}
    specs = [
        ModelSpec("radiomics_7rad", "7-rad", STABL_FEATURES),
        ModelSpec("radioclinical_mpd_thickness", "7-rad + MPD/thickness", combined_features),
    ]

    metrics_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    params_rows: List[Dict[str, Any]] = []
    pred_frames: List[pd.DataFrame] = []
    apparent_predictions: Dict[str, np.ndarray] = {}
    oof_predictions: Dict[str, np.ndarray] = {}
    full_models: Dict[str, Pipeline] = {}
    boot_frames: List[pd.DataFrame] = []

    for spec in specs:
        cols = [feature_index[feature] for feature in spec.features]
        X = X_common[:, cols]
        params, inner_auc = tune_params(X, y, args.inner_folds, RANDOM_SEED)
        model = make_model(params)
        model.fit(X, y)
        full_models[spec.slug] = model
        apparent = model.predict_proba(X)[:, 1]
        oof, fold_details = repeated_oof_predictions(X, y, args.outer_folds, args.repeats, args.inner_folds)
        boot_summary, boot_df = bootstrap_632plus(X, y, args.bootstrap_n, args.inner_folds, params)

        apparent_predictions[spec.slug] = apparent
        oof_predictions[spec.slug] = oof
        params_rows.append({"slug": spec.slug, "model": spec.label, "best_inner_auc_full_data": inner_auc, **params})
        metrics_rows.append({"slug": spec.slug, "model": spec.label, "probability_source": "apparent_full_cohort", "n": len(y), "events": int(y.sum()), **summarize(y, apparent)})
        metrics_rows.append({"slug": spec.slug, "model": spec.label, "probability_source": f"{args.outer_folds}x{args.repeats}_nested_oof", "n": len(y), "events": int(y.sum()), **summarize(y, oof)})
        boot_rows.append({"slug": spec.slug, "model": spec.label, **boot_summary})
        boot_frames.append(boot_df.assign(slug=spec.slug, model=spec.label))
        pred_frames.append(
            pd.DataFrame(
                {
                    "row_index": eligible_indices,
                    "cr_popf": y,
                    "slug": spec.slug,
                    "model": spec.label,
                    "oof_probability": oof,
                    "apparent_probability": apparent,
                }
            )
        )
        pd.DataFrame([{**row, "slug": spec.slug, "model": spec.label} for row in fold_details]).to_csv(
            args.output_dir / f"{spec.slug}_fold_details.csv", index=False
        )

    boot_df = pd.DataFrame(boot_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    pd.DataFrame(params_rows).to_csv(args.output_dir / "full_data_elasticnet_hyperparameters.csv", index=False)
    metrics_df.to_csv(args.output_dir / "model_metrics.csv", index=False)
    boot_df.to_csv(args.output_dir / "bootstrap_632plus_metrics.csv", index=False)
    pd.concat(boot_frames, ignore_index=True).to_csv(args.output_dir / "bootstrap_632plus_replicates.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(args.output_dir / "predictions_anonymized.csv", index=False)

    benchmark_rows: List[Dict[str, Any]] = []
    for slug, label, key in [
        ("dfrs_preop", "DP-FRS preoperative", "dfrs_preop_prob"),
        ("dfrs_intra", "DP-FRS intraoperative", "dfrs_intraop_prob"),
        ("dispair", "DISPAIR", "dispair_legacy_prob"),
    ]:
        score_y, score_prob = score_predictions(records, eligible_indices, key)
        summary = summarize(score_y, score_prob)
        ci_low, ci_high = bootstrap_auc_ci(score_y, score_prob)
        benchmark_rows.append(
            {
                "slug": slug,
                "model": label,
                "probability_source": "published_score_formula",
                "n": int(len(score_y)),
                "events": int(score_y.sum()),
                "auc": summary["auc"],
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "brier": summary["brier"],
                "ece": summary["ece"],
                "calibration_intercept": summary["calibration_intercept"],
                "calibration_slope": summary["calibration_slope"],
                "complete_on_full_model_cohort": bool(len(score_y) == len(eligible_indices)),
            }
        )
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(args.output_dir / "clinical_score_benchmarks.csv", index=False)

    refit_rows = []
    for _, row in boot_df.iterrows():
        oof_summary = metrics_df[(metrics_df["slug"] == row["slug"]) & (metrics_df["probability_source"].str.contains("nested_oof"))].iloc[0]
        refit_rows.append(
            {
                "slug": row["slug"],
                "model": row["model"],
                "probability_source": "elasticnet_refit_bootstrap_632plus",
                "n": int(len(y)),
                "events": int(y.sum()),
                "auc": float(row["auc_632plus"]),
                "auc_ci_low": float(row["auc_632plus_ci_low"]),
                "auc_ci_high": float(row["auc_632plus_ci_high"]),
                "brier": float(oof_summary["brier"]),
                "ece": float(oof_summary["ece"]),
                "calibration_intercept": float(oof_summary["calibration_intercept"]),
                "calibration_slope": float(oof_summary["calibration_slope"]),
                "complete_on_full_model_cohort": True,
            }
        )
    benchmark_plot_df = pd.concat([benchmark_df, pd.DataFrame(refit_rows)], ignore_index=True)
    benchmark_plot_df.to_csv(args.output_dir / "benchmark_plot_table.csv", index=False)

    oof_comp = paired_bootstrap_auc_delta(y, oof_predictions["radiomics_7rad"], oof_predictions["radioclinical_mpd_thickness"])
    app_comp = paired_bootstrap_auc_delta(y, apparent_predictions["radiomics_7rad"], apparent_predictions["radioclinical_mpd_thickness"])
    delong_oof = delong_compare(y, oof_predictions["radiomics_7rad"], oof_predictions["radioclinical_mpd_thickness"])
    delong_app = delong_compare(y, apparent_predictions["radiomics_7rad"], apparent_predictions["radioclinical_mpd_thickness"])
    comparison_df = pd.DataFrame(
        [
            {
                "reference": "7-rad",
                "alternative": "7-rad + MPD/thickness",
                "delta_auc_oof": oof_comp["delta_auc"],
                "delta_auc_oof_ci_low": oof_comp["delta_auc_ci_low"],
                "delta_auc_oof_ci_high": oof_comp["delta_auc_ci_high"],
                "paired_bootstrap_p_oof": oof_comp["paired_bootstrap_p"],
                "delong_z_oof": delong_oof["delong_z"],
                "delong_p_oof": delong_oof["delong_p"],
                "delta_auc_apparent": app_comp["delta_auc"],
                "delta_auc_apparent_ci_low": app_comp["delta_auc_ci_low"],
                "delta_auc_apparent_ci_high": app_comp["delta_auc_ci_high"],
                "paired_bootstrap_p_apparent": app_comp["paired_bootstrap_p"],
                "delong_z_apparent": delong_app["delong_z"],
                "delong_p_apparent": delong_app["delong_p"],
            }
        ]
    )
    comparison_df.to_csv(args.output_dir / "paired_auc_comparison_7rad_vs_mpd_thickness.csv", index=False)

    reliability_frames = []
    for slug, label in [("radiomics_7rad", "7-rad"), ("radioclinical_mpd_thickness", "7-rad + MPD/thickness")]:
        reliability_frames.append(quantile_reliability_bins(apparent_predictions[slug], y).assign(slug=slug, model=label, probability_source="apparent_full_cohort"))
        reliability_frames.append(quantile_reliability_bins(oof_predictions[slug], y).assign(slug=slug, model=label, probability_source=f"{args.outer_folds}x{args.repeats}_nested_oof"))
    pd.concat(reliability_frames, ignore_index=True).to_csv(args.output_dir / "calibration_reliability_bins.csv", index=False)

    plot_figure(args.output_dir, y, apparent_predictions, boot_df, benchmark_plot_df, comparison_df)

    rad = boot_df.loc[boot_df["slug"] == "radiomics_7rad"].iloc[0]
    comb = boot_df.loc[boot_df["slug"] == "radioclinical_mpd_thickness"].iloc[0]
    comp = comparison_df.iloc[0]
    report = {
        "cohort": {"n": int(len(y)), "events": int(y.sum()), "event_rate": float(y.mean())},
        "features": {"radiomics_7rad": STABL_FEATURES, "clinical_added": CLINICAL_FEATURES},
        "primary": rad.to_dict(),
        "comparative_model": comb.to_dict(),
        "paired_comparison": comp.to_dict(),
        "clinical_benchmarks": benchmark_df.to_dict(orient="records"),
    }
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2))
    (args.output_dir / "analysis_report.md").write_text(
        "\n".join(
            [
                "# R0_v2 elastic-net 7-rad +/- MPD/thickness analysis",
                "",
                f"Cohort: {len(y)} patients, {int(y.sum())} CR-POPF events ({y.mean():.1%}).",
                "",
                f"Primary 7-rad bootstrap .632+ AUC: {rad['auc_632plus']:.3f} [{rad['auc_632plus_ci_low']:.3f}-{rad['auc_632plus_ci_high']:.3f}].",
                f"7-rad + MPD/thickness bootstrap .632+ AUC: {comb['auc_632plus']:.3f} [{comb['auc_632plus_ci_low']:.3f}-{comb['auc_632plus_ci_high']:.3f}].",
                f"Validated paired OOF delta AUC: {comp['delta_auc_oof']:.3f} [{comp['delta_auc_oof_ci_low']:.3f} to {comp['delta_auc_oof_ci_high']:.3f}], bootstrap P={comp['paired_bootstrap_p_oof']:.3f}.",
                "",
                "## Standalone published-score benchmarks",
                benchmark_df[["model", "n", "events", "auc", "auc_ci_low", "auc_ci_high", "brier", "ece"]].to_markdown(index=False),
                "",
            ]
        )
    )

    if args.export_model_pkl is not None:
        import joblib

        args.export_model_pkl.parent.mkdir(parents=True, exist_ok=True)
        model_bundle = {
            "schema_version": "r0_v2_elasticnet_7rad_v1",
            "model": full_models["radiomics_7rad"],
            "feature_names": STABL_FEATURES,
            "model_label": "7-rad elastic-net full-cohort refit",
            "metadata": {
                "cohort_n": int(len(y)),
                "events": int(y.sum()),
                "class_weight": "none",
                "estimator": "StandardScaler + LogisticRegression elastic-net",
                "primary_performance": {
                    "metric": "bootstrap .632+ AUC",
                    "auc": float(rad["auc_632plus"]),
                    "ci_low": float(rad["auc_632plus_ci_low"]),
                    "ci_high": float(rad["auc_632plus_ci_high"]),
                },
                "full_data_hyperparameters": pd.DataFrame(params_rows)
                .loc[lambda df: df["slug"] == "radiomics_7rad"]
                .iloc[0]
                .to_dict(),
            },
        }
        joblib.dump(model_bundle, args.export_model_pkl)

    print(f"Saved R0_v2 public outputs to {args.output_dir}")
    print(boot_df[["model", "auc_632plus", "auc_632plus_ci_low", "auc_632plus_ci_high"]].to_string(index=False))
    print(comparison_df.to_string(index=False))
    print(benchmark_df[["model", "n", "events", "auc", "auc_ci_low", "auc_ci_high"]].to_string(index=False))


if __name__ == "__main__":
    main()
