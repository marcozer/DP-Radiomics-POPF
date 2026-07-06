#!/usr/bin/env python3
"""Comparative risk stratification v2

Evaluates canonical clinical fistula scores (D-FRS pre-, intra-operative and
DISPAIR) alongside the frozen seven-feature radiomics signature and combined
models that augment the clinical scores with radiomics. The current public
export uses unweighted logistic regression for radiomics/refit models, keeps
published clinical scores unrefit, and exports out-of-fold metrics, calibration
diagnostics, and risk group summaries for manuscript-quality reporting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import csv
import unicodedata
import re
from statistics import NormalDist
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import sys

SCRIPT_PATH = Path(__file__).resolve()
CODE_DIR = SCRIPT_PATH.parent.parent
REPO_DIR = CODE_DIR.parent
DATA_DIR = REPO_DIR / "data"
RESULTS_DIR = REPO_DIR / "results"

RAD_PATH = DATA_DIR / "HF3.csv"
TRUSTABLE_PATH = DATA_DIR / "final_clinical_db.csv"
OUTPUT_DIR_DEFAULT = RESULTS_DIR / "comparative_risk_stratification_v2"

base_str = str(CODE_DIR)
if base_str not in sys.path:
    sys.path.append(base_str)

import numpy as np
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (brier_score_loss, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import PercentFormatter

from statistical_tests import delong_test

import optuna

from utils.plotting_utils import (
    NORD_COLORS,
    create_beautiful_figure,
    save_beautiful_figure,
    setup_plotting,
    plot_model_comparison_with_ci,
)

from calibration_utils import select_best_calibrator, cross_validated_calibrate


STABL_FEATURES = [
    "log-sigma-3-0-mm-3D_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_glcm_ClusterShade",
    "log-sigma-3-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis",
    "log-sigma-7-0-mm-3D_ngtdm_Strength",
    "original_shape_MinorAxisLength",
    "wavelet-HLH_firstorder_Median",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
]

HONORIFICS = {
    "mr", "mrs", "ms", "mme", "mlle", "dr", "prof", "m", "mme.", "mlle.",
    "monsieur", "madame", "doctor", "docteur", "docteure", "professeur"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparative risk stratification with canonical clinical scores")
    parser.add_argument("--radiomics-path", type=Path, default=RAD_PATH,
                        help="Local radiomics feature CSV; not bundled in the public repository")
    parser.add_argument("--clinical-path", type=Path, default=TRUSTABLE_PATH,
                        help="Local clinical score input CSV; not bundled in the public repository")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT),
                        help="Directory where metrics/plots are saved")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Optional cap on number of cases to process (for quick smoke tests)")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of stratified folds for radiomics CV")
    parser.add_argument("--cv-repeats", type=int, default=1,
                        help="Number of CV repetitions (use >1 to densify calibration)")
    parser.add_argument("--calibration-method", type=str, default="auto",
                        choices=["auto", "isotonic", "sigmoid"],
                        help="Calibration strategy: auto chooses best by Brier")
    parser.add_argument("--optuna-trials", type=int, default=40,
                        help="Number of Optuna trials for hyperparameter tuning")
    parser.add_argument("--optuna-timeout", type=int, default=None,
                        help="Optional Optuna timeout per model (seconds)")
    parser.add_argument("--optuna-inner-folds", type=int, default=3,
                        help="Inner CV folds used during Optuna tuning")
    return parser.parse_args()


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def _canonicalize_identifier(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).lower().strip()
    if not text:
        return ""
    text = _strip_accents(text)
    text = text.replace('-', ' ').replace('_', ' ')
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok and tok not in HONORIFICS]
    return "".join(tokens)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "" or text.lower() == "nan":
        return None
    try:
        out = float(text)
        if math.isnan(out):
            return None
        return out
    except ValueError:
        return None


def _is_valid(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _male_indicator(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"male", "m", "1", "true", "oui", "yes"}:
            return 1.0
        if text in {"female", "f", "0", "false", "non", "no"}:
            return 0.0
    try:
        num = float(value)
        if not math.isnan(num):
            return 1.0 if num > 0 else 0.0
    except (TypeError, ValueError):
        return None
    return None


def _truncated_cube(age: Optional[float], knot: float) -> Optional[float]:
    if age is None or math.isnan(age):
        return None
    diff = age - knot
    if diff <= 0:
        return 0.0
    return diff ** 3


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _texture_to_indicator(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"mou", "soft", "souple", "molle"}:
        return 1.0
    if text in {"dur", "dure", "firm", "ferme"}:
        return 0.0
    return None


def _load_radiomics(max_cases: Optional[int] = None) -> Tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    radiomics_map: Dict[str, Dict[str, Optional[float]]] = {}
    order: List[str] = []
    with open(RAD_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            pid = (row.get("scanner_patient_name") or "").strip()
            if not pid:
                continue
            if pid in radiomics_map:
                continue
            feats = {feat: _safe_float(row.get(feat)) for feat in STABL_FEATURES}
            radiomics_map[pid] = feats
            order.append(pid)
            if max_cases is not None and max_cases > 0 and len(order) >= max_cases:
                break
    return radiomics_map, order


def load_data(max_cases: Optional[int] = None) -> List[Dict[str, Any]]:
    radiomics_map, order = _load_radiomics(max_cases)
    clinical_map: Dict[str, Dict[str, Any]] = {}
    with open(TRUSTABLE_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("scanner_patient_name") or "").strip()
            if not pid:
                continue
            entry: Dict[str, Any] = {
                "cr_popf": _safe_float(row.get("cr_popf")),
                "age": _safe_float(row.get("age")),
                "sex": row.get("sex"),
                "bmi": _safe_float(row.get("bmi")),
                "mpd_diameter": _safe_float(row.get("mpd_diameter")),
                "neck_thickness": _safe_float(row.get("neck_thickness")),
                "blood_loss": _safe_float(row.get("blood_loss")),
                "op_duration": _safe_float(row.get("op_duration")),
                "pancreas_texture": (row.get("pancreas_texture") or "").strip(),
                "lesion_body": _safe_float(row.get("lesion_body")),
                "lesion_tail": _safe_float(row.get("lesion_tail")),
                "lesion_isthmus": _safe_float(row.get("lesion_isthmus")),
                "diabetes": row.get("diabetes"),
                "pancreas_transection_site": (row.get("pancreas_transection_site") or "").strip(),
                "tumor_location": (row.get("tumor_location") or "").strip(),
            }
            clinical_map[pid] = entry

    canonical_clinical: Dict[str, Dict[str, Any]] = {}
    for pid, info in clinical_map.items():
        canon = _canonicalize_identifier(pid)
        if canon and canon not in canonical_clinical:
            canonical_clinical[canon] = info

    records: List[Dict[str, Any]] = []
    missing_clinical: List[str] = []
    missing_outcome: List[str] = []
    canonical_matches = 0

    for pid in order:
        clinical = clinical_map.get(pid)
        match_type = "exact"
        if clinical is None:
            canon = _canonicalize_identifier(pid)
            clinical = canonical_clinical.get(canon)
            if clinical is not None:
                canonical_matches += 1
                match_type = "canonical"
        if clinical is None:
            if len(missing_clinical) < 5:
                missing_clinical.append(pid)
            continue
        if clinical.get("cr_popf") is None:
            if len(missing_outcome) < 5:
                missing_outcome.append(pid)
            continue
        rec: Dict[str, Any] = {
            "scanner_patient_name": pid,
            "cr_popf": int(round(clinical.get("cr_popf", 0) or 0)),
            "age": _safe_float(clinical.get("age")),
            "sex": clinical.get("sex"),
            "bmi": _safe_float(clinical.get("bmi")),
            "mpd_diameter": _safe_float(clinical.get("mpd_diameter")),
            "neck_thickness": _safe_float(clinical.get("neck_thickness")),
            "blood_loss": _safe_float(clinical.get("blood_loss")),
            "op_duration": _safe_float(clinical.get("op_duration")),
            "pancreas_texture": clinical.get("pancreas_texture"),
            "pancreas_transection_site": clinical.get("pancreas_transection_site"),
            "tumor_location": clinical.get("tumor_location"),
            "lesion_body": _safe_float(clinical.get("lesion_body")),
            "lesion_tail": _safe_float(clinical.get("lesion_tail")),
            "lesion_isthmus": _safe_float(clinical.get("lesion_isthmus")),
            "lesion_head": _safe_float(clinical.get("lesion_head")),
            "diabetes": clinical.get("diabetes"),
        }
        rec.update(radiomics_map.get(pid, {}))
        rec["_match_type"] = match_type

        records.append(rec)

    total_ids = len(order)
    missing_clinical_count = len([
        pid for pid in order
        if pid not in clinical_map and _canonicalize_identifier(pid) not in canonical_clinical
    ])
    missing_outcome_count = 0
    for pid in order:
        clinical = clinical_map.get(pid)
        if clinical is None:
            clinical = canonical_clinical.get(_canonicalize_identifier(pid))
        if clinical is not None and clinical.get("cr_popf") is None:
            missing_outcome_count += 1
    print(f"[load_data] radiomics IDs: {total_ids}")
    print(f"[load_data] missing clinical rows: {missing_clinical_count}")
    if missing_clinical:
        print(f"    examples: {missing_clinical}")
    print(f"[load_data] missing outcomes: {missing_outcome_count}")
    if missing_outcome:
        print(f"    examples: {missing_outcome}")
    print(f"[load_data] canonical matches used: {canonical_matches}")

    return records


def _diabetes_to_numeric(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not math.isnan(value):
        return 1.0 if value > 0 else 0.0
    text = str(value).strip().lower()
    if text in {"true", "1", "oui", "yes"}:
        return 1.0
    if text in {"false", "0", "non", "no"}:
        return 0.0
    try:
        num = float(text)
        return 1.0 if num > 0 else 0.0
    except ValueError:
        return None


def compute_clinical_scores(records: List[Dict[str, Any]]) -> None:
    for rec in records:
        mpd = rec.get("mpd_diameter")
        neck = rec.get("neck_thickness")
        bmi = rec.get("bmi")
        op_duration_minutes = rec.get("op_duration")
        blood_loss = rec.get("blood_loss")

        soft = _texture_to_indicator(rec.get("pancreas_texture"))
        rec["soft_pancreas"] = soft
        rec["soft_pancreas_flag"] = float(soft > 0) if soft is not None else None

        lesion_head_val = rec.get("lesion_head")
        lesion_body_val = rec.get("lesion_body")
        lesion_tail_val = rec.get("lesion_tail")
        lesion_isthmus_val = rec.get("lesion_isthmus")

        lesion_head = _is_valid(lesion_head_val) and lesion_head_val > 0
        lesion_body = _is_valid(lesion_body_val) and lesion_body_val > 0
        lesion_tail = _is_valid(lesion_tail_val) and lesion_tail_val > 0
        lesion_isthmus = _is_valid(lesion_isthmus_val) and lesion_isthmus_val > 0

        if lesion_head or lesion_isthmus:
            transection = "head"
        elif lesion_body:
            transection = "isthmus"
        elif lesion_tail:
            transection = "body"
        else:
            # Default to an isthmus transection when no lesion flag is provided
            transection = "isthmus"
        rec["transection_site"] = transection
        rec["transection_at_neck"] = 1.0 if transection in {"head", "isthmus"} else 0.0

        diabetes_numeric = _diabetes_to_numeric(rec.get("diabetes"))
        if diabetes_numeric is None:
            diabetes_numeric = 0.0
        rec["diabetes_numeric"] = diabetes_numeric

        male_indicator = _male_indicator(rec.get("sex"))
        rec["male_indicator"] = male_indicator

        if mpd is not None and neck is not None:
            logit = -4.211 + 0.388 * mpd + 0.131 * neck
            if math.isnan(logit):
                rec["dfrs_preop_logit"] = None
                rec["dfrs_preop_prob"] = None
            else:
                rec["dfrs_preop_logit"] = logit
                rec["dfrs_preop_prob"] = float(1.0 / (1.0 + math.exp(-logit)))
        else:
            rec["dfrs_preop_logit"] = None
            rec["dfrs_preop_prob"] = None

        if None not in (mpd, neck, bmi, soft, op_duration_minutes):
            logit = -11.923 + 0.783 * mpd + 0.199 * neck + 0.107 * bmi + 1.592 * soft + 0.005 * op_duration_minutes
            if math.isnan(logit):
                rec["dfrs_intraop_logit"] = None
                rec["dfrs_intraop_prob"] = None
            else:
                rec["dfrs_intraop_logit"] = logit
                rec["dfrs_intraop_prob"] = float(1.0 / (1.0 + math.exp(-logit)))
        else:
            rec["dfrs_intraop_logit"] = None
            rec["dfrs_intraop_prob"] = None

        if _is_valid(neck) and rec.get("transection_at_neck") is not None and diabetes_numeric is not None:
            lp_legacy = -8.322 + 0.384 * neck + 0.545 * rec["transection_at_neck"] - 1.116 * diabetes_numeric
            rec["dispair_legacy_logit"] = lp_legacy
            rec["dispair_legacy_prob"] = float(1.0 / (1.0 + math.exp(-lp_legacy)))
        else:
            rec["dispair_legacy_logit"] = None
            rec["dispair_legacy_prob"] = None

        pt = neck
        mpd_value = mpd
        age_value = rec.get("age")
        male_value = male_indicator

        if _is_valid(pt) and _is_valid(mpd_value) and _is_valid(age_value) and male_value is not None:
            tran_site = rec.get("transection_site") or "unknown"
            tran_neck = 1.0 if tran_site == "isthmus" else 0.0
            tran_head = 1.0 if tran_site == "head" else 0.0

            age_k41 = _truncated_cube(age_value, 41.0)
            age_k65 = _truncated_cube(age_value, 65.0)
            age_k77 = _truncated_cube(age_value, 77.0)

            if None in (age_k41, age_k65, age_k77):
                rec["dispair_refined_logit"] = None
                rec["dispair_refined_prob"] = None
            else:
                lp_refined = (
                    -4.220
                    + 0.111 * pt
                    + 0.00326 * age_value
                    - 1.489e-5 * age_k41
                    + 4.468e-5 * age_k65
                    - 2.979e-5 * age_k77
                    + 0.437 * tran_neck
                    + 0.539 * tran_head
                    - 0.0838 * mpd_value
                    + 0.197 * male_value
                )
                rec["dispair_refined_logit"] = lp_refined
                rec["dispair_refined_prob"] = float(1.0 / (1.0 + math.exp(-lp_refined)))
        else:
            rec["dispair_refined_logit"] = None
            rec["dispair_refined_prob"] = None

        rec["dispair_logit"] = rec.get("dispair_legacy_logit")
        rec["dispair_prob"] = rec.get("dispair_legacy_prob")

def _build_logistic_pipeline(params: Optional[Dict[str, float]] = None) -> Pipeline:
    if params:
        model = LogisticRegression(
            class_weight=None,
            penalty="elasticnet",
            solver="saga",
            max_iter=4000,
            C=params.get("C", 1.0),
            l1_ratio=params.get("l1_ratio", 0.5),
            random_state=42,
        )
    else:
        model = LogisticRegression(
            class_weight=None,
            penalty="l2",
            solver="lbfgs",
            max_iter=4000,
            random_state=42,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", model),
    ])


def _suggest_param(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> float:
    ptype = spec.get("type", "uniform")
    if ptype == "log":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if ptype == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    raise ValueError(f"Unsupported search space type: {ptype}")


def _run_optuna_search(
    X: np.ndarray,
    y: np.ndarray,
    search_space: Dict[str, Dict[str, float]],
    trials: int,
    timeout: Optional[int],
    inner_folds: int,
) -> Tuple[Dict[str, float], float]:
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: _suggest_param(trial, name, spec)
            for name, spec in search_space.items()
        }
        pipeline = _build_logistic_pipeline(params)
        scores: List[float] = []
        for train_idx, val_idx in inner_cv.split(X, y):
            model = clone(pipeline)
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[val_idx])[:, 1]
            if np.unique(y[val_idx]).size < 2:
                continue
            scores.append(roc_auc_score(y[val_idx], prob))
        if not scores:
            raise optuna.TrialPruned()
        mean_score = float(np.mean(scores))
        trial.set_user_attr("params", params)
        return mean_score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials, timeout=timeout, show_progress_bar=False)

    best_trial = study.best_trial
    best_params = best_trial.user_attrs.get("params", {})
    best_score = float(best_trial.value)
    return best_params, best_score


def fit_logistic_oof(records: List[Dict[str, Any]], feature_cols: Sequence[str],
                     target: str, n_splits: int, n_repeats: int = 1,
                     search_space: Optional[Dict[str, Dict[str, float]]] = None,
                     optuna_cfg: Optional[Dict[str, Any]] = None,
                     eligible_subset: Optional[Sequence[int]] = None) -> Optional[Dict[str, Any]]:
    if eligible_subset is not None:
        candidate_indices = [idx for idx in eligible_subset]
    else:
        candidate_indices = range(len(records))

    eligible_indices = [
        idx for idx in candidate_indices
        if _is_valid(records[idx].get(target)) and all(_is_valid(records[idx].get(feat)) for feat in feature_cols)
    ]
    if len(eligible_indices) < n_splits * 2:
        return None

    X = np.array([[float(records[idx][feat]) for feat in feature_cols] for idx in eligible_indices], dtype=float)
    y = np.array([int(records[idx][target]) for idx in eligible_indices], dtype=int)
    if np.unique(y).size < 2:
        return None

    best_params: Optional[Dict[str, float]] = None
    best_score = None
    if search_space and (optuna_cfg or {}).get("trials", 0) > 0:
        cfg = optuna_cfg or {"trials": 40, "timeout": None, "inner_folds": 3}
        best_params, best_score = _run_optuna_search(
            X,
            y,
            search_space,
            cfg.get("trials", 40),
            cfg.get("timeout"),
            cfg.get("inner_folds", 3),
        )

    pipeline = _build_logistic_pipeline(best_params)

    oof_sum = np.zeros(len(eligible_indices), dtype=float)
    counts = np.zeros(len(eligible_indices), dtype=float)
    fold_ids = np.zeros(len(eligible_indices), dtype=int)

    repeats = max(1, n_repeats)
    for repeat in range(repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            model = clone(pipeline)
            model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[test_idx])[:, 1]
            oof_sum[test_idx] += preds
            counts[test_idx] += 1
            if repeat == 0:
                fold_ids[test_idx] = fold

    counts[counts == 0] = 1.0
    oof = oof_sum / counts

    final_model = clone(pipeline)
    final_model.fit(X, y)

    return {
        "indices": eligible_indices,
        "oof": oof,
        "y": y,
        "model": final_model,
        "fold_ids": fold_ids,
        "best_params": best_params,
        "best_score": best_score,
    }


def calibration_error_summary(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    max_error = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        frac_pos = y_true[mask].mean()
        mean_pred = probs[mask].mean()
        error = abs(frac_pos - mean_pred)
        weight = mask.mean()
        ece += weight * error
        max_error = max(max_error, error)
    return float(ece), float(max_error)


def calibration_intercept_slope(y_true: np.ndarray, probs: np.ndarray,
                                max_iter: int = 200, tol: float = 1e-6) -> Tuple[float, float]:
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1 - eps) / np.clip(1 - probs, eps, 1 - eps))
    intercept = 0.0
    slope = 1.0

    for _ in range(max_iter):
        linear = intercept + slope * logits
        linear = np.clip(linear, -50, 50)
        mu = 1.0 / (1.0 + np.exp(-linear))
        diff = y_true - mu

        w = mu * (1 - mu)
        h11 = -(w).sum()
        h12 = -(w * logits).sum()
        h22 = -(w * logits * logits).sum()
        det = h11 * h22 - h12 * h12
        if abs(det) < 1e-12:
            break

        grad_intercept = diff.sum()
        grad_slope = (diff * logits).sum()

        delta_intercept = (-grad_intercept * h22 + grad_slope * h12) / det
        delta_slope = (-grad_slope * h11 + grad_intercept * h12) / det

        intercept += delta_intercept
        slope += delta_slope

        if abs(delta_intercept) < tol and abs(delta_slope) < tol:
            break

    return float(intercept), float(slope)


def youden_operating_point(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    youden = tpr - fpr
    idx = int(np.argmax(youden)) if len(youden) else 0
    threshold = thresholds[idx] if len(thresholds) else 0.5
    sensitivity = tpr[idx] if len(tpr) else float("nan")
    specificity = 1 - fpr[idx] if len(fpr) else float("nan")
    return {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray,
                      thresholds: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 501)

    sensitivities = []
    specificities = []
    ppvs = []
    npvs = []
    youdens = []

    for thr in thresholds:
        pred = probs >= thr
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        tn = float(np.sum((pred == 0) & (y_true == 0)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))

        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        if not math.isnan(sens) and not math.isnan(spec):
            youden = sens + spec - 1.0
        else:
            youden = float("nan")

        sensitivities.append(sens)
        specificities.append(spec)
        ppvs.append(ppv)
        npvs.append(npv)
        youdens.append(youden)

    return {
        "thresholds": np.array(thresholds, dtype=float),
        "sensitivity": np.array(sensitivities, dtype=float),
        "specificity": np.array(specificities, dtype=float),
        "ppv": np.array(ppvs, dtype=float),
        "npv": np.array(npvs, dtype=float),
        "youden": np.array(youdens, dtype=float),
    }


def build_risk_table(probs: np.ndarray, labels: np.ndarray,
                     low_thr: float, high_thr: float) -> List[Dict[str, float]]:
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(int)

    if not np.isfinite(low_thr) or not np.isfinite(high_thr):
        quantiles = np.quantile(probs, [0.33, 0.66])
        low_thr, high_thr = float(quantiles[0]), float(quantiles[1])
    if low_thr > high_thr:
        low_thr, high_thr = high_thr, low_thr

    groups = [
        ("Low", probs <= low_thr),
        ("Intermediate", (probs > low_thr) & (probs <= high_thr)),
        ("High", probs > high_thr),
    ]

    rows: List[Dict[str, float]] = []
    for name, mask in groups:
        mask = np.asarray(mask)
        n = int(mask.sum())
        events = int(labels[mask].sum()) if n else 0
        rate = events / n if n else 0.0
        rows.append({
            "risk_group": name,
            "n": n,
            "events": events,
            "event_rate": rate,
            "event_rate_pct": rate * 100.0,
            "threshold_low": float(low_thr),
            "threshold_high": float(high_thr),
        })
    return rows


def _merge_adjacent_bins(sorted_indices: np.ndarray, y_sorted: np.ndarray,
                         min_events: int, min_count: int) -> List[np.ndarray]:
    merged: List[np.ndarray] = []
    start = 0
    total = len(sorted_indices)
    while start < total:
        end = start + 1
        while end < total:
            segment = sorted_indices[start:end]
            n = len(segment)
            events = int(y_sorted[segment].sum())
            if n >= min_count or events >= min_events:
                break
            end += 1
        merged.append(sorted_indices[start:end])
        start = end
    return merged


def compute_reliability_bins(prob: np.ndarray, y: np.ndarray, n_bins: int = 8,
                             min_events: int = 3, min_count: int = 35) -> Optional[List[Dict[str, float]]]:
    prob = np.asarray(prob)
    y = np.asarray(y)
    if prob.size == 0:
        return None
    order = np.argsort(prob)
    prob_sorted = prob[order]
    y_sorted = y[order]

    bin_edges = np.linspace(0, prob.size, n_bins + 1, dtype=int)
    initial_bins = [np.arange(bin_edges[i], bin_edges[i + 1]) for i in range(n_bins) if bin_edges[i] < bin_edges[i + 1]]
    merged_bins = _merge_adjacent_bins(np.arange(prob.size), y_sorted, min_events, min_count)

    ndist = NormalDist()
    z = ndist.inv_cdf(1 - 0.05 / 2)
    rows: List[Dict[str, float]] = []
    for segment in merged_bins:
        n = len(segment)
        if n == 0:
            continue
        probs_segment = prob_sorted[segment]
        labels_segment = y_sorted[segment]
        positives = int(labels_segment.sum())
        mean_pred = float(probs_segment.mean())
        frac_pos = positives / n if n else 0.0

        denominator = 1 + z ** 2 / n
        center = (frac_pos + z ** 2 / (2 * n)) / denominator
        margin = (z * math.sqrt((frac_pos * (1 - frac_pos) + z ** 2 / (4 * n)) / n)) / denominator
        ci_low = max(0.0, center - margin)
        ci_high = min(1.0, center + margin)

        rows.append({
            "n": n,
            "positives": positives,
            "mean_pred": mean_pred,
            "event_rate": frac_pos,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return rows


def bootstrap_reliability_curve(prob: np.ndarray, y: np.ndarray,
                                grid_size: int = 101, n_bootstrap: int = 300,
                                random_state: int = 42) -> Optional[Dict[str, np.ndarray]]:
    prob = np.asarray(prob)
    y = np.asarray(y)
    if prob.size == 0:
        return None
    rng = np.random.default_rng(random_state)
    grid = np.linspace(0.0, 1.0, grid_size)
    curves = np.zeros((n_bootstrap, grid_size), dtype=float)

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, prob.size, prob.size)
        try:
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(prob[sample_idx], y[sample_idx])
            curves[b] = iso.transform(grid)
        except ValueError:
            curves[b] = y.mean()

    return {
        "grid": grid,
        "mean": curves.mean(axis=0),
        "low": np.quantile(curves, 0.025, axis=0),
        "high": np.quantile(curves, 0.975, axis=0),
    }


def bootstrap_auc_ci(y_true: np.ndarray, probs: np.ndarray,
                     n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)

    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")

    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[idx]
        sample_p = probs[idx]
        if np.unique(sample_y).size < 2:
            continue
        aucs.append(roc_auc_score(sample_y, sample_p))

    if not aucs:
        return float("nan"), float("nan")

    lower = float(np.percentile(aucs, alpha / 2 * 100))
    upper = float(np.percentile(aucs, (1 - alpha / 2) * 100))
    return lower, upper


def make_performance_figure(name: str, slug: str, y: np.ndarray, prob: np.ndarray,
                            output_dir: Path, metrics: Dict[str, Any],
                            prob_raw: Optional[np.ndarray] = None,
                            metrics_raw: Optional[Dict[str, Any]] = None,
                            reliability_cal: Optional[List[Dict[str, float]]] = None,
                            reliability_raw: Optional[List[Dict[str, float]]] = None,
                            curve_cal: Optional[Dict[str, np.ndarray]] = None,
                            curve_raw: Optional[Dict[str, np.ndarray]] = None) -> None:
    setup_plotting()
    fig, axes = create_beautiful_figure("wide")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    raw_color = NORD_COLORS["nord10"]
    cal_color = NORD_COLORS["nord11"]

    # ROC curves
    fpr_cal, tpr_cal, _ = roc_curve(y, prob)
    auc_calibrated = metrics.get("auc_calibrated", metrics["auc"])
    axes[0].plot(fpr_cal, tpr_cal, color=cal_color, linewidth=2.4,
                 label=f"Calibrated AUC = {auc_calibrated:.3f}")
    if prob_raw is not None:
        fpr_raw, tpr_raw, _ = roc_curve(y, prob_raw)
        legend_auc = metrics.get("auc", roc_auc_score(y, prob_raw))
        axes[0].plot(fpr_raw, tpr_raw, color=raw_color, linewidth=2.0, linestyle="--",
                     label=f"Raw AUC = {legend_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color=NORD_COLORS["nord3"], alpha=0.6)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{name} ROC curves")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3, color=NORD_COLORS["nord3"])

    # Calibration plot
    axes[1].plot([0, 1], [0, 1], linestyle="--", color=NORD_COLORS["nord3"], alpha=0.6, label="Ideal")

    if curve_raw is not None:
        axes[1].plot(curve_raw["grid"], curve_raw["mean"], color=raw_color,
                     linestyle="--", linewidth=1.6, alpha=0.7, label="Raw curve")

    if reliability_raw:
        raw_means = [row["mean_pred"] for row in reliability_raw]
        raw_rates = [row["event_rate"] for row in reliability_raw]
        raw_err_low = [max(0.0, row["event_rate"] - row["ci_low"]) for row in reliability_raw]
        raw_err_high = [max(0.0, row["ci_high"] - row["event_rate"]) for row in reliability_raw]
        axes[1].errorbar(raw_means, raw_rates,
                         yerr=[raw_err_low, raw_err_high], fmt="o", markersize=5,
                         mfc="white", mec=raw_color, mew=1.2,
                         ecolor=mcolors.to_rgba(raw_color, alpha=0.35),
                         elinewidth=1.0, capsize=2.5, label="Raw bins")

    if curve_cal is not None:
        axes[1].plot(curve_cal["grid"], curve_cal["mean"], color=cal_color,
                     linewidth=1.8, alpha=0.8, label="Calibrated curve")

    if reliability_cal:
        cal_means = [row["mean_pred"] for row in reliability_cal]
        cal_rates = [row["event_rate"] for row in reliability_cal]
        cal_err_low = [max(0.0, row["event_rate"] - row["ci_low"]) for row in reliability_cal]
        cal_err_high = [max(0.0, row["ci_high"] - row["event_rate"]) for row in reliability_cal]
        axes[1].errorbar(cal_means, cal_rates,
                         yerr=[cal_err_low, cal_err_high], fmt="s", markersize=5.5,
                         mfc="white", mec=cal_color, mew=1.2,
                         ecolor=mcolors.to_rgba(cal_color, alpha=0.35),
                         elinewidth=1.0, capsize=2.5, label="Calibrated bins")

    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Observed CR-POPF rate")
    axes[1].set_title(f"{name} calibration")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3, color=NORD_COLORS["nord3"])
    axes[1].legend(loc="upper left", fontsize=10)

    text_lines = [
        f"Calib AUC={auc_calibrated:.3f}",
        f"Calib Brier={metrics['brier']:.3f}",
        f"Calib ECE={metrics['ece']:.3f}",
        f"Calib slope={metrics['calibration_slope']:.2f}",
        f"Calib intercept={metrics['calibration_intercept']:.2f}",
    ]
    if metrics_raw is not None:
        text_lines.extend([
            f"Raw AUC={metrics_raw.get('auc', float('nan')):.3f}",
            f"Raw Brier={metrics_raw.get('brier', float('nan')):.3f}",
            f"Raw ECE={metrics_raw.get('ece', float('nan')):.3f}",
            f"Raw slope={metrics_raw.get('calibration_slope', float('nan')):.2f}",
            f"Raw intercept={metrics_raw.get('calibration_intercept', float('nan')):.2f}",
        ])
    axes[1].text(
        0.02,
        0.02,
        "\n".join(text_lines),
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=NORD_COLORS["nord3"], alpha=0.9),
    )

    # Probability histogram
    axes[2].hist(prob, bins=20, color=cal_color, alpha=0.65, edgecolor=NORD_COLORS["nord3"], label="Calibrated")
    if prob_raw is not None:
        axes[2].hist(prob_raw, bins=20, color=raw_color, alpha=0.45,
                     edgecolor=NORD_COLORS["nord3"], label="Raw")
    axes[2].set_xlabel("Predicted probability")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"{name} probability distribution")
    axes[2].grid(True, alpha=0.3, color=NORD_COLORS["nord3"])
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    save_beautiful_figure(fig, output_dir / f"{slug}_performance")
    plt.close(fig)


def plot_risk_group_comparison(
    risk_rows: List[Dict[str, Any]],
    output_path: Path,
    models_filter: Optional[Sequence[str]] = None,
) -> None:
    if not risk_rows:
        return

    group_order = ["Low", "Intermediate", "High"]
    if models_filter:
        models = [m for m in models_filter if any(row.get("model") == m for row in risk_rows)]
        filtered_rows = [row for row in risk_rows if row.get("model") in models]
        if not models:
            return
    else:
        models = []
        for row in risk_rows:
            model = row.get("model")
            if model and model not in models:
                models.append(model)
        filtered_rows = list(risk_rows)

    if not models:
        return

    fig, ax = create_beautiful_figure("wide")
    x = np.arange(len(group_order))
    width = 0.8 / max(len(models), 1)

    palette = [
        NORD_COLORS["nord9"],
        NORD_COLORS["nord11"],
        NORD_COLORS["nord14"],
        NORD_COLORS["nord15"],
        NORD_COLORS["nord13"],
    ]

    grouped: Dict[str, Dict[str, Dict[str, Any]]] = {
        model: {grp: {} for grp in group_order} for model in models
    }
    for row in filtered_rows:
        model = row.get("model")
        group = row.get("risk_group")
        if model in grouped and group in grouped[model]:
            grouped[model][group] = row

    ymax = 0.0
    for model in models:
        for group in group_order:
            rate = float(grouped[model][group].get("event_rate", 0.0) or 0.0)
            ymax = max(ymax, rate)
    ymax = 1.0 if ymax <= 0 else min(1.0, ymax * 1.25)

    for idx, model in enumerate(models):
        subset = [grouped[model].get(grp, {}) for grp in group_order]
        rates = np.array([float(item.get("event_rate", 0.0) or 0.0) for item in subset])
        counts = np.array([int(item.get("n", 0) or 0) for item in subset])
        offset = (idx - (len(models) - 1) / 2.0) * width
        color = palette[idx % len(palette)]
        bars = ax.bar(
            x + offset,
            rates,
            width=width,
            label=model,
            color=color,
            edgecolor=NORD_COLORS["nord3"],
            linewidth=1.5,
        )
        for bar, count, rate in zip(bars, counts, rates):
            height = float(bar.get_height())
            pct = rate * 100.0
            label = f"n={count} | {pct:.1f}%"
            y_pos = height + ymax * 0.02 if height > 0 else ymax * 0.02
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                label,
                ha="center",
                va="bottom",
                fontsize=12,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(group_order)
    ax.set_ylabel("Observed CR-POPF rate")
    ax.set_title("Risk group event rates")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0.0, ymax)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_facecolor("white")

    save_beautiful_figure(fig, output_path)
    plt.close(fig)


def evaluate_predictions(name: str, slug: str, y: np.ndarray, prob: np.ndarray,
                         output_dir: Path, prob_raw: Optional[np.ndarray] = None,
                         metrics_raw: Optional[Dict[str, Any]] = None,
                         prob_for_auc: Optional[np.ndarray] = None,
                         make_plots: bool = True) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    metrics: Dict[str, Any] = {
        "model": name,
        "n": int(len(y)),
        "events": int(np.sum(y)),
    }

    if np.unique(y).size < 2:
        metrics.update({
            "auc": float("nan"),
            "auc_ci_low": float("nan"),
            "auc_ci_high": float("nan"),
            "brier": float("nan"),
            "ece": float("nan"),
            "max_calibration_error": float("nan"),
            "calibration_intercept": float("nan"),
            "calibration_slope": float("nan"),
            "youden_threshold": float("nan"),
            "youden_sensitivity": float("nan"),
            "youden_specificity": float("nan"),
            "low_threshold": float("nan"),
            "high_threshold": float("nan"),
        })
        return metrics, []

    prob_auc = prob_for_auc if prob_for_auc is not None else prob
    auc = roc_auc_score(y, prob_auc)
    ci_low, ci_high = bootstrap_auc_ci(y, prob_auc)
    brier = brier_score_loss(y, prob)
    ece, max_err = calibration_error_summary(y, prob)
    cal_intercept, cal_slope = calibration_intercept_slope(y, prob)
    youden = youden_operating_point(y, prob)
    thresh = threshold_metrics(y, prob)
    if np.isfinite(thresh["youden"]).any():
        idx_high = int(np.nanargmax(thresh["youden"]))
        idx_low = int(np.nanargmin(thresh["youden"]))
        high_thr = float(thresh["thresholds"][idx_high])
        low_thr = float(thresh["thresholds"][idx_low])
        if low_thr > high_thr:
            low_thr, high_thr = high_thr, low_thr
    else:
        quantiles = np.quantile(prob, [0.33, 0.66])
        low_thr, high_thr = float(quantiles[0]), float(quantiles[1])

    # Ensure groups are populated; fall back to tertiles if degeneracy occurs
    min_group = max(5, int(0.05 * len(prob)))
    low_count = int((prob <= low_thr).sum())
    high_count = int((prob > high_thr).sum())
    if low_count < min_group or high_count < min_group:
        quantiles = np.quantile(prob, [0.33, 0.66])
        low_thr, high_thr = float(quantiles[0]), float(quantiles[1])

    if low_thr > high_thr:
        low_thr, high_thr = high_thr, low_thr

    metrics.update({
        "auc": float(auc),
        "auc_ci_low": float(ci_low),
        "auc_ci_high": float(ci_high),
        "brier": float(brier),
        "ece": float(ece),
        "max_calibration_error": float(max_err),
        "calibration_intercept": float(cal_intercept),
        "calibration_slope": float(cal_slope),
        "youden_threshold": youden["threshold"],
        "youden_sensitivity": youden["sensitivity"],
        "youden_specificity": youden["specificity"],
        "low_threshold": low_thr,
        "high_threshold": high_thr,
    })

    if prob_for_auc is not None:
        metrics["auc_probability_source"] = "raw_oof"
        metrics["auc_calibrated"] = float(roc_auc_score(y, prob))
        cal_ci_low, cal_ci_high = bootstrap_auc_ci(y, prob)
        metrics["auc_calibrated_ci_low"] = float(cal_ci_low)
        metrics["auc_calibrated_ci_high"] = float(cal_ci_high)

    risk_rows = build_risk_table(prob, y, low_thr, high_thr)
    for row in risk_rows:
        row["model"] = name

    if make_plots:
        reliability_cal = compute_reliability_bins(prob, y)
        reliability_raw = compute_reliability_bins(prob_raw, y) if prob_raw is not None else None
        curve_cal = bootstrap_reliability_curve(prob, y)
        curve_raw = bootstrap_reliability_curve(prob_raw, y) if prob_raw is not None else None

        make_performance_figure(
            name,
            slug,
            y,
            prob,
            output_dir,
            metrics,
            prob_raw=prob_raw,
            metrics_raw=metrics_raw,
            reliability_cal=reliability_cal,
            reliability_raw=reliability_raw,
            curve_cal=curve_cal,
            curve_raw=curve_raw,
        )
        thresholds_metrics = threshold_metrics(y, prob)
        fig_thr, axes_thr = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        low_thr_val = metrics["low_threshold"]
        high_thr_val = metrics["high_threshold"]
        thresholds = thresholds_metrics['thresholds']
        youden = thresholds_metrics['youden']
        axes_thr[0].plot(thresholds, youden, color=NORD_COLORS['nord9'], linewidth=2.0)
        axes_thr[0].set_ylabel('Youden index')
        axes_thr[0].set_title(f'{name} Youden index across thresholds')
        axes_thr[0].grid(True, alpha=0.3, color=NORD_COLORS['nord3'])
        axes_thr[0].set_facecolor('white')
        for thr, label_str in [(high_thr_val, 'High threshold'), (low_thr_val, 'Low threshold')]:
            if math.isnan(thr):
                continue
            axes_thr[0].axvline(thr, color=NORD_COLORS['nord11'], linestyle='--', linewidth=1.5)
            axes_thr[0].text(thr, 0.02, f'{label_str} = {thr:.3f}', rotation=90,
                             va='bottom', ha='right', fontsize=10, color=NORD_COLORS['nord11'])

        axes_thr[1].plot(thresholds, thresholds_metrics['sensitivity'], label='Sensitivity',
                         color=NORD_COLORS['nord11'], linewidth=2.0)
        axes_thr[1].plot(thresholds, thresholds_metrics['specificity'], label='Specificity',
                         color=NORD_COLORS['nord10'], linewidth=2.0)
        axes_thr[1].plot(thresholds, thresholds_metrics['ppv'], label='PPV',
                         color=NORD_COLORS['nord14'], linewidth=2.0)
        axes_thr[1].plot(thresholds, thresholds_metrics['npv'], label='NPV',
                         color=NORD_COLORS['nord15'], linewidth=2.0)
        axes_thr[1].set_xlabel('Threshold')
        axes_thr[1].set_ylabel('Value')
        axes_thr[1].set_ylim(-0.05, 1.05)
        axes_thr[1].grid(True, alpha=0.3, color=NORD_COLORS['nord3'])
        axes_thr[1].set_facecolor('white')
        axes_thr[1].legend(loc='center right', fontsize=10)
        for thr in [low_thr_val, high_thr_val]:
            if math.isnan(thr):
                continue
            axes_thr[1].axvline(thr, color=NORD_COLORS['nord3'], linestyle='--', linewidth=1.0)

        fig_thr.tight_layout()
        save_beautiful_figure(fig_thr, output_dir / f"{slug}_threshold_diagnostics")
        plt.close(fig_thr)

    return metrics, risk_rows


def main() -> None:
    args = parse_args()
    global RAD_PATH, TRUSTABLE_PATH
    RAD_PATH = args.radiomics_path
    TRUSTABLE_PATH = args.clinical_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_data(max_cases=args.max_cases)
    compute_clinical_scores(records)

    # Debug counts for availability
    total = len(records)
    pre_available = sum(1 for r in records if r.get("dfrs_preop_prob") is not None)
    intra_available = sum(1 for r in records if r.get("dfrs_intraop_prob") is not None)
    dispair_available = sum(1 for r in records if r.get("dispair_prob") is not None)
    print(f"Total records: {total}")
    print(f"  DP-FRS pre-op available: {pre_available}")
    print(f"  DP-FRS intra-op available: {intra_available}")
    print(f"  DISPAIR available: {dispair_available}")

    clinical_feature_set = [
        "mpd_diameter",
        "neck_thickness",
        "bmi",
        "blood_loss",
        "op_duration",
        "soft_pancreas_flag",
        "diabetes_numeric",
        "transection_at_neck",
    ]

    clinical_available = sum(1 for r in records if all(_is_valid(r.get(f)) for f in clinical_feature_set))
    print(f"  Clinical refit features available: {clinical_available}")

    # To compare models fairly, restrict logistic-heavy cohorts to patients with
    # complete clinical score information (pre-, intra-operative, and DISPAIR).
    common_indices = [
        idx for idx, rec in enumerate(records)
        if all(
            rec.get(key) is not None
            for key in ("dfrs_preop_prob", "dfrs_intraop_prob", "dispair_legacy_prob")
        ) and all(_is_valid(rec.get(feat)) for feat in STABL_FEATURES)
    ]
    print(f"  Common radiomics/clinical cohort: {len(common_indices)}")

    optuna_cfg = {
        "trials": args.optuna_trials,
        "timeout": args.optuna_timeout,
        "inner_folds": args.optuna_inner_folds,
    }

    models = [
        {"name": "Radiomics signature", "slug": "radiomics", "type": "logistic", "features": list(STABL_FEATURES),
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Clinical refit", "slug": "clinical_refit", "type": "logistic", "features": list(clinical_feature_set)},
        {"name": "Radiomics + clinical refit", "slug": "radiomics_clinical_refit", "type": "logistic",
         "features": list(STABL_FEATURES) + list(clinical_feature_set),
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Radiomics + D-FRS pre-operative", "slug": "radiomics_dfrs_preop", "type": "logistic",
         "features": list(STABL_FEATURES) + ["dfrs_preop_logit"],
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Radiomics + D-FRS intra-operative", "slug": "radiomics_dfrs_intra", "type": "logistic",
         "features": list(STABL_FEATURES) + ["dfrs_intraop_logit"],
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Radiomics + DISPAIR", "slug": "radiomics_dispair", "type": "logistic",
         "features": list(STABL_FEATURES) + ["dispair_legacy_logit"],
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Radiomics + DISPAIR refined", "slug": "radiomics_dispair_refined", "type": "logistic",
         "features": list(STABL_FEATURES) + ["dispair_refined_logit"],
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "Radiomics + all scores", "slug": "radiomics_all_scores", "type": "logistic",
         "features": list(STABL_FEATURES) + ["dfrs_preop_logit", "dfrs_intraop_logit", "dispair_legacy_logit"],
         "search_space": {
             "C": {"type": "log", "low": 1e-2, "high": 1e2},
             "l1_ratio": {"type": "uniform", "low": 0.2, "high": 0.55},
         }},
        {"name": "D-FRS pre-operative", "slug": "dfrs_preop", "type": "score", "prob_key": "dfrs_preop_prob"},
        {"name": "D-FRS intra-operative", "slug": "dfrs_intra", "type": "score", "prob_key": "dfrs_intraop_prob"},
        {"name": "DISPAIR-FRS", "slug": "dispair", "type": "score", "prob_key": "dispair_legacy_prob"},
        {"name": "DISPAIR-FRS refined", "slug": "dispair_refined", "type": "score", "prob_key": "dispair_refined_prob"},
    ]

    metrics_rows: List[Dict[str, Any]] = []
    risk_rows: List[Dict[str, Any]] = []
    model_predictions: Dict[str, Dict[str, Any]] = {}

    for spec in models:
        if spec["type"] in ("radiomics", "logistic"):
            result = fit_logistic_oof(
                records,
                spec["features"],
                "cr_popf",
                args.cv_folds,
                args.cv_repeats,
                search_space=spec.get("search_space"),
                optuna_cfg=optuna_cfg,
                eligible_subset=common_indices,
            )
            if result is None:
                print(f"[WARN] Skipping {spec['name']} (insufficient data)")
                continue
            y = result["y"]
            prob_raw = result["oof"]
            fold_ids = result["fold_ids"]
            best_params = result.get("best_params")
            best_score = result.get("best_score")

            if args.calibration_method == "auto":
                candidate_methods = ("isotonic", "sigmoid")
            else:
                candidate_methods = (args.calibration_method,)

            best_method, best_payload, diagnostics = select_best_calibrator(prob_raw, y, methods=candidate_methods)
            prob = best_payload["prob"]

            metrics_raw_info, _ = evaluate_predictions(
                f"{spec['name']} (raw)",
                f"{spec['slug']}_raw",
                y,
                prob_raw,
                output_dir,
                prob_for_auc=prob_raw,
                make_plots=False,
            )

            prob_cal_cv = cross_validated_calibrate(prob_raw, y, fold_ids, best_method)

            patient_ids = [records[idx]["scanner_patient_name"] for idx in result["indices"]]
            model_predictions[spec["slug"]] = {
                "patients": patient_ids,
                "labels": y.tolist(),
                "raw": prob_raw.tolist(),
                "calibrated": prob_cal_cv.tolist(),
            }

            metrics, risk = evaluate_predictions(
                spec["name"],
                spec["slug"],
                y,
                prob_cal_cv,
                output_dir,
                prob_raw=prob_raw,
                metrics_raw=metrics_raw_info,
                prob_for_auc=prob_raw,
            )

            brier_raw = float(brier_score_loss(y, prob_raw))
            brier_cal_cv = float(brier_score_loss(y, prob_cal_cv))
            brier_cal_full = float(brier_score_loss(y, prob))
            metrics["calibration_method"] = best_method
            metrics["calibration"] = best_method
            metrics["calibration_brier"] = brier_cal_cv
            metrics["brier_raw"] = brier_raw
            metrics["calibration_brier_delta"] = brier_cal_cv - brier_raw
            metrics["raw_calibration_slope"] = metrics_raw_info.get("calibration_slope")
            metrics["raw_calibration_intercept"] = metrics_raw_info.get("calibration_intercept")
            metrics["raw_ece"] = metrics_raw_info.get("ece")
            metrics["calibration_brier_full"] = brier_cal_full
            if best_params:
                metrics["optuna_best_params"] = best_params
            if best_score is not None:
                metrics["optuna_best_score"] = best_score

            calibration_report = {
                "method": best_method,
                "brier_cv": brier_cal_cv,
                "brier_full": brier_cal_full,
                "brier_raw": brier_raw,
                "raw_ece": metrics_raw_info.get("ece"),
                "raw_calibration_slope": metrics_raw_info.get("calibration_slope"),
                "raw_calibration_intercept": metrics_raw_info.get("calibration_intercept"),
                "auc_raw": metrics_raw_info.get("auc"),
                "auc_calibrated": metrics.get("auc_calibrated"),
                "optuna_best_params": best_params,
                "optuna_best_score": best_score,
                "diagnostics": {
                    method: {
                        "brier": float(payload["brier"]),
                        "info": payload["info"],
                    }
                    for method, payload in diagnostics.items()
                },
            }
            with open(output_dir / f"{spec['slug']}_calibration.json", "w", encoding="utf-8") as fh:
                json.dump(calibration_report, fh, indent=2)
        else:
            indices = [
                idx for idx, rec in enumerate(records)
                if rec.get(spec["prob_key"]) is not None and rec.get("cr_popf") is not None
            ]
            values = [(int(records[idx]["cr_popf"]), float(records[idx][spec["prob_key"]]))
                      for idx in indices
                      if records[idx][spec["prob_key"]] is not None and not math.isnan(records[idx][spec["prob_key"]])]
            if len(values) < 5:
                print(f"[WARN] Skipping {spec['name']} (insufficient data)")
                continue
            y = np.array([v[0] for v in values], dtype=int)
            if np.unique(y).size < 2:
                print(f"[WARN] Skipping {spec['name']} (single-class subset)")
                continue
            prob = np.array([v[1] for v in values], dtype=float)
            metrics, risk = evaluate_predictions(spec["name"], spec["slug"], y, prob, output_dir)
            brier_val = metrics.get("brier", float("nan"))
            metrics["calibration_method"] = "precomputed"
            metrics["calibration"] = "precomputed"
            metrics["calibration_brier"] = brier_val
            metrics["brier_raw"] = brier_val
            metrics["calibration_brier_delta"] = 0.0
            metrics["calibration_brier_full"] = brier_val
            metrics["auc_calibrated"] = metrics["auc"]
            metrics["auc_calibrated_ci_low"] = metrics["auc_ci_low"]
            metrics["auc_calibrated_ci_high"] = metrics["auc_ci_high"]
            if "auc_probability_source" not in metrics:
                metrics["auc_probability_source"] = "precomputed"

        metrics_rows.append(metrics)
        risk_rows.extend(risk)

    comparisons: Dict[str, Any] = {}
    slug_a = "radiomics"
    slug_b = "radiomics_dfrs_preop"
    if slug_a in model_predictions and slug_b in model_predictions:
        data_a = model_predictions[slug_a]
        data_b = model_predictions[slug_b]

        map_a = {
            pid: (float(prob), int(label))
            for pid, prob, label in zip(
                data_a["patients"], data_a["calibrated"], data_a["labels"]
            )
        }
        map_b = {
            pid: (float(prob), int(label))
            for pid, prob, label in zip(
                data_b["patients"], data_b["calibrated"], data_b["labels"]
            )
        }

        common_ids = [pid for pid in data_b["patients"] if pid in map_a]
        labels: List[int] = []
        preds_a: List[float] = []
        preds_b: List[float] = []
        for pid in common_ids:
            prob_a, label_a = map_a[pid]
            prob_b, label_b = map_b[pid]
            if label_a != label_b:
                continue
            labels.append(label_a)
            preds_a.append(prob_a)
            preds_b.append(prob_b)

        if labels and len(set(labels)) == 2:
            delong_res = delong_test(
                np.array(labels, dtype=float),
                np.array(preds_a, dtype=float),
                np.array(preds_b, dtype=float),
            )
            comparisons["radiomics_vs_radiomics_dfrs_calibrated"] = delong_res

    if not metrics_rows:
        raise RuntimeError("No models produced evaluable results.")
    metrics_path = output_dir / "model_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)
    payload: Dict[str, Any] = {"models": metrics_rows}
    if comparisons:
        payload["comparisons"] = comparisons
    with open(output_dir / "model_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    if risk_rows:
        risk_path = output_dir / "risk_group_summary.csv"
        with open(risk_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(risk_rows[0].keys()))
            writer.writeheader()
            writer.writerows(risk_rows)
    else:
        risk_path = None

    if risk_rows:
        plot_risk_group_comparison(risk_rows, output_dir / "risk_group_event_rates")
        plot_risk_group_comparison(
            risk_rows,
            output_dir / "risk_group_event_rates_radiomics_vs_dfrs",
            models_filter=[
                "Radiomics signature",
                "Radiomics + D-FRS pre-operative",
            ],
        )

    valid_auc = [m for m in metrics_rows if not math.isnan(m.get("auc", float("nan")))]
    if valid_auc:
        order = sorted(valid_auc, key=lambda m: m["auc"], reverse=True)
        model_names = [m["model"] for m in order]
        aucs = [m["auc"] for m in order]
        cis = [(m["auc_ci_low"], m["auc_ci_high"]) for m in order]
        fig, ax = plot_model_comparison_with_ci(model_names, aucs, cis,
                                                title="Comparative AUROC (OOF predictions)")
        save_beautiful_figure(fig, output_dir / "auc_comparison")
        plt.close(fig)

    print("\nComparative risk stratification v2 results saved to:")
    print(f"  Metrics: {metrics_path}")
    if risk_path:
        print(f"  Risk table: {risk_path}")
    print(f"  Figures: {output_dir}")


if __name__ == "__main__":
    main()
