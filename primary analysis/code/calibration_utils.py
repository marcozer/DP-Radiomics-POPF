"""
Calibration utilities used by `code/models/comparative_risk_stratification_v2.py`.

This module is intentionally lightweight and self-contained so calibration can be:
- selected (isotonic vs sigmoid) by Brier score,
- cross-validated to avoid optimistic calibration estimates, and
- exported in JSON-friendly form for deployment (see the `info` payloads).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class CalibratorPayload:
    method: str
    prob: np.ndarray
    brier: float
    info: Dict[str, Any]


def _fit_sigmoid_calibrator(prob: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Platt-style recalibration on logit(prob).
    x = _logit(prob).reshape(-1, 1)
    y = y.astype(int)

    # Prefer penalty='none' when supported; fall back to a very weak L2 otherwise.
    try:
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
        lr.fit(x, y)
    except Exception:
        lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=2000)
        lr.fit(x, y)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    prob_cal = _sigmoid(intercept + slope * _logit(prob))
    info = {"type": "sigmoid", "intercept": intercept, "slope": slope}
    return prob_cal, info


def _fit_isotonic_calibrator(prob: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(prob.astype(float), y.astype(float))
    prob_cal = ir.predict(prob.astype(float))

    # JSON-friendly representation for deployment scripts.
    info = {
        "type": "isotonic",
        "x": [float(v) for v in getattr(ir, "X_thresholds_", []).tolist()],
        "y": [float(v) for v in getattr(ir, "y_thresholds_", []).tolist()],
        "out_of_bounds": "clip",
    }
    return prob_cal, info


def _fit_calibrator(prob: np.ndarray, y: np.ndarray, method: str) -> CalibratorPayload:
    prob = np.asarray(prob, dtype=float)
    y = np.asarray(y, dtype=int)

    if np.unique(y).size < 2:
        # Degenerate subset: return identity mapping.
        info = {"type": "identity", "reason": "single_class_subset"}
        return CalibratorPayload(method=method, prob=prob, brier=float("nan"), info=info)

    method = method.lower().strip()
    if method == "sigmoid":
        prob_cal, info = _fit_sigmoid_calibrator(prob, y)
    elif method == "isotonic":
        prob_cal, info = _fit_isotonic_calibrator(prob, y)
    else:
        raise ValueError(f"Unsupported calibration method: {method!r}")

    return CalibratorPayload(
        method=method,
        prob=np.asarray(prob_cal, dtype=float),
        brier=float(brier_score_loss(y, prob_cal)),
        info=info,
    )


def select_best_calibrator(
    prob_raw: np.ndarray,
    y_true: np.ndarray,
    *,
    methods: Iterable[str] = ("isotonic", "sigmoid"),
) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Fit candidate calibrators on the full set of predictions and pick the best by Brier score.

    Returns
    -------
    best_method, best_payload, diagnostics

    Each payload is JSON-friendly:
      {"method": ..., "prob": np.ndarray, "brier": float, "info": dict}
    """
    diagnostics: Dict[str, Dict[str, Any]] = {}

    best_method = None
    best_payload = None
    best_brier = float("inf")

    for method in methods:
        payload = _fit_calibrator(prob_raw, y_true, method)
        diagnostics[payload.method] = {
            "brier": float(payload.brier),
            "prob": payload.prob,
            "info": payload.info,
        }
        if np.isfinite(payload.brier) and payload.brier < best_brier:
            best_brier = float(payload.brier)
            best_method = payload.method
            best_payload = diagnostics[payload.method]

    if best_method is None or best_payload is None:
        raise RuntimeError("Could not select a calibrator (no finite Brier scores).")

    return best_method, best_payload, diagnostics


def cross_validated_calibrate(
    prob_raw: np.ndarray,
    y_true: np.ndarray,
    fold_ids: np.ndarray,
    method: str,
) -> np.ndarray:
    """
    Cross-validated calibration: fit calibrator on all folds except one, apply to held-out fold.

    Parameters
    ----------
    prob_raw : array-like
        Out-of-fold probabilities (raw, uncalibrated).
    y_true : array-like
        Binary labels aligned to prob_raw.
    fold_ids : array-like
        Fold assignment per sample (0..K-1).
    method : str
        'sigmoid' or 'isotonic'.
    """
    prob_raw = np.asarray(prob_raw, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    fold_ids = np.asarray(fold_ids, dtype=int)

    calibrated = np.full_like(prob_raw, fill_value=np.nan, dtype=float)
    for fold in np.unique(fold_ids):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        payload = _fit_calibrator(prob_raw[train_mask], y_true[train_mask], method)

        info = payload.info
        if info.get("type") == "identity":
            calibrated[test_mask] = prob_raw[test_mask]
            continue

        if info["type"] == "sigmoid":
            intercept = float(info["intercept"])
            slope = float(info["slope"])
            calibrated[test_mask] = _sigmoid(intercept + slope * _logit(prob_raw[test_mask]))
        elif info["type"] == "isotonic":
            x = np.asarray(info["x"], dtype=float)
            y = np.asarray(info["y"], dtype=float)
            if x.size == 0 or y.size == 0 or x.size != y.size:
                calibrated[test_mask] = prob_raw[test_mask]
            else:
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                calibrated[test_mask] = np.interp(prob_raw[test_mask], x, y, left=y[0], right=y[-1])
        else:
            raise ValueError(f"Unsupported calibrator payload type: {info.get('type')!r}")

    # Fallback: if any NaNs remain (should not happen), revert to raw.
    nan_mask = ~np.isfinite(calibrated)
    calibrated[nan_mask] = prob_raw[nan_mask]
    return np.clip(calibrated, 0.0, 1.0)

