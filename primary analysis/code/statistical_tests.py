"""
Statistical test utilities.

Currently provides DeLong's test for comparing two correlated ROC AUCs.
Used by `code/models/comparative_risk_stratification_v2.py`.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.stats import norm


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    z = x[order]
    n = z.size
    midranks = np.zeros(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        # midrank in 1-based indexing
        mid = 0.5 * (i + j - 1) + 1
        midranks[i:j] = mid
        i = j

    out = np.empty(n, dtype=float)
    out[order] = midranks
    return out


def _fast_delong(preds_sorted: np.ndarray, n_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong algorithm.

    Parameters
    ----------
    preds_sorted : ndarray, shape (n_classifiers, n_samples)
        Predictions sorted so positives are first.
    n_pos : int
        Number of positive samples.
    """
    preds_sorted = np.asarray(preds_sorted, dtype=float)
    m = int(n_pos)
    n = int(preds_sorted.shape[1] - m)
    if m <= 0 or n <= 0:
        raise ValueError("Need at least one positive and one negative sample.")

    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r] = _compute_midrank(pos[r])
        ty[r] = _compute_midrank(neg[r])
        tz[r] = _compute_midrank(preds_sorted[r])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    # Covariances across classifiers.
    sx = np.cov(v01, bias=False)
    sy = np.cov(v10, bias=False)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict[str, Any]:
    """
    DeLong test for correlated ROC AUCs (two models evaluated on the same cases).

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Binary labels (0/1).
    pred_a : array-like, shape (n_samples,)
        Probabilities/scores from model A.
    pred_b : array-like, shape (n_samples,)
        Probabilities/scores from model B.
    """
    y_true = np.asarray(y_true, dtype=int)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)

    if y_true.shape[0] != pred_a.shape[0] or y_true.shape[0] != pred_b.shape[0]:
        raise ValueError("y_true, pred_a and pred_b must have the same length.")
    if np.unique(y_true).size < 2:
        raise ValueError("y_true must contain both classes for DeLong test.")

    order = np.argsort(-y_true)  # positives first
    y_sorted = y_true[order]
    n_pos = int(np.sum(y_sorted == 1))

    preds_sorted = np.vstack([pred_a[order], pred_b[order]])
    aucs, cov = _fast_delong(preds_sorted, n_pos)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])

    # Variance of the difference.
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    delta = auc_a - auc_b

    if var <= 0 or not np.isfinite(var):
        z = float("nan")
        p = float("nan")
    else:
        z = float(delta / np.sqrt(var))
        p = float(2.0 * norm.sf(abs(z)))

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "delta_auc": float(delta),
        "z": z,
        "p_value": p,
        "var_delta": var,
    }

