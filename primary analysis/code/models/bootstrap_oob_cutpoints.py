#!/usr/bin/env python3
"""Validate 7-rad rule-out and rule-in cutpoints out of bag.

Each class-stratified bootstrap replicate refits the StandardScaler and locked
elastic-net model, selects constrained-MCC cutpoints in bag, and applies those
cutpoints unchanged to the corresponding out-of-bag patients.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ANALYSIS_ROOT / "data_anonymized/radiomics_features_anonymized.csv"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "results/bootstrap_oob_cutpoints"
MODEL_SEED = 20260616
BOOTSTRAP_SEED = 20260721
MODEL_PARAMS = {"C": 3.0, "l1_ratio": 0.05}

FEATURES = [
    "log-sigma-3-0-mm-3D_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_glcm_ClusterShade",
    "log-sigma-3-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis",
    "log-sigma-7-0-mm-3D_ngtdm_Strength",
    "original_shape_MinorAxisLength",
    "wavelet-HLH_firstorder_Median",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
]


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = ["patient_id", "cr_popf", *FEATURES]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if not frame["patient_id"].astype(str).str.fullmatch(r"RDP_\d{3}").all():
        raise ValueError("The public input must contain only RDP_### identifiers")
    matrix = frame[FEATURES].apply(pd.to_numeric, errors="raise").to_numpy(float)
    outcome = frame["cr_popf"].to_numpy(int)
    if len(frame) != 195 or int(outcome.sum()) != 36:
        raise ValueError(
            f"Expected 195 patients and 36 events; found {len(frame)} and "
            f"{int(outcome.sum())}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("The locked seven-feature matrix contains missing values")
    return matrix, outcome


def make_model(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=MODEL_PARAMS["C"],
                    l1_ratio=MODEL_PARAMS["l1_ratio"],
                    class_weight=None,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=seed,
                ),
            ),
        ]
    )


def stratified_bootstrap_indices(
    outcome: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    negative = np.flatnonzero(outcome == 0)
    positive = np.flatnonzero(outcome == 1)
    in_bag = np.concatenate(
        [
            rng.choice(negative, size=len(negative), replace=True),
            rng.choice(positive, size=len(positive), replace=True),
        ]
    )
    rng.shuffle(in_bag)
    represented = np.zeros(len(outcome), dtype=bool)
    represented[np.unique(in_bag)] = True
    return in_bag, np.flatnonzero(~represented)


def auc_632plus(
    outcome_in_bag: np.ndarray,
    probability_in_bag: np.ndarray,
    outcome_oob: np.ndarray,
    probability_oob: np.ndarray,
) -> dict[str, float]:
    auc_in_bag = float(roc_auc_score(outcome_in_bag, probability_in_bag))
    auc_oob = float(roc_auc_score(outcome_oob, probability_oob))
    error_in_bag = 1.0 - auc_in_bag
    error_oob = 1.0 - auc_oob
    relative_overfit = (error_oob - error_in_bag) / max(
        0.5 - error_in_bag, 1e-12
    )
    relative_overfit = float(np.clip(relative_overfit, 0.0, 1.0))
    weight = 0.632 / (1.0 - 0.368 * relative_overfit)
    corrected_auc = 1.0 - (
        (1.0 - weight) * error_in_bag + weight * error_oob
    )
    return {
        "auc_in_bag": auc_in_bag,
        "auc_oob": auc_oob,
        "relative_overfit": relative_overfit,
        "weight_632plus": weight,
        "auc_632plus": corrected_auc,
    }


def threshold_metrics(
    outcome: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        outcome, predicted, labels=[0, 1]
    ).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity_rule_out": sensitivity,
        "specificity_rule_in": specificity,
        "positive_predictive_value": ppv,
        "negative_predictive_value": npv,
        "balanced_accuracy": float(
            balanced_accuracy_score(outcome, predicted)
        ),
        "f1_score": float(f1_score(outcome, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(outcome, predicted)),
    }


def choose_threshold(
    outcome: np.ndarray, probability: np.ndarray, mode: str
) -> dict[str, float | int]:
    candidates = np.unique(np.r_[0.0, probability, 1.0])
    predicted = probability[np.newaxis, :] >= candidates[:, np.newaxis]
    positive = outcome.astype(bool)[np.newaxis, :]
    negative = ~positive
    tp = np.sum(predicted & positive, axis=1)
    fp = np.sum(predicted & negative, axis=1)
    fn = np.sum(~predicted & positive, axis=1)
    tn = np.sum(~predicted & negative, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.divide(tp, tp + fn)
        specificity = np.divide(tn, tn + fp)
        ppv = np.divide(tp, tp + fp)
        npv = np.divide(tn, tn + fn)
        f1 = np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros_like(tp, dtype=float),
            where=(2 * tp + fp + fn) != 0,
        )
        mcc_denominator = np.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        )
        mcc = np.divide(
            tp * tn - fp * fn,
            mcc_denominator,
            out=np.zeros_like(tp, dtype=float),
            where=mcc_denominator != 0,
        )
    balanced = (sensitivity + specificity) / 2
    rows = [
        {
            "threshold": float(candidates[index]),
            "tp": int(tp[index]),
            "fp": int(fp[index]),
            "tn": int(tn[index]),
            "fn": int(fn[index]),
            "sensitivity_rule_out": float(sensitivity[index]),
            "specificity_rule_in": float(specificity[index]),
            "positive_predictive_value": float(ppv[index]),
            "negative_predictive_value": float(npv[index]),
            "balanced_accuracy": float(balanced[index]),
            "f1_score": float(f1[index]),
            "mcc": float(mcc[index]),
        }
        for index in range(len(candidates))
    ]
    if mode == "rule_out":
        eligible = [
            row for row in rows if row["sensitivity_rule_out"] >= 0.90
        ]
        secondary = "sensitivity_rule_out"
    elif mode == "rule_in":
        eligible = [
            row for row in rows if row["specificity_rule_in"] >= 0.90
        ]
        secondary = "specificity_rule_in"
    else:
        raise ValueError(mode)
    if not eligible:
        eligible = rows
    return max(
        eligible,
        key=lambda row: (
            row["mcc"],
            row["balanced_accuracy"],
            row[secondary],
            -abs(float(row["threshold"]) - 0.5),
        ),
    )


def percentile_interval(values: pd.Series) -> tuple[float, float]:
    clean = values.dropna().to_numpy(float)
    if not len(clean):
        return math.nan, math.nan
    return float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))


def full_cohort_analysis(
    matrix: np.ndarray, outcome: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray, float]:
    estimator = make_model(MODEL_SEED)
    estimator.fit(matrix, outcome)
    probability = estimator.predict_proba(matrix)[:, 1]
    rows = []
    for mode, label, constraint in (
        (
            "rule_out",
            "Rule-out",
            "full-cohort sensitivity >=90%; maximize MCC",
        ),
        (
            "rule_in",
            "Rule-in",
            "full-cohort specificity >=90%; maximize MCC",
        ),
    ):
        rows.append(
            {
                "strategy": label,
                "constraint": constraint,
                **choose_threshold(outcome, probability, mode),
            }
        )
    return (
        pd.DataFrame(rows),
        probability,
        float(roc_auc_score(outcome, probability)),
    )


def summarize_risk_strata(
    probability: np.ndarray,
    outcome: np.ndarray,
    cutpoints: pd.DataFrame,
) -> pd.DataFrame:
    rule_out = float(
        cutpoints.loc[cutpoints["strategy"].eq("Rule-out"), "threshold"].iloc[0]
    )
    rule_in = float(
        cutpoints.loc[cutpoints["strategy"].eq("Rule-in"), "threshold"].iloc[0]
    )
    strata = np.where(
        probability < rule_out,
        "Rule-out",
        np.where(probability >= rule_in, "Rule-in", "Intermediate"),
    )
    rows = []
    for label in ("Rule-out", "Intermediate", "Rule-in"):
        mask = strata == label
        events = int(outcome[mask].sum())
        rows.append(
            {
                "stratum": label,
                "n": int(mask.sum()),
                "events": events,
                "event_rate": events / int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_cutpoints(
    full_cutpoints: pd.DataFrame, replicates: pd.DataFrame
) -> pd.DataFrame:
    metrics = [
        "sensitivity_rule_out",
        "specificity_rule_in",
        "positive_predictive_value",
        "negative_predictive_value",
        "balanced_accuracy",
        "f1_score",
        "mcc",
    ]
    rows: list[dict[str, Any]] = []
    for strategy in ("Rule-out", "Rule-in"):
        group = replicates.loc[replicates["strategy"].eq(strategy)]
        point = full_cutpoints.loc[
            full_cutpoints["strategy"].eq(strategy)
        ].iloc[0]
        threshold_low, threshold_high = percentile_interval(
            group["selected_threshold"]
        )
        row: dict[str, Any] = {
            "strategy": strategy,
            "constraint": point["constraint"],
            "full_cohort_threshold": float(point["threshold"]),
            "full_cohort_tp": int(point["tp"]),
            "full_cohort_fp": int(point["fp"]),
            "full_cohort_tn": int(point["tn"]),
            "full_cohort_fn": int(point["fn"]),
            "bootstrap_threshold_mean": float(
                group["selected_threshold"].mean()
            ),
            "bootstrap_threshold_median": float(
                group["selected_threshold"].median()
            ),
            "bootstrap_threshold_ci_low": threshold_low,
            "bootstrap_threshold_ci_high": threshold_high,
            "bootstrap_replicates": int(len(group)),
            "median_oob_n": float(group["n_oob"].median()),
            "median_oob_events": float(group["events_oob"].median()),
        }
        for metric in metrics:
            column = f"oob_{metric}"
            low, high = percentile_interval(group[column])
            row[f"oob_{metric}_mean"] = float(group[column].mean())
            row[f"oob_{metric}_median"] = float(group[column].median())
            row[f"oob_{metric}_ci_low"] = low
            row[f"oob_{metric}_ci_high"] = high
            row[f"oob_{metric}_valid_replicates"] = int(
                group[column].notna().sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    output: Path,
    apparent_auc: float,
    auc_summary: pd.DataFrame,
    cutpoint_summary: pd.DataFrame,
    strata: pd.DataFrame,
) -> None:
    auc = auc_summary.iloc[0]
    lines = [
        "# Bootstrap .632+ AUC and OOB cutpoint validation",
        "",
        "Cohort: 195 patients; 36 CR-POPF events.",
        "Predictors: locked seven-feature radiomics panel.",
        "Model: standardized, unweighted elastic-net logistic regression.",
        "Post-hoc calibration: none.",
        "",
        f"Apparent AUC: {apparent_auc:.3f}.",
        f"Bootstrap .632+ AUC: {auc['auc']:.3f} "
        f"({auc['ci_low']:.3f}-{auc['ci_high']:.3f}).",
        "",
        "Cutpoints were selected in bag using constrained MCC and evaluated "
        "unchanged out of bag.",
        "",
    ]
    for row in cutpoint_summary.itertuples(index=False):
        lines.append(
            f"- {row.strategy}: full-cohort threshold "
            f"{row.full_cohort_threshold:.6f}; mean OOB sensitivity "
            f"{row.oob_sensitivity_rule_out_mean:.3f}; mean OOB specificity "
            f"{row.oob_specificity_rule_in_mean:.3f}; mean OOB MCC "
            f"{row.oob_mcc_mean:.3f}."
        )
    lines.extend(["", "Full-cohort risk strata:", ""])
    for row in strata.itertuples(index=False):
        lines.append(
            f"- {row.stratum}: {row.n} patients, {row.events} events "
            f"({100 * row.event_rate:.1f}%)."
        )
    (output / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 2000:
        raise ValueError("At least 2,000 bootstrap replicates are required")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix, outcome = load_data(args.data)
    full_cutpoints, full_probability, apparent_auc = full_cohort_analysis(
        matrix, outcome
    )
    strata = summarize_risk_strata(
        full_probability, outcome, full_cutpoints
    )

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    auc_rows: list[dict[str, Any]] = []
    cutpoint_rows: list[dict[str, Any]] = []
    attempts = 0
    while len(auc_rows) < args.n_bootstrap:
        in_bag, out_of_bag = stratified_bootstrap_indices(outcome, rng)
        iteration = attempts
        attempts += 1
        if (
            len(out_of_bag) < 10
            or np.unique(outcome[in_bag]).size != 2
            or np.unique(outcome[out_of_bag]).size != 2
        ):
            if attempts > args.n_bootstrap * 2:
                raise RuntimeError("Too many invalid bootstrap samples")
            continue
        estimator = make_model(MODEL_SEED + iteration + 1000)
        estimator.fit(matrix[in_bag], outcome[in_bag])
        probability_in_bag = estimator.predict_proba(matrix[in_bag])[:, 1]
        probability_oob = estimator.predict_proba(matrix[out_of_bag])[:, 1]
        auc_rows.append(
            {
                **auc_632plus(
                    outcome[in_bag],
                    probability_in_bag,
                    outcome[out_of_bag],
                    probability_oob,
                ),
                "iteration": iteration,
                "n_in_bag": int(len(in_bag)),
                "unique_in_bag": int(len(np.unique(in_bag))),
                "n_oob": int(len(out_of_bag)),
                "events_oob": int(outcome[out_of_bag].sum()),
            }
        )
        for mode, label in (("rule_out", "Rule-out"), ("rule_in", "Rule-in")):
            selected = choose_threshold(
                outcome[in_bag], probability_in_bag, mode
            )
            evaluated = threshold_metrics(
                outcome[out_of_bag],
                probability_oob,
                float(selected["threshold"]),
            )
            cutpoint_rows.append(
                {
                    "iteration": iteration,
                    "strategy": label,
                    "constraint": (
                        "in-bag sensitivity >=90%; maximize MCC"
                        if mode == "rule_out"
                        else "in-bag specificity >=90%; maximize MCC"
                    ),
                    "selected_threshold": float(selected["threshold"]),
                    "in_bag_sensitivity": float(
                        selected["sensitivity_rule_out"]
                    ),
                    "in_bag_specificity": float(
                        selected["specificity_rule_in"]
                    ),
                    "in_bag_mcc": float(selected["mcc"]),
                    "n_oob": int(len(out_of_bag)),
                    "events_oob": int(outcome[out_of_bag].sum()),
                    **{
                        f"oob_{key}": value
                        for key, value in evaluated.items()
                    },
                }
            )

    auc_replicates = pd.DataFrame(auc_rows)
    cutpoint_replicates = pd.DataFrame(cutpoint_rows)
    auc_low, auc_high = percentile_interval(
        auc_replicates["auc_632plus"]
    )
    auc_summary = pd.DataFrame(
        [
            {
                "method": "stratified bootstrap .632+",
                "replicates": int(len(auc_replicates)),
                "auc": float(auc_replicates["auc_632plus"].mean()),
                "ci_low": auc_low,
                "ci_high": auc_high,
                "mean_oob_auc": float(auc_replicates["auc_oob"].mean()),
                "median_oob_n": float(auc_replicates["n_oob"].median()),
                "median_oob_events": float(
                    auc_replicates["events_oob"].median()
                ),
            }
        ]
    )
    cutpoint_summary = summarize_cutpoints(
        full_cutpoints, cutpoint_replicates
    )

    full_cutpoints.to_csv(
        args.output / "full_cohort_cutpoints.csv", index=False
    )
    strata.to_csv(args.output / "full_cohort_risk_strata.csv", index=False)
    auc_summary.to_csv(args.output / "auc_632plus_summary.csv", index=False)
    auc_replicates.to_csv(
        args.output / "auc_632plus_replicates.csv", index=False
    )
    cutpoint_summary.to_csv(
        args.output / "bootstrap_oob_cutpoint_summary.csv", index=False
    )
    cutpoint_replicates.to_csv(
        args.output / "bootstrap_oob_cutpoint_replicates.csv", index=False
    )
    config = {
        "cohort_n": int(len(outcome)),
        "events": int(outcome.sum()),
        "features": FEATURES,
        "model": "unweighted elastic-net logistic regression",
        "model_parameters": MODEL_PARAMS,
        "scaling": "StandardScaler refitted in every bootstrap sample",
        "calibration": "none",
        "bootstrap": "class-stratified nonparametric bootstrap",
        "requested_valid_replicates": int(args.n_bootstrap),
        "valid_replicates": int(len(auc_replicates)),
        "attempts": int(attempts),
        "seed": BOOTSTRAP_SEED,
    }
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(
        args.output,
        apparent_auc,
        auc_summary,
        cutpoint_summary,
        strata,
    )
    print(auc_summary.to_string(index=False))
    print(cutpoint_summary.to_string(index=False))
    print(strata.to_string(index=False))
    print(f"\nWrote outputs to {args.output}")


if __name__ == "__main__":
    main()
