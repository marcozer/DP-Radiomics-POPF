#!/usr/bin/env python3
"""Screen locked-panel estimators with paired stratified bootstrap .632+.

The seven radiomics predictors are fixed. Hyperparameters are tuned once on
the full cohort, then each estimator is refitted on the same class-stratified
bootstrap samples. The .632+ estimate combines in-bag and out-of-bag AUCs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ANALYSIS_ROOT / "data_anonymized/radiomics_features_anonymized.csv"
DEFAULT_OUTPUT = ANALYSIS_ROOT / "results/locked_panel_candidate_632plus"
MODEL_SEED = 20260616
BOOTSTRAP_SEED = 20260721

FEATURES = [
    "log-sigma-3-0-mm-3D_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_glcm_ClusterShade",
    "log-sigma-3-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis",
    "log-sigma-7-0-mm-3D_ngtdm_Strength",
    "original_shape_MinorAxisLength",
    "wavelet-HLH_firstorder_Median",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
]

MODEL_SHORT = {
    "Elastic-net logistic regression": "EN",
    "L2 logistic regression": "L2-LR",
    "Support vector machine": "SVM",
    "Random forest": "RF",
    "LightGBM": "LGB",
    "XGBoost": "XGB",
}

NORD = {
    "ink": "#2E3440",
    "slate": "#4C566A",
    "grid": "#D8DEE9",
    "blue": "#5E81AC",
    "cyan": "#88C0D0",
    "green": "#A3BE8C",
    "yellow": "#EBCB8B",
    "orange": "#D08770",
    "red": "#BF616A",
    "paper": "#FFFFFF",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 18,
            "legend.fontsize": 11,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "svg.fonttype": "none",
            "text.color": NORD["ink"],
            "axes.labelcolor": NORD["ink"],
            "axes.edgecolor": NORD["slate"],
            "xtick.color": NORD["ink"],
            "ytick.color": NORD["ink"],
            "figure.facecolor": NORD["paper"],
            "axes.facecolor": NORD["paper"],
        }
    )


def style_axis(axis: plt.Axes) -> None:
    axis.grid(color=NORD["grid"], alpha=0.55, linewidth=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color(NORD["slate"])
        spine.set_linewidth(1.2)


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


def make_elastic_net(params: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=float(params["C"]),
                    l1_ratio=float(params["l1_ratio"]),
                    class_weight=None,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=MODEL_SEED,
                ),
            ),
        ]
    )


def candidate_factories() -> dict[
    str, tuple[Callable[[dict[str, Any]], BaseEstimator], list[dict[str, Any]]]
]:
    def scaled(estimator: BaseEstimator) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    return {
        "Elastic-net logistic regression": (
            make_elastic_net,
            list(
                ParameterGrid(
                    {
                        "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
                        "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
                    }
                )
            ),
        ),
        "L2 logistic regression": (
            lambda params: scaled(
                LogisticRegression(
                    penalty="l2",
                    C=float(params["C"]),
                    solver="lbfgs",
                    class_weight=None,
                    max_iter=10000,
                    random_state=MODEL_SEED,
                )
            ),
            list(ParameterGrid({"C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]})),
        ),
        "Support vector machine": (
            lambda params: scaled(
                SVC(
                    C=float(params["C"]),
                    gamma=params["gamma"],
                    kernel="rbf",
                    probability=True,
                    class_weight=None,
                    random_state=MODEL_SEED,
                )
            ),
            list(
                ParameterGrid(
                    {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1, 1.0]}
                )
            ),
        ),
        "Random forest": (
            lambda params: RandomForestClassifier(
                n_estimators=300,
                max_depth=params["max_depth"],
                min_samples_leaf=int(params["min_samples_leaf"]),
                class_weight=None,
                random_state=MODEL_SEED,
                n_jobs=1,
            ),
            list(
                ParameterGrid(
                    {
                        "max_depth": [2, 3, 5, None],
                        "min_samples_leaf": [2, 5, 10],
                    }
                )
            ),
        ),
        "XGBoost": (
            lambda params: xgb.XGBClassifier(
                n_estimators=200,
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=MODEL_SEED,
                n_jobs=1,
            ),
            list(
                ParameterGrid(
                    {"max_depth": [1, 2, 3], "learning_rate": [0.03, 0.1]}
                )
            ),
        ),
        "LightGBM": (
            lambda params: lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=int(params["num_leaves"]),
                min_child_samples=int(params["min_child_samples"]),
                learning_rate=0.05,
                verbosity=-1,
                random_state=MODEL_SEED,
                n_jobs=1,
            ),
            list(
                ParameterGrid(
                    {
                        "num_leaves": [3, 7, 15],
                        "min_child_samples": [10, 20, 30],
                    }
                )
            ),
        ),
    }


def tune_candidate(
    factory: Callable[[dict[str, Any]], BaseEstimator],
    grid: Iterable[dict[str, Any]],
    matrix: np.ndarray,
    outcome: np.ndarray,
) -> tuple[BaseEstimator, dict[str, Any], float]:
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=MODEL_SEED)
    best_auc = -math.inf
    best_params: dict[str, Any] | None = None
    for params in grid:
        scores: list[float] = []
        for train, test in splitter.split(matrix, outcome):
            estimator = factory(params)
            estimator.fit(matrix[train], outcome[train])
            scores.append(
                float(
                    roc_auc_score(
                        outcome[test], estimator.predict_proba(matrix[test])[:, 1]
                    )
                )
            )
        score = float(np.mean(scores))
        if score > best_auc:
            best_auc = score
            best_params = dict(params)
    if best_params is None:
        raise RuntimeError("Candidate-model tuning failed")
    estimator = factory(best_params)
    estimator.fit(matrix, outcome)
    return estimator, best_params, best_auc


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


def build_paired_samples(
    outcome: np.ndarray, n_bootstrap: int
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[tuple[int, np.ndarray, np.ndarray]] = []
    attempt = 0
    while len(samples) < n_bootstrap:
        in_bag, out_of_bag = stratified_bootstrap_indices(outcome, rng)
        if (
            len(out_of_bag) >= 10
            and np.unique(outcome[in_bag]).size == 2
            and np.unique(outcome[out_of_bag]).size == 2
        ):
            samples.append((attempt, in_bag, out_of_bag))
        attempt += 1
        if attempt > n_bootstrap * 2:
            raise RuntimeError("Too many invalid bootstrap samples")
    return samples


def set_iteration_seed(estimator: BaseEstimator, seed: int) -> BaseEstimator:
    available = estimator.get_params(deep=True)
    updates = {
        key: seed
        for key in ("random_state", "model__random_state")
        if key in available
    }
    if updates:
        estimator.set_params(**updates)
    return estimator


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


def evaluate_model(
    label: str,
    factory: Callable[[dict[str, Any]], BaseEstimator],
    grid: list[dict[str, Any]],
    matrix: np.ndarray,
    outcome: np.ndarray,
    samples: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    estimator, params, tuning_auc = tune_candidate(factory, grid, matrix, outcome)
    apparent_probability = estimator.predict_proba(matrix)[:, 1]
    rows: list[dict[str, Any]] = []
    for replicate, (attempt, in_bag, out_of_bag) in enumerate(samples):
        fitted = set_iteration_seed(
            clone(estimator), MODEL_SEED + attempt + 1000
        )
        fitted.fit(matrix[in_bag], outcome[in_bag])
        rows.append(
            {
                "model": label,
                "replicate": replicate,
                "attempt": attempt,
                "n_oob": int(len(out_of_bag)),
                "events_oob": int(outcome[out_of_bag].sum()),
                **auc_632plus(
                    outcome[in_bag],
                    fitted.predict_proba(matrix[in_bag])[:, 1],
                    outcome[out_of_bag],
                    fitted.predict_proba(matrix[out_of_bag])[:, 1],
                ),
            }
        )
    replicates = pd.DataFrame(rows)
    low, high = np.quantile(replicates["auc_632plus"], [0.025, 0.975])
    summary = {
        "model": label,
        "probability_source": (
            "apparent ROC from full-cohort fit; AUC from paired "
            "class-stratified bootstrap .632+"
        ),
        "apparent_auc": float(roc_auc_score(outcome, apparent_probability)),
        "inner_cv_auc_for_tuning": float(tuning_auc),
        "auc_632plus": float(replicates["auc_632plus"].mean()),
        "auc_632plus_ci_low": float(low),
        "auc_632plus_ci_high": float(high),
        "mean_oob_auc": float(replicates["auc_oob"].mean()),
        "valid_replicates": int(len(replicates)),
        "hyperparameters": json.dumps(params, sort_keys=True),
    }
    return summary, replicates, apparent_probability


def paired_comparisons(replicates: pd.DataFrame) -> pd.DataFrame:
    pivot = replicates.pivot(
        index="replicate", columns="model", values="auc_632plus"
    )
    reference = "Elastic-net logistic regression"
    rows: list[dict[str, Any]] = []
    for model in pivot.columns:
        if model == reference:
            continue
        differences = pivot[model] - pivot[reference]
        low, high = np.quantile(differences, [0.025, 0.975])
        rows.append(
            {
                "comparison": f"{model} minus {reference}",
                "mean_delta_auc_632plus": float(differences.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "paired_replicates": int(differences.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_figure(
    metrics: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    outcome: np.ndarray,
    output: Path,
) -> None:
    configure_style()
    indexed = metrics.set_index("model")
    remaining = (
        metrics.loc[
            ~metrics["model"].eq("Elastic-net logistic regression")
        ]
        .sort_values("auc_632plus", ascending=False)["model"]
        .tolist()
    )
    ordered = ["Elastic-net logistic regression", *remaining]
    colors = [
        NORD["blue"],
        NORD["cyan"],
        NORD["green"],
        NORD["yellow"],
        NORD["orange"],
        NORD["red"],
    ]

    figure, (estimate_axis, roc_axis) = plt.subplots(
        1, 2, figsize=(15.5, 7.8), gridspec_kw={"width_ratios": [0.95, 1.25]}
    )
    positions = np.arange(len(ordered))[::-1]
    values = indexed.loc[ordered, "auc_632plus"].to_numpy(float)
    low = indexed.loc[ordered, "auc_632plus_ci_low"].to_numpy(float)
    high = indexed.loc[ordered, "auc_632plus_ci_high"].to_numpy(float)
    estimate_axis.errorbar(
        values,
        positions,
        xerr=np.vstack([values - low, high - values]),
        fmt="o",
        markersize=9,
        capsize=4,
        color=NORD["blue"],
        ecolor=NORD["slate"],
        linewidth=1.8,
    )
    estimate_axis.axvline(0.5, color=NORD["red"], linestyle="--", linewidth=1.2)
    estimate_axis.set_yticks(positions)
    estimate_axis.set_yticklabels([MODEL_SHORT[label] for label in ordered])
    estimate_axis.set_xlim(0.48, 0.91)
    estimate_axis.set_xlabel("Bootstrap .632+ AUC (95% CI)")
    estimate_axis.set_title("Locked-panel model screening")
    for y_pos, value in zip(positions, values):
        estimate_axis.text(
            min(value + 0.012, 0.87),
            y_pos,
            f"{value:.3f}",
            va="center",
            fontsize=12,
            color=NORD["ink"],
        )
    style_axis(estimate_axis)

    roc_axis.plot(
        [0, 1], [0, 1], color=NORD["grid"], linestyle="--", linewidth=1.3
    )
    for label, color in zip(ordered, colors):
        false_positive, true_positive, _ = roc_curve(outcome, probabilities[label])
        apparent = float(indexed.loc[label, "apparent_auc"])
        corrected = float(indexed.loc[label, "auc_632plus"])
        roc_axis.step(
            false_positive,
            true_positive,
            where="post",
            color=color,
            linewidth=2.2,
            label=(
                f"{MODEL_SHORT[label]}: apparent {apparent:.3f}; "
                f".632+ {corrected:.3f}"
            ),
        )
    roc_axis.set(
        title="Apparent full-cohort ROC curves",
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    roc_axis.legend(loc="lower right", frameon=False, fontsize=10)
    style_axis(roc_axis)
    figure.tight_layout()
    figure.savefig(
        output / "figure4_locked_panel_model_screening.svg",
        bbox_inches="tight",
        facecolor=NORD["paper"],
    )
    plt.close(figure)


def write_report(
    output: Path,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    n_bootstrap: int,
) -> None:
    lines = [
        "# Locked-panel model-family screening",
        "",
        "Cohort: 195 patients; 36 CR-POPF events.",
        f"Paired class-stratified bootstrap replicates: {n_bootstrap}.",
        "Scaling and estimator coefficients were refitted in every replicate.",
        "Hyperparameters were tuned once on the full cohort and then held fixed.",
        "",
        "## Model estimates",
        "",
    ]
    for row in metrics.itertuples(index=False):
        lines.append(
            f"- {row.model}: apparent AUC {row.apparent_auc:.3f}; "
            f".632+ AUC {row.auc_632plus:.3f} "
            f"({row.auc_632plus_ci_low:.3f}-{row.auc_632plus_ci_high:.3f})."
        )
    lines.extend(["", "## Paired differences versus elastic net", ""])
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"- {row.comparison}: delta {row.mean_delta_auc_632plus:.3f} "
            f"({row.ci_low:.3f}-{row.ci_high:.3f})."
        )
    (output / "analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_bootstrap < 2000:
        raise ValueError("At least 2,000 bootstrap replicates are required")
    args.output.mkdir(parents=True, exist_ok=True)
    matrix, outcome = load_data(args.data)
    samples = build_paired_samples(outcome, args.n_bootstrap)
    factories = candidate_factories()
    results = Parallel(
        n_jobs=min(args.jobs, len(factories)), prefer="processes"
    )(
        delayed(evaluate_model)(
            label, factory, grid, matrix, outcome, samples
        )
        for label, (factory, grid) in factories.items()
    )
    metrics = pd.DataFrame([summary for summary, _, _ in results])
    replicates = pd.concat([frame for _, frame, _ in results], ignore_index=True)
    probabilities = {
        label: probability
        for label, (_, _, probability) in zip(factories, results)
    }
    comparisons = paired_comparisons(replicates)

    metrics.to_csv(args.output / "candidate_model_metrics.csv", index=False)
    replicates.to_csv(
        args.output / "candidate_model_632plus_replicates.csv", index=False
    )
    comparisons.to_csv(
        args.output / "paired_model_comparisons.csv", index=False
    )
    config = {
        "cohort_n": int(len(outcome)),
        "events": int(outcome.sum()),
        "features": FEATURES,
        "bootstrap_method": "paired class-stratified nonparametric bootstrap",
        "bootstrap_replicates": int(args.n_bootstrap),
        "auc_estimator": ".632+",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "model_tuning": "3-fold stratified CV once on the full cohort",
        "refit_per_replicate": True,
    }
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output, metrics, comparisons, args.n_bootstrap)
    plot_figure(metrics, probabilities, outcome, args.output)
    print(metrics.to_string(index=False))
    print(f"\nWrote outputs to {args.output}")


if __name__ == "__main__":
    main()
