#!/usr/bin/env python3
"""Reproduce the R0_v3 apparent radiomics and radioclinical comparison.

All elastic-net models are refitted on the full cohort. Their apparent AUCs
are shown with simple-bootstrap confidence intervals and compared by paired
DeLong tests. Published DP-FRS and 2025 DISPAIR equations are not refitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import ParameterGrid, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ANALYSIS_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RADIOMICS = (
    ANALYSIS_ROOT / "data_anonymized/radiomics_features_anonymized.csv"
)
DEFAULT_CLINICAL = (
    ANALYSIS_ROOT / "data_anonymized/model_covariates_anonymized.csv"
)
DEFAULT_OUTPUT = ANALYSIS_ROOT / "results/r0_v3_apparent_model_comparison"
MODEL_SEED = 20260616
BOOTSTRAP_SEED = 20260715

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from statistical_tests import delong_test  # noqa: E402


RADIOMICS_FEATURES = [
    "log-sigma-3-0-mm-3D_glcm_ClusterProminence",
    "log-sigma-3-0-mm-3D_glcm_ClusterShade",
    "log-sigma-3-0-mm-3D_gldm_SmallDependenceHighGrayLevelEmphasis",
    "log-sigma-7-0-mm-3D_ngtdm_Strength",
    "original_shape_MinorAxisLength",
    "wavelet-HLH_firstorder_Median",
    "wavelet-HLH_gldm_LargeDependenceLowGrayLevelEmphasis",
]

NORD = {
    "ink": "#2E3440",
    "slate": "#4C566A",
    "grid": "#D8DEE9",
    "blue": "#5E81AC",
    "green": "#A3BE8C",
    "orange": "#D08770",
    "red": "#BF616A",
    "purple": "#B48EAD",
    "paper": "#FFFFFF",
}

MODEL_COLORS = {
    "7-rad": NORD["blue"],
    "7-rad + MPD/thickness": NORD["orange"],
    "7-rad + DISPAIR-FRS 2025 covariates": NORD["purple"],
    "Preoperative DP-FRS (published)": NORD["green"],
    "DISPAIR-FRS 2025 (published; all-neck)": NORD["red"],
}


@dataclass(frozen=True)
class FittedModel:
    slug: str
    label: str
    features: list[str]
    estimator: Pipeline
    probability: np.ndarray
    tuning_auc: float
    parameters: dict[str, float]


def male_indicator(value: Any) -> float:
    text = str(value).strip().lower()
    if text in {"m", "male", "man", "homme", "1", "1.0"}:
        return 1.0
    if text in {"f", "female", "woman", "femme", "0", "0.0"}:
        return 0.0
    raise ValueError(f"Unrecognized sex value: {value!r}")


def truncated_cube(value: float, knot: float) -> float:
    return max(float(value) - knot, 0.0) ** 3


def load_data(radiomics_path: Path, clinical_path: Path) -> pd.DataFrame:
    radiomics = pd.read_csv(radiomics_path)
    clinical = pd.read_csv(clinical_path)
    for name, frame in (("radiomics", radiomics), ("clinical", clinical)):
        if "patient_id" not in frame or "cr_popf" not in frame:
            raise ValueError(f"{name} input lacks patient_id or cr_popf")
        if not frame["patient_id"].astype(str).str.fullmatch(r"RDP_\d{3}").all():
            raise ValueError(f"{name} input must contain only RDP_### identifiers")
        if frame["patient_id"].duplicated().any():
            raise ValueError(f"{name} input contains duplicate identifiers")
    clinical_columns = [
        "patient_id",
        "cr_popf",
        "age",
        "sex",
        "mpd_diameter",
        "neck_thickness",
    ]
    frame = radiomics[["patient_id", "cr_popf", *RADIOMICS_FEATURES]].merge(
        clinical[clinical_columns],
        on="patient_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_radiomics", "_clinical"),
    )
    if not np.array_equal(
        frame["cr_popf_radiomics"].to_numpy(int),
        frame["cr_popf_clinical"].to_numpy(int),
    ):
        raise ValueError("Outcome mismatch between anonymized inputs")
    frame = frame.rename(columns={"cr_popf_radiomics": "cr_popf"}).drop(
        columns="cr_popf_clinical"
    )
    numeric = [
        *RADIOMICS_FEATURES,
        "age",
        "mpd_diameter",
        "neck_thickness",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if len(frame) != 195 or int(frame["cr_popf"].sum()) != 36:
        raise ValueError(
            f"Expected 195 patients and 36 events; found {len(frame)} and "
            f"{int(frame['cr_popf'].sum())}"
        )
    if frame[numeric].isna().any().any():
        raise ValueError("Model predictors must be complete")

    frame["male_indicator"] = frame["sex"].map(male_indicator)
    for knot in (41.0, 65.0, 77.0):
        frame[f"age_spline_{int(knot)}"] = frame["age"].map(
            lambda value, knot=knot: truncated_cube(value, knot)
        )
    frame["dpfrs_published_probability"] = expit(
        -4.211
        + 0.388 * frame["mpd_diameter"]
        + 0.131 * frame["neck_thickness"]
    )
    frame["dispair_2025_published_probability"] = expit(
        -3.587
        + 0.115 * frame["neck_thickness"]
        + 0.000932 * frame["age"]
        - 1.316e-5 * frame["age_spline_41"]
        + 3.948e-5 * frame["age_spline_65"]
        - 2.632e-5 * frame["age_spline_77"]
        + 0.894
        - 0.0407 * frame["mpd_diameter"]
        + 0.230 * frame["male_indicator"]
    )
    return frame


def make_model(parameters: dict[str, float]) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=float(parameters["C"]),
                    l1_ratio=float(parameters["l1_ratio"]),
                    class_weight=None,
                    max_iter=30000,
                    tol=1e-4,
                    random_state=MODEL_SEED,
                ),
            ),
        ]
    )


def tune_model(
    matrix: np.ndarray, outcome: np.ndarray
) -> tuple[dict[str, float], float]:
    grid = ParameterGrid(
        {
            "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            "l1_ratio": [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
        }
    )
    splitter = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=MODEL_SEED
    )
    best_auc = -math.inf
    best: dict[str, float] | None = None
    for parameters in grid:
        scores = []
        for train, test in splitter.split(matrix, outcome):
            estimator = make_model(parameters)
            estimator.fit(matrix[train], outcome[train])
            scores.append(
                roc_auc_score(
                    outcome[test],
                    estimator.predict_proba(matrix[test])[:, 1],
                )
            )
        score = float(np.mean(scores))
        if score > best_auc:
            best_auc = score
            best = {key: float(value) for key, value in parameters.items()}
    if best is None:
        raise RuntimeError("Elastic-net tuning failed")
    return best, best_auc


def fit_models(frame: pd.DataFrame) -> list[FittedModel]:
    outcome = frame["cr_popf"].to_numpy(int)
    dispair_covariates = [
        "neck_thickness",
        "mpd_diameter",
        "age",
        "age_spline_41",
        "age_spline_65",
        "age_spline_77",
        "male_indicator",
    ]
    specifications = [
        ("radiomics_7rad", "7-rad", RADIOMICS_FEATURES),
        (
            "radioclinical_dpfrs",
            "7-rad + MPD/thickness",
            [*RADIOMICS_FEATURES, "mpd_diameter", "neck_thickness"],
        ),
        (
            "radioclinical_dispair_2025",
            "7-rad + DISPAIR-FRS 2025 covariates",
            [*RADIOMICS_FEATURES, *dispair_covariates],
        ),
        (
            "dpfrs_covariates_refit",
            "DP-FRS covariates refit",
            ["mpd_diameter", "neck_thickness"],
        ),
        (
            "dispair_covariates_refit",
            "DISPAIR-FRS 2025 covariates refit",
            dispair_covariates,
        ),
    ]
    fitted: list[FittedModel] = []
    for slug, label, features in specifications:
        matrix = frame[features].to_numpy(float)
        parameters, tuning_auc = tune_model(matrix, outcome)
        estimator = make_model(parameters)
        estimator.fit(matrix, outcome)
        fitted.append(
            FittedModel(
                slug=slug,
                label=label,
                features=list(features),
                estimator=estimator,
                probability=estimator.predict_proba(matrix)[:, 1],
                tuning_auc=tuning_auc,
                parameters=parameters,
            )
        )
    return fitted


def bootstrap_auc_interval(
    outcome: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sample = rng.integers(0, len(outcome), len(outcome))
        if np.unique(outcome[sample]).size == 2:
            values.append(
                roc_auc_score(outcome[sample], probability[sample])
            )
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())


def calibration_parameters(
    outcome: np.ndarray, probability: np.ndarray
) -> tuple[float, float]:
    predictor = logit(np.clip(probability, 1e-8, 1 - 1e-8)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
    model.fit(predictor, outcome)
    return float(model.intercept_[0]), float(model.coef_[0, 0])


def calibration_bins(
    outcome: np.ndarray, probability: np.ndarray, n_bins: int = 6
) -> pd.DataFrame:
    frame = pd.DataFrame({"outcome": outcome, "probability": probability})
    frame["bin"] = pd.qcut(
        frame["probability"], q=n_bins, labels=False, duplicates="drop"
    )
    rows = []
    for bin_number, group in frame.groupby("bin", observed=True):
        count = int(len(group))
        events = int(group["outcome"].sum())
        observed = events / count
        z = 1.959963984540054
        denominator = 1 + z**2 / count
        center = (observed + z**2 / (2 * count)) / denominator
        half = (
            z
            * math.sqrt(
                observed * (1 - observed) / count
                + z**2 / (4 * count**2)
            )
            / denominator
        )
        rows.append(
            {
                "bin": int(bin_number),
                "n": count,
                "events": events,
                "mean_predicted": float(group["probability"].mean()),
                "observed": observed,
                "observed_ci_low": max(0.0, center - half),
                "observed_ci_high": min(1.0, center + half),
            }
        )
    return pd.DataFrame(rows)


def coefficient_table(models: Sequence[FittedModel]) -> pd.DataFrame:
    rows = []
    for result in models:
        scaler: StandardScaler = result.estimator.named_steps["scaler"]
        estimator: LogisticRegression = result.estimator.named_steps["model"]
        rows.append(
            {
                "model": result.label,
                "term": "Intercept",
                "coefficient_standardized": float(estimator.intercept_[0]),
                "predictor_mean": math.nan,
                "predictor_sd": math.nan,
            }
        )
        for feature, coefficient, mean, scale in zip(
            result.features,
            estimator.coef_[0],
            scaler.mean_,
            scaler.scale_,
        ):
            rows.append(
                {
                    "model": result.label,
                    "term": feature,
                    "coefficient_standardized": float(coefficient),
                    "predictor_mean": float(mean),
                    "predictor_sd": float(scale),
                }
            )
    return pd.DataFrame(rows)


def summarize(
    frame: pd.DataFrame,
    fitted: Sequence[FittedModel],
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcome = frame["cr_popf"].to_numpy(int)
    main_labels = {
        "7-rad",
        "7-rad + MPD/thickness",
        "7-rad + DISPAIR-FRS 2025 covariates",
    }
    predictions = {
        result.label: result.probability
        for result in fitted
        if result.label in main_labels
    }
    predictions["Preoperative DP-FRS (published)"] = frame[
        "dpfrs_published_probability"
    ].to_numpy(float)
    predictions["DISPAIR-FRS 2025 (published; all-neck)"] = frame[
        "dispair_2025_published_probability"
    ].to_numpy(float)
    sources = {
        "7-rad": "full-cohort elastic-net refit",
        "7-rad + MPD/thickness": "full-cohort elastic-net refit",
        "7-rad + DISPAIR-FRS 2025 covariates": (
            "full-cohort elastic-net refit under an all-neck assumption"
        ),
        "Preoperative DP-FRS (published)": "published equation, no refit",
        "DISPAIR-FRS 2025 (published; all-neck)": (
            "BJS 2025 research equation, all-neck sensitivity, no refit"
        ),
    }
    metric_rows = []
    bin_frames = []
    for position, (label, probability) in enumerate(predictions.items()):
        low, high = bootstrap_auc_interval(
            outcome,
            probability,
            n_bootstrap,
            BOOTSTRAP_SEED + 100 + position,
        )
        intercept, slope = calibration_parameters(outcome, probability)
        metric_rows.append(
            {
                "model": label,
                "probability_source": sources[label],
                "n": int(len(outcome)),
                "events": int(outcome.sum()),
                "auc_apparent": float(roc_auc_score(outcome, probability)),
                "auc_ci_low": low,
                "auc_ci_high": high,
                "calibration_intercept": intercept,
                "calibration_slope": slope,
                "mean_predicted_risk": float(probability.mean()),
                "observed_risk": float(outcome.mean()),
            }
        )
        bin_frames.append(
            calibration_bins(outcome, probability).assign(model=label)
        )

    comparisons = []
    labels = list(predictions)
    for index, reference in enumerate(labels):
        for alternative in labels[index + 1 :]:
            result = delong_test(
                outcome, predictions[reference], predictions[alternative]
            )
            comparisons.append(
                {
                    "reference": reference,
                    "alternative": alternative,
                    "delta_auc_alternative_minus_reference": float(
                        roc_auc_score(outcome, predictions[alternative])
                        - roc_auc_score(outcome, predictions[reference])
                    ),
                    "delong_z": float(result["z"]),
                    "delong_p": float(result["p_value"]),
                }
            )
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(comparisons),
        pd.concat(bin_frames, ignore_index=True),
    )


def plot_comparison(
    frame: pd.DataFrame,
    fitted: Sequence[FittedModel],
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    output: Path,
) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "legend.fontsize": 10,
            "svg.fonttype": "none",
            "text.color": NORD["ink"],
            "axes.labelcolor": NORD["ink"],
            "axes.edgecolor": NORD["slate"],
        }
    )
    outcome = frame["cr_popf"].to_numpy(int)
    main_labels = {
        "7-rad",
        "7-rad + MPD/thickness",
        "7-rad + DISPAIR-FRS 2025 covariates",
    }
    predictions = {
        result.label: result.probability
        for result in fitted
        if result.label in main_labels
    }
    predictions["Preoperative DP-FRS (published)"] = frame[
        "dpfrs_published_probability"
    ].to_numpy(float)
    predictions["DISPAIR-FRS 2025 (published; all-neck)"] = frame[
        "dispair_2025_published_probability"
    ].to_numpy(float)
    metric_map = metrics.set_index("model")

    figure, (roc_axis, calibration_axis) = plt.subplots(
        1, 2, figsize=(16.5, 7.2)
    )
    for label, probability in predictions.items():
        false_positive, true_positive, _ = roc_curve(outcome, probability)
        row = metric_map.loc[label]
        roc_axis.plot(
            false_positive,
            true_positive,
            color=MODEL_COLORS[label],
            linewidth=2.5,
            label=(
                f"{label}: {row['auc_apparent']:.3f} "
                f"[{row['auc_ci_low']:.3f}-{row['auc_ci_high']:.3f}]"
            ),
        )
    roc_axis.plot(
        [0, 1], [0, 1], linestyle="--", color=NORD["grid"], linewidth=1.4
    )
    roc_axis.set(
        title="A. Apparent discrimination",
        xlabel="1 - Specificity",
        ylabel="Sensitivity",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    roc_axis.legend(loc="lower right", frameon=False, fontsize=9)

    calibration_axis.plot(
        [0, 1], [0, 1], linestyle="--", color=NORD["slate"], linewidth=1.4
    )
    for result in fitted:
        if result.label not in main_labels:
            continue
        points = bins.loc[bins["model"].eq(result.label)]
        calibration_axis.errorbar(
            points["mean_predicted"],
            points["observed"],
            yerr=[
                points["observed"] - points["observed_ci_low"],
                points["observed_ci_high"] - points["observed"],
            ],
            color=MODEL_COLORS[result.label],
            marker="o",
            linewidth=2.0,
            capsize=3,
            label=result.label,
        )
    calibration_axis.set(
        title="B. Apparent fitted-model calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed CR-POPF rate",
        xlim=(-0.02, 0.72),
        ylim=(-0.02, 0.82),
    )
    calibration_axis.legend(loc="upper left", frameon=False)
    for axis in (roc_axis, calibration_axis):
        axis.grid(color=NORD["grid"], alpha=0.55, linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        output / "figure6_apparent_roc_and_fitted_model_calibration.svg",
        bbox_inches="tight",
        facecolor=NORD["paper"],
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics", type=Path, default=DEFAULT_RADIOMICS)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--auc-bootstrap", type=int, default=5000)
    parser.add_argument(
        "--export-model",
        type=Path,
        default=None,
        help="Optional path for the non-patient-level deployable 7-rad bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = load_data(args.radiomics, args.clinical)
    fitted = fit_models(frame)
    metrics, comparisons, bins = summarize(
        frame, fitted, args.auc_bootstrap
    )
    tuning = pd.DataFrame(
        [
            {
                "slug": result.slug,
                "model": result.label,
                "n_features": len(result.features),
                "inner_cv_auc": result.tuning_auc,
                **result.parameters,
            }
            for result in fitted
        ]
    )
    metrics.to_csv(args.output / "main_apparent_model_metrics.csv", index=False)
    comparisons.to_csv(args.output / "main_pairwise_delong.csv", index=False)
    bins.to_csv(args.output / "main_calibration_points.csv", index=False)
    coefficient_table(fitted).to_csv(
        args.output / "elasticnet_coefficients_intercepts.csv", index=False
    )
    tuning.to_csv(args.output / "elasticnet_tuning.csv", index=False)
    config = {
        "cohort_n": int(len(frame)),
        "events": int(frame["cr_popf"].sum()),
        "radiomics_features": RADIOMICS_FEATURES,
        "estimator": "standardized unweighted elastic-net logistic regression",
        "apparent_auc_ci": (
            f"simple nonparametric bootstrap; {args.auc_bootstrap} resamples"
        ),
        "paired_auc_comparison": "DeLong",
        "dispair_equation": "BJS 2025 research equation; all-neck assumption",
        "patient_level_outputs_written": False,
    }
    (args.output / "analysis_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    if args.export_model is not None:
        primary = next(result for result in fitted if result.label == "7-rad")
        args.export_model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "schema_version": "r0_v3_elasticnet_7rad_v1",
                "model_label": "7-rad elastic-net full-cohort refit",
                "feature_names": primary.features,
                "model": primary.estimator,
                "metadata": {
                    "cohort_n": int(len(frame)),
                    "events": int(frame["cr_popf"].sum()),
                    "class_weight": "none",
                    "estimator": (
                        "StandardScaler + LogisticRegression elastic-net"
                    ),
                    "full_data_hyperparameters": primary.parameters,
                    "validation_reference": (
                        "results_reference/locked_panel_candidate_632plus/"
                        "candidate_model_metrics.csv"
                    ),
                    "cutpoint_reference": (
                        "results_reference/bootstrap_oob_cutpoints/"
                        "full_cohort_cutpoints.csv"
                    ),
                },
            },
            args.export_model,
        )
    plot_comparison(frame, fitted, metrics, bins, args.output)
    print(metrics.to_string(index=False))
    print(f"\nWrote outputs to {args.output}")


if __name__ == "__main__":
    main()
