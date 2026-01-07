#!/usr/bin/env python3
"""
Fit ComBat harmonization parameters on a reference cohort.

Best practice for evaluation/deployment:
- Fit ComBat on the training/reference cohort only (no leakage).
- Save the fitted ComBat estimates.
- Apply the same estimates to new patients (see combat_apply.py).
"""

from __future__ import annotations

import argparse
import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# Compatibility for older neuroCombat versions on NumPy>=1.24
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


DEFAULT_EXCLUDE_COLS = {
    "extraction_timestamp",
    "extraction_date",
    "extraction_settings",
    "config_version",
    "binWidth_used",
    "resampling_used",
    "head_volume_ml",
    "mean_hu",
    "harmonized",
    "harmonization_applied",
    "reference_batch",
    "variance_preservation",
    "batch_group",
    "combat_batch",
    "combat_harmonized",
    "scanner_id",
    "ScannerID",
}


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _is_shape_feature(col: str) -> bool:
    # pyradiomics shape features typically look like:
    # - original_shape_*
    # - original_shape2D_*
    return col.startswith("original_shape")


def _select_feature_columns(
    df: pd.DataFrame,
    *,
    patient_col: str | None,
    exclude_shape: bool,
    extra_exclude: set[str],
) -> list[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = set(extra_exclude)
    if patient_col:
        exclude.add(patient_col)

    selected: list[str] = []
    for col in numeric_cols:
        if col in exclude:
            continue
        if col.startswith("diagnostics_"):
            continue
        if exclude_shape and _is_shape_feature(col):
            continue
        selected.append(col)
    return selected


def _build_batch_labels(
    features_df: pd.DataFrame,
    *,
    patient_col: str | None,
    batch_col_in_features: str | None,
    scanner_metadata_path: Path | None,
    metadata_id_col: str | None,
    metadata_batch_col: str | None,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Returns
    -------
    - batch_labels: Series aligned to features_df rows
    - meta: dict with resolved merge/batch info (for saving)
    """
    if batch_col_in_features:
        if batch_col_in_features not in features_df.columns:
            raise ValueError(
                f"--batch-col '{batch_col_in_features}' not found in features CSV"
            )
        return (
            features_df[batch_col_in_features],
            {
                "batch_source": "features_csv",
                "batch_col_in_features": batch_col_in_features,
            },
        )

    if scanner_metadata_path is None:
        raise ValueError("Provide --scanner-metadata or --batch-col")
    if patient_col is None:
        raise ValueError("patient_col is required when using --scanner-metadata")

    meta_df = pd.read_csv(scanner_metadata_path)
    resolved_id_col = metadata_id_col or _detect_column(
        meta_df, ["PatientName", "scanner_patient_name", "patient_id", "PatientID"]
    )
    if resolved_id_col is None:
        raise ValueError(
            "Could not detect metadata id column; pass --metadata-id-col "
            "(e.g. PatientName)"
        )

    resolved_batch_col = metadata_batch_col or _detect_column(
        meta_df, ["ScannerID", "scanner_id", "combat_batch", "batch_group"]
    )
    if resolved_batch_col is None:
        if {"Manufacturer", "ManufacturerModelName"}.issubset(meta_df.columns):
            resolved_batch_col = "__ScannerID__"
            meta_df[resolved_batch_col] = (
                meta_df["Manufacturer"].astype(str)
                + "_"
                + meta_df["ManufacturerModelName"].astype(str)
            )
        else:
            raise ValueError(
                "Could not detect metadata batch column; pass --metadata-batch-col "
                "(e.g. ScannerID) or provide Manufacturer+ManufacturerModelName."
            )

    merged = features_df[[patient_col]].merge(
        meta_df[[resolved_id_col, resolved_batch_col]],
        how="left",
        left_on=patient_col,
        right_on=resolved_id_col,
    )
    return (
        merged[resolved_batch_col],
        {
            "batch_source": "scanner_metadata",
            "scanner_metadata_path": str(scanner_metadata_path),
            "metadata_id_col": resolved_id_col,
            "metadata_batch_col": resolved_batch_col,
        },
    )


def _encode_batches(batch_labels: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    labels = batch_labels.astype(str).to_numpy()
    unique = np.unique(labels)
    mapping = {label: int(i) for i, label in enumerate(unique)}
    encoded = np.array([mapping[l] for l in labels], dtype=int)
    return encoded, mapping


@dataclass(frozen=True)
class CombatFitResult:
    feature_cols: list[str]
    medians: dict[str, float] | None
    batch_label_to_code: dict[str, int]
    ref_batch_label: str | None
    ref_batch_code: int | None
    combat_estimates: dict[str, Any]
    harmonized_data: np.ndarray


def fit_combat(
    features_df: pd.DataFrame,
    batch_labels: pd.Series,
    feature_cols: list[str],
    *,
    ref_batch: str | None,
    eb: bool,
    parametric: bool,
    mean_only: bool,
    impute_missing: str,
) -> CombatFitResult:
    from neuroCombat import neuroCombat

    if batch_labels.isna().any():
        raise ValueError("batch_labels contains missing values; filter first")

    X = features_df[feature_cols].copy()
    medians: dict[str, float] | None = None

    if X.isna().any().any():
        if impute_missing == "error":
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"Found NaNs in {len(nan_cols)} feature columns. "
                "Impute before ComBat, or pass --impute-missing median. "
                f"Example columns: {nan_cols[:10]}"
            )
        if impute_missing != "median":
            raise ValueError(f"Unsupported --impute-missing: {impute_missing}")
        medians = {c: float(X[c].median()) for c in X.columns}
        for c, v in medians.items():
            X[c] = X[c].fillna(v)

    batch_encoded, mapping = _encode_batches(batch_labels)

    if ref_batch is None:
        ref_code = None
        ref_label = None
    else:
        if ref_batch == "auto":
            ref_label = batch_labels.astype(str).value_counts().idxmax()
        else:
            ref_label = ref_batch
        if ref_label not in mapping:
            raise ValueError(
                f"ref batch '{ref_label}' not found in batches; "
                f"known: {sorted(mapping)[:10]}"
            )
        ref_code = mapping[ref_label]

    covars = pd.DataFrame({"batch": batch_encoded})
    dat = X.to_numpy(dtype=float).T  # (features, samples)
    out = neuroCombat(
        dat=dat,
        covars=covars,
        batch_col="batch",
        eb=eb,
        parametric=parametric,
        mean_only=mean_only,
        ref_batch=ref_code,
    )

    return CombatFitResult(
        feature_cols=feature_cols,
        medians=medians,
        batch_label_to_code=mapping,
        ref_batch_label=ref_label,
        ref_batch_code=ref_code,
        combat_estimates=out["estimates"],
        harmonized_data=np.asarray(out["data"]),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Fit ComBat on a reference cohort and save estimates for deployment."
    )
    parser.add_argument("--features-csv", type=Path, required=True, help="Reference cohort features CSV")
    parser.add_argument("--output-estimates", type=Path, required=True, help="Where to write fitted estimates (.pkl)")

    parser.add_argument(
        "--patient-col",
        type=str,
        default=None,
        help="Patient id column in features CSV (auto-detects common names)",
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch/scanner column already present in the features CSV (skip scanner metadata merge)",
    )
    parser.add_argument("--scanner-metadata", type=Path, default=None, help="Scanner metadata CSV")
    parser.add_argument(
        "--metadata-id-col",
        type=str,
        default=None,
        help="Patient id column in scanner metadata (auto-detects common names)",
    )
    parser.add_argument(
        "--metadata-batch-col",
        type=str,
        default=None,
        help="Batch/scanner column in scanner metadata (auto-detects ScannerID/Manufacturer+Model)",
    )

    parser.add_argument(
        "--ref-batch",
        type=str,
        default="auto",
        help="Reference batch label, 'auto' (largest), or 'none' (grand-mean ComBat)",
    )
    parser.add_argument("--exclude-shape", action="store_true", default=True, help="Do not harmonize shape features")
    parser.add_argument("--include-shape", dest="exclude_shape", action="store_false", help="Also harmonize shape features")
    parser.add_argument(
        "--exclude-col",
        action="append",
        default=[],
        help="Column name to exclude from harmonization (repeatable)",
    )
    parser.add_argument(
        "--impute-missing",
        choices=["error", "median"],
        default="error",
        help="How to handle NaNs in features before ComBat",
    )

    parser.add_argument("--no-eb", action="store_true", help="Disable empirical Bayes (EB)")
    parser.add_argument("--non-parametric", action="store_true", help="Use non-parametric EB")
    parser.add_argument("--mean-only", action="store_true", help="Adjust means only (no scaling)")

    parser.add_argument(
        "--output-harmonized-csv",
        type=Path,
        default=None,
        help="Optionally write harmonized reference cohort CSV",
    )

    args = parser.parse_args()

    features_df = pd.read_csv(args.features_csv)
    if args.patient_col:
        patient_col = args.patient_col
    else:
        patient_col = _detect_column(
            features_df, ["patient_id", "scanner_patient_name", "PatientName", "PatientID"]
        )
    if patient_col is None and (args.scanner_metadata is not None):
        raise ValueError(
            "Could not detect patient id column in features CSV; pass --patient-col."
        )
    if patient_col is None and (args.batch_col is None):
        raise ValueError(
            "Provide --batch-col (if already in the CSV) or --patient-col + --scanner-metadata."
        )

    exclude_cols = set(DEFAULT_EXCLUDE_COLS)
    exclude_cols.update(args.exclude_col)

    feature_cols = _select_feature_columns(
        features_df,
        patient_col=patient_col,
        exclude_shape=args.exclude_shape,
        extra_exclude=exclude_cols,
    )
    if not feature_cols:
        raise ValueError("No numeric feature columns found to harmonize.")
    LOGGER.info("Selected %d numeric feature columns for ComBat", len(feature_cols))

    if args.ref_batch == "none":
        ref_batch: str | None = None
    else:
        ref_batch = args.ref_batch

    batch_labels, batch_meta = _build_batch_labels(
        features_df,
        patient_col=patient_col,
        batch_col_in_features=args.batch_col,
        scanner_metadata_path=args.scanner_metadata,
        metadata_id_col=args.metadata_id_col,
        metadata_batch_col=args.metadata_batch_col,
    )

    valid_mask = batch_labels.notna()
    if not valid_mask.all():
        missing = int((~valid_mask).sum())
        LOGGER.warning("Dropping %d rows with missing batch labels", missing)

    features_fit = features_df.loc[valid_mask].reset_index(drop=True)
    batch_fit = batch_labels.loc[valid_mask].reset_index(drop=True)

    result = fit_combat(
        features_fit,
        batch_fit,
        feature_cols,
        ref_batch=ref_batch,
        eb=not args.no_eb,
        parametric=not args.non_parametric,
        mean_only=args.mean_only,
        impute_missing=args.impute_missing,
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "features_csv": str(args.features_csv),
        "patient_col": patient_col,
        "feature_cols": result.feature_cols,
        "exclude_shape": bool(args.exclude_shape),
        "exclude_cols": sorted(exclude_cols),
        "impute_missing": args.impute_missing,
        "train_feature_medians": result.medians,
        "batch_label_to_code": result.batch_label_to_code,
        "batch_code_to_label": {v: k for k, v in result.batch_label_to_code.items()},
        "ref_batch_label": result.ref_batch_label,
        "ref_batch_code": result.ref_batch_code,
        "neurocombat_kwargs": {
            "eb": not args.no_eb,
            "parametric": not args.non_parametric,
            "mean_only": bool(args.mean_only),
            "ref_batch": result.ref_batch_code,
        },
        "batch_meta": batch_meta,
        "combat_estimates": result.combat_estimates,
    }

    args.output_estimates.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_estimates, "wb") as f:
        pickle.dump(payload, f)
    LOGGER.info("Wrote ComBat estimates to: %s", args.output_estimates)

    if args.output_harmonized_csv:
        harmonized = result.harmonized_data.T  # (samples, features)
        out_df = features_fit.copy()
        out_df.loc[:, result.feature_cols] = harmonized
        out_df.to_csv(args.output_harmonized_csv, index=False)
        LOGGER.info("Wrote harmonized reference CSV to: %s", args.output_harmonized_csv)


if __name__ == "__main__":
    main()
