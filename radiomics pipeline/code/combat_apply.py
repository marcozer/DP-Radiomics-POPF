#!/usr/bin/env python3
"""
Apply a pre-fitted ComBat transform to new patient(s).

Use combat_fit.py once on the reference cohort to create the estimates file,
then run this script for each new radiomics CSV you want to harmonize.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _detect_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _resolve_batch_labels(
    features_df: pd.DataFrame,
    *,
    patient_col: str | None,
    scanner_id: str | None,
    batch_col_in_features: str | None,
    scanner_metadata_path: Path | None,
    metadata_id_col: str | None,
    metadata_batch_col: str | None,
) -> pd.Series:
    if batch_col_in_features:
        if batch_col_in_features not in features_df.columns:
            raise ValueError(f"--batch-col '{batch_col_in_features}' not found in features CSV")
        return features_df[batch_col_in_features]

    if scanner_id is not None:
        return pd.Series([scanner_id] * len(features_df), index=features_df.index)

    if scanner_metadata_path is None:
        raise ValueError("Provide --scanner-id or --batch-col or --scanner-metadata")
    if patient_col is None:
        raise ValueError("patient_col is required when using --scanner-metadata")

    meta_df = pd.read_csv(scanner_metadata_path)
    resolved_id_col = metadata_id_col or _detect_column(
        meta_df, ["PatientName", "scanner_patient_name", "patient_id", "PatientID"]
    )
    if resolved_id_col is None:
        raise ValueError(
            "Could not detect metadata id column; pass --metadata-id-col (e.g. PatientName)"
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
    return merged[resolved_batch_col]


def _combat_from_training(
    dat: np.ndarray, *, batch: np.ndarray, estimates: dict[str, Any]
) -> np.ndarray:
    """
    Apply a pre-fitted neuroCombat model to new samples.

    This mirrors `neuroCombat.neuroCombatFromTraining` but avoids its stdout prints.
    """
    batch = np.array(batch, dtype=str)
    new_levels = np.unique(batch)
    old_levels = np.array(estimates["batches"], dtype=str)
    missing_levels = np.setdiff1d(new_levels, old_levels)
    if missing_levels.shape[0] != 0:
        raise ValueError(
            "The batches "
            + str(missing_levels.tolist())
            + " are not part of the training dataset"
        )

    wh = [int(np.where(old_levels == x)[0][0]) for x in batch]

    var_pooled = np.asarray(estimates["var.pooled"])
    stand_mean = np.asarray(estimates["stand.mean"])[:, 0]
    mod_mean = np.asarray(estimates["mod.mean"])
    gamma_star = np.asarray(estimates["gamma.star"])
    delta_star = np.asarray(estimates["delta.star"])

    n_array = dat.shape[1]
    stand_mean = stand_mean + mod_mean.mean(axis=1)
    stand_mean = np.transpose([stand_mean] * n_array)  # (features, samples)

    bayesdata = (dat - stand_mean) / np.sqrt(var_pooled)
    gamma = np.transpose(gamma_star[wh, :])
    delta = np.transpose(delta_star[wh, :])
    bayesdata = (bayesdata - gamma) / np.sqrt(delta)
    bayesdata = bayesdata * np.sqrt(var_pooled) + stand_mean
    return bayesdata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Apply pre-fitted ComBat estimates to a features CSV")
    parser.add_argument("--features-csv", type=Path, required=True, help="New patient(s) features CSV")
    parser.add_argument("--estimates-pkl", type=Path, required=True, help="Pickle written by combat_fit.py")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV with harmonized features")

    parser.add_argument("--scanner-id", type=str, default=None, help="Scanner/batch label for all rows")
    parser.add_argument(
        "--batch-col",
        type=str,
        default=None,
        help="Batch/scanner column already present in the features CSV",
    )
    parser.add_argument("--scanner-metadata", type=Path, default=None, help="Scanner metadata CSV to look up batch labels")
    parser.add_argument("--metadata-id-col", type=str, default=None, help="ID column name in metadata")
    parser.add_argument("--metadata-batch-col", type=str, default=None, help="Batch column name in metadata")

    args = parser.parse_args()

    features_df = pd.read_csv(args.features_csv)
    with open(args.estimates_pkl, "rb") as f:
        payload: dict[str, Any] = pickle.load(f)

    patient_col = payload.get("patient_col")
    feature_cols: list[str] = list(payload["feature_cols"])
    batch_label_to_code: dict[str, int] = dict(payload["batch_label_to_code"])
    medians: dict[str, float] | None = payload.get("train_feature_medians")
    combat_estimates: dict[str, Any] = payload["combat_estimates"]

    if patient_col and patient_col not in features_df.columns:
        # Try common alternatives for single-patient exports.
        alt = _detect_column(features_df, ["patient_id", "scanner_patient_name", "PatientName", "PatientID"])
        if alt and alt != patient_col:
            LOGGER.warning("Expected patient_col '%s' not found; using '%s' instead", patient_col, alt)
            patient_col = alt

    batch_labels = _resolve_batch_labels(
        features_df,
        patient_col=patient_col,
        scanner_id=args.scanner_id,
        batch_col_in_features=args.batch_col,
        scanner_metadata_path=args.scanner_metadata,
        metadata_id_col=args.metadata_id_col,
        metadata_batch_col=args.metadata_batch_col,
    )

    if batch_labels.isna().any():
        missing = int(batch_labels.isna().sum())
        raise ValueError(
            f"Missing batch labels for {missing} rows. Provide --scanner-id, or ensure metadata contains these patients."
        )

    batch_codes: list[int] = []
    unknown: set[str] = set()
    for b in batch_labels.astype(str).to_list():
        if b not in batch_label_to_code:
            unknown.add(b)
            continue
        batch_codes.append(batch_label_to_code[b])
    if unknown:
        known = sorted(batch_label_to_code.keys())
        raise ValueError(
            "Unknown scanner/batch label(s): "
            + ", ".join(sorted(unknown)[:10])
            + f"\nKnown batches (first 20): {known[:20]}"
        )

    # Ensure all required columns exist and are numeric
    missing_cols = [c for c in feature_cols if c not in features_df.columns]
    if missing_cols:
        if medians is None:
            raise ValueError(
                f"Missing {len(missing_cols)} required feature columns and no training medians available to fill. "
                f"Example: {missing_cols[:10]}"
            )
        for c in missing_cols:
            features_df[c] = np.nan

    X = features_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if medians is not None:
        for c, v in medians.items():
            if c in X.columns:
                X[c] = X[c].fillna(float(v))

    if X.isna().any().any():
        nan_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(
            f"NaNs remain in {len(nan_cols)} feature columns after coercion/imputation. "
            f"Example: {nan_cols[:10]}"
        )

    dat = X.to_numpy(dtype=float).T  # (features, samples)
    harmonized = _combat_from_training(
        dat, batch=np.array(batch_codes, dtype=int), estimates=combat_estimates
    ).T  # (samples, features)

    out_df = features_df.copy()
    out_df.loc[:, feature_cols] = harmonized
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    LOGGER.info("Wrote harmonized features to: %s", args.output_csv)


if __name__ == "__main__":
    main()
