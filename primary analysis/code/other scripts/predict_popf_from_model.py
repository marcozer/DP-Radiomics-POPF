#!/usr/bin/env python3
"""
Infer CR-POPF risk for new patients using a serialized model/pipeline.

This script:
- Loads a pickle (`deploy_model.pkl` or another path) that must expose
  `predict_proba` (ideally a sklearn Pipeline).
- Reads a radiomics CSV, aligns columns to the expected feature set
  (taken from the model or an explicit panel file), and outputs per-patient
  risk probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


def load_feature_list(model, panel_file: Optional[Path]) -> List[str]:
    """Resolve the feature list either from a panel file or the model metadata."""
    if panel_file:
        lines = panel_file.read_text().splitlines()
        feats = [ln.strip() for ln in lines if ln.strip()]
        if not feats:
            raise ValueError(f"Panel file {panel_file} is empty")
        return feats

    # Try common sklearn attributes
    for obj in (model, getattr(model, "named_steps", {}).values() if hasattr(model, "named_steps") else []):
        if hasattr(obj, "feature_names_in_"):
            feats = list(obj.feature_names_in_)
            if feats:
                return feats

    raise ValueError(
        "Could not infer feature names from the model. Provide --panel-file with the expected feature list."
    )


def ensure_columns(df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Subset and order columns; fail if required features are missing."""
    feature_names = list(feature_names)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features in data: {missing}")
    return df[feature_names]


def predict(args: argparse.Namespace) -> pd.DataFrame:
    model = joblib.load(args.model_path)
    feature_names = load_feature_list(model, args.panel_file)

    df = pd.read_csv(args.data_path)
    if args.id_col not in df.columns:
        raise ValueError(f"ID column '{args.id_col}' not found in data")

    X = ensure_columns(df, feature_names)

    # Convert to numeric; leave non-numeric as-is and rely on pipeline encoders if present.
    X_numeric = X.apply(pd.to_numeric, errors="ignore")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_numeric)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X_numeric)
        proba = 1 / (1 + np.exp(-scores))
    else:
        raise ValueError("Model does not implement predict_proba or decision_function")

    out = pd.DataFrame(
        {
            args.id_col: df[args.id_col],
            "popf_risk": proba,
        }
    )

    meta = {
        "model_path": str(Path(args.model_path).resolve()),
        "data_path": str(Path(args.data_path).resolve()),
        "id_col": args.id_col,
        "feature_count": len(feature_names),
        "n_patients": len(out),
    }
    return out, meta


def main():
    parser = argparse.ArgumentParser(description="Infer CR-POPF risk from a serialized model and radiomics CSV.")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to deploy_model.pkl (or compatible sklearn pipeline).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Radiomics CSV with patient rows.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="patient_id",
        help="Identifier column to carry through to the output.",
    )
    parser.add_argument(
        "--panel-file",
        type=Path,
        default=None,
        help="Optional text file with one feature name per line; if omitted, uses model.feature_names_in_.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("popf_predictions.csv"),
        help="Where to write the predictions CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("popf_predictions_meta.json"),
        help="Where to write a small metadata JSON.",
    )

    args = parser.parse_args()

    preds, meta = predict(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output, index=False)
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_output.write_text(json.dumps(meta, indent=2))

    print(f"Wrote predictions: {args.output}")
    print(f"Wrote metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
