#!/usr/bin/env python3
"""
Predict CR-POPF risk for new patients using the exported radiomics signature.

This script is designed for the model bundle exported as:
  `primary analysis/configs/exported_model.pkl`

That artifact is a dict containing:
  - preprocessor: fitted PreprocessingPipeline (custom class)
  - model: fitted sklearn LogisticRegression
  - preprocessed_feature_names: list[str] of required feature columns (HF3 schema)
  - panel_indices: list[int] indices of the 7-feature radiomics signature

Optionally, you can apply a post-hoc calibration mapping (sigmoid or isotonic)
exported by `code/models/comparative_risk_stratification_v2.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_stabl_available() -> None:
    """
    The exported preprocessing pipeline contains STABL's LowInfoFilter.
    Install it via pip (recommended) before running inference.
    """
    try:
        import stabl  # noqa: F401
        return
    except Exception:
        raise RuntimeError(
            "Could not import `stabl`. Install it via pip before running inference, e.g. "
            "`pip install git+https://github.com/gregbellan/Stabl.git@<commit>`."
        )

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# Compatibility shim for unpickling `configs/exported_model.pkl`
# ---------------------------------------------------------------------------
#
# The model bundle was pickled from a script execution context, so the
# preprocessor class is stored as `__main__.PreprocessingPipeline`.
#
# Defining this class in *this* script ensures joblib can unpickle the bundle.
# Only `.transform()` is required for inference (the fitted sklearn Pipeline is
# stored as `self.pipeline` inside the pickled object).
#


class PreprocessingPipeline:  # noqa: D101
    def __init__(
        self,
        use_variance_filter: bool = True,
        variance_threshold: float = 0.01,
        use_low_info_filter: bool = True,
        max_nan_fraction: float = 0.2,
        impute_strategy: str = "median",
        use_scaler: bool = True,
    ):
        self.use_variance_filter = use_variance_filter
        self.variance_threshold = variance_threshold
        self.use_low_info_filter = use_low_info_filter
        self.max_nan_fraction = max_nan_fraction
        self.impute_strategy = impute_strategy
        self.use_scaler = use_scaler
        self.pipeline = None
        self.feature_names_out_ = None

    def transform(self, X):  # noqa: D401
        """Transform data using the fitted pipeline without refitting."""
        if self.pipeline is None:
            raise RuntimeError("Preprocessing pipeline not fitted. Cannot transform.")
        return self.pipeline.transform(X)


def _detect_id_col(df: pd.DataFrame) -> str | None:
    for candidate in ("patient_id", "scanner_patient_name", "PatientName", "PatientID"):
        if candidate in df.columns:
            return candidate
    return None


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_calibration_info(calibration_json: Path) -> dict[str, Any]:
    """
    Supports either:
    - a plain calibrator file: {"type": "sigmoid", "intercept": ..., "slope": ...}
    - a report produced by `comparative_risk_stratification_v2.py`:
        {"method": "...", "diagnostics": {"sigmoid": {"info": {...}}, ...}}
    """
    payload = json.loads(calibration_json.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Calibration JSON must be an object/dict.")

    if "type" in payload:
        return payload

    method = payload.get("method")
    diagnostics = payload.get("diagnostics", {})
    if isinstance(method, str) and isinstance(diagnostics, dict):
        method_payload = diagnostics.get(method)
        if isinstance(method_payload, dict):
            info = method_payload.get("info")
            if isinstance(info, dict) and "type" in info:
                return info

    raise ValueError(
        "Unsupported calibration JSON format. Provide either a calibrator dict with a "
        "`type` key, or a calibration report produced by "
        "`code/models/comparative_risk_stratification_v2.py` that includes "
        "`diagnostics[method].info`."
    )


def _load_risk_thresholds(risk_thresholds_json: Path) -> dict[str, Any]:
    payload = json.loads(risk_thresholds_json.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Risk thresholds JSON must be an object/dict.")

    low = payload.get("low_threshold")
    high = payload.get("high_threshold")
    if low is None or high is None:
        raise ValueError("Risk thresholds JSON must include low_threshold and high_threshold.")
    low_thr = float(low)
    high_thr = float(high)
    if not np.isfinite(low_thr) or not np.isfinite(high_thr):
        raise ValueError("Risk thresholds must be finite numbers.")
    if low_thr > high_thr:
        low_thr, high_thr = high_thr, low_thr

    probability_source = payload.get("probability_source", "popf_risk_calibrated")
    if not isinstance(probability_source, str) or not probability_source.strip():
        probability_source = "popf_risk_calibrated"

    payload["low_threshold"] = float(low_thr)
    payload["high_threshold"] = float(high_thr)
    payload["probability_source"] = probability_source
    return payload


def _risk_group_from_probability(p: float, low_thr: float, high_thr: float) -> str:
    # Match `build_risk_table()` in `comparative_risk_stratification_v2.py`
    if p <= low_thr:
        return "Low"
    if p <= high_thr:
        return "Intermediate"
    return "High"


def _apply_calibration(prob_raw: np.ndarray, calibrator: dict[str, Any]) -> np.ndarray:
    cal_type = str(calibrator.get("type", "")).lower().strip()
    if cal_type == "sigmoid":
        intercept = float(calibrator["intercept"])
        slope = float(calibrator["slope"])
        return _sigmoid(intercept + slope * _logit(prob_raw))

    if cal_type == "isotonic":
        x = np.asarray(calibrator["x"], dtype=float)
        y = np.asarray(calibrator["y"], dtype=float)
        if x.size == 0 or y.size == 0 or x.size != y.size:
            raise ValueError("Invalid isotonic calibrator: x/y must be same non-zero length.")
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        out = np.interp(prob_raw, x, y, left=y[0], right=y[-1])
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unsupported calibrator type: {cal_type!r}")


def _load_model_bundle(model_pkl: Path) -> dict[str, Any]:
    _ensure_stabl_available()

    import joblib

    bundle = joblib.load(model_pkl)
    if not isinstance(bundle, dict):
        raise ValueError(
            f"Expected a dict model bundle in {model_pkl}, got {type(bundle)}."
        )

    required = {"model", "preprocessor", "preprocessed_feature_names", "panel_indices"}
    missing = required.difference(bundle.keys())
    if missing:
        raise ValueError(f"Model bundle is missing keys: {sorted(missing)}")

    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict CR-POPF risk from radiomics features (exported model bundle).")
    parser.add_argument(
        "--model-pkl",
        type=Path,
        default=Path("configs/exported_model.pkl"),
        help="Path to exported model bundle (.pkl).",
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        required=True,
        help="Radiomics CSV (one row per patient). Must contain the HF3 feature columns.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="auto",
        help="Patient identifier column name (default: auto-detect).",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Optional filter: run prediction for a single patient ID.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="Optional calibration JSON (sigmoid/isotonic) to convert raw to calibrated risk.",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable calibration even if a default calibration JSON exists.",
    )
    parser.add_argument(
        "--risk-thresholds-json",
        type=Path,
        default=None,
        help="Optional risk stratification thresholds JSON (Low/Intermediate/High). "
        "If omitted, the script will auto-use `configs/calibration/radiomics_risk_stratification.json` if present.",
    )
    parser.add_argument(
        "--no-risk-group",
        action="store_true",
        help="Disable risk group assignment even if thresholds JSON exists.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("popf_predictions.csv"),
        help="Output CSV with predictions.",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("popf_predictions_meta.json"),
        help="Output JSON with run metadata.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    args = parse_args()

    default_calibration_json = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "calibration"
        / "radiomics_calibration.json"
    )
    default_risk_thresholds_json = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "calibration"
        / "radiomics_risk_stratification.json"
    )

    model_pkl = args.model_pkl
    if not model_pkl.is_absolute():
        # Assume execution from `primary analysis/` root.
        model_pkl = (Path.cwd() / model_pkl).resolve()

    bundle = _load_model_bundle(model_pkl)

    df = pd.read_csv(args.features_csv)
    if args.id_col == "auto":
        id_col = _detect_id_col(df)
        if id_col is None:
            raise ValueError(
                "Could not auto-detect ID column. Provide --id-col (e.g. patient_id)."
            )
    else:
        id_col = args.id_col
        if id_col not in df.columns:
            raise ValueError(f"ID column '{id_col}' not found in {args.features_csv}.")

    if args.patient_id is not None:
        df = df[df[id_col].astype(str) == str(args.patient_id)]
        if df.empty:
            raise ValueError(f"No rows found for {id_col} == {args.patient_id!r}.")

    required_features: list[str] = list(bundle["preprocessed_feature_names"])
    missing = [c for c in required_features if c not in df.columns]
    if missing:
        example = ", ".join(missing[:10])
        raise ValueError(
            f"Missing {len(missing)} required feature columns. Example: {example}\n"
            "This usually means the radiomics extraction config does not match HF3 "
            "(e.g., LoG sigma 3/5/7 vs 2/3/4). Use the YAML config "
            "`radiomics pipeline/code/configs/radiomics_config_2mm.yaml`."
        )

    # Use a NumPy array to avoid downstream warnings about feature names.
    X = df[required_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    preprocessor = bundle["preprocessor"]
    X_pre = preprocessor.transform(X)

    panel_indices: list[int] = [int(i) for i in list(bundle["panel_indices"])]

    # If preprocessing ever drops columns, remap indices by feature name.
    if X_pre.shape[1] != len(required_features):
        names_out = getattr(preprocessor, "feature_names_out_", None)
        if not isinstance(names_out, list) or not names_out:
            raise RuntimeError(
                "Preprocessor changed feature dimensionality but did not expose feature_names_out_. "
                "Cannot safely map the radiomics panel."
            )
        panel_names = [required_features[i] for i in panel_indices]
        panel_indices = [names_out.index(name) for name in panel_names]

    X_panel = X_pre[:, panel_indices]
    model = bundle["model"]
    prob_raw = model.predict_proba(X_panel)[:, 1]

    output = pd.DataFrame({id_col: df[id_col].astype(str), "popf_risk_raw": prob_raw.astype(float)})

    calibration_json = None if args.no_calibration else args.calibration_json
    if calibration_json is None and not args.no_calibration and default_calibration_json.exists():
        calibration_json = default_calibration_json

    cal_info = None
    calibration_applied = False
    if calibration_json is not None:
        if not calibration_json.exists():
            raise FileNotFoundError(f"Calibration JSON not found: {calibration_json}")
        cal_info = _load_calibration_info(calibration_json)
        output["popf_risk_calibrated"] = _apply_calibration(prob_raw, cal_info).astype(float)
        calibration_applied = True

    thresholds_json = None if args.no_risk_group else args.risk_thresholds_json
    if thresholds_json is None and not args.no_risk_group and default_risk_thresholds_json.exists():
        thresholds_json = default_risk_thresholds_json

    risk_info = None
    risk_applied = False
    if thresholds_json is not None:
        if not thresholds_json.exists():
            raise FileNotFoundError(f"Risk thresholds JSON not found: {thresholds_json}")
        risk_info = _load_risk_thresholds(thresholds_json)
        src = str(risk_info.get("probability_source", "popf_risk_calibrated"))
        if src not in output.columns:
            LOGGER.warning(
                "Risk thresholds expect probability_source=%s but that column is missing; skipping risk_group. "
                "(Did you disable calibration?)",
                src,
            )
        else:
            low_thr = float(risk_info["low_threshold"])
            high_thr = float(risk_info["high_threshold"])
            output["risk_threshold_low"] = low_thr
            output["risk_threshold_high"] = high_thr
            output["risk_probability_source"] = src
            output["risk_group"] = [
                _risk_group_from_probability(float(p), low_thr, high_thr)
                for p in output[src].astype(float).tolist()
            ]
            risk_applied = True

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)

    meta = {
        "model_pkl": str(model_pkl),
        "features_csv": str(args.features_csv.resolve()),
        "id_col": id_col,
        "n_patients": int(len(output)),
        "required_feature_count": int(len(required_features)),
        "panel_indices": panel_indices,
        "panel_feature_names": [required_features[i] for i in list(bundle["panel_indices"])],
        "calibration_json": str(calibration_json) if calibration_json is not None else None,
        "calibration": cal_info,
        "risk_thresholds_json": str(thresholds_json) if thresholds_json is not None else None,
        "risk_thresholds": risk_info,
        "risk_applied": bool(risk_applied),
        "calibration_applied": bool(calibration_applied),
    }
    args.output_meta.parent.mkdir(parents=True, exist_ok=True)
    args.output_meta.write_text(json.dumps(meta, indent=2))

    LOGGER.info("Wrote predictions: %s", args.output_csv)
    LOGGER.info("Wrote metadata: %s", args.output_meta)


if __name__ == "__main__":
    main()
