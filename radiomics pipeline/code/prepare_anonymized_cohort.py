#!/usr/bin/env python3
"""
Prepare anonymized radiomics cohort + scanner metadata.

Steps:
1) Replace identifying patient IDs with anonymous IDs using a mapping file.
2) Join anonymized outcomes for modeling.
3) Create anonymized scanner metadata keyed by anonymous IDs.
4) Optionally delete identifying CSVs after outputs are written.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _resolve_default_paths():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"
    return repo_root, data_dir


def _require_columns(df: pd.DataFrame, cols: list[str], path: Path):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s) {missing} in {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    _, data_dir = _resolve_default_paths()

    parser = argparse.ArgumentParser(description="Anonymize radiomics cohort and scanner metadata")
    parser.add_argument(
        "--radiomics-csv",
        type=Path,
        default=data_dir / "raw_cohort_radiomics.csv",
        help="Raw radiomics CSV with identifying patient IDs",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=data_dir / "patient_mapping.csv",
        help="Mapping CSV: original_name -> anonymous_id",
    )
    parser.add_argument(
        "--outcomes-csv",
        type=Path,
        default=data_dir / "popf_anonymized.csv",
        help="Anonymized outcomes CSV (patient_id, popf_grade, cr_popf)",
    )
    parser.add_argument(
        "--scanner-metadata-csv",
        type=Path,
        default=data_dir / "scanner_metadata_collapsed_20250917_212453.csv",
        help="Scanner metadata CSV with identifying patient names",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_dir,
        help="Directory to write anonymized outputs",
    )
    parser.add_argument(
        "--radiomics-id-col",
        type=str,
        default="patient_id",
        help="ID column in radiomics CSV",
    )
    parser.add_argument(
        "--mapping-original-col",
        type=str,
        default="original_name",
        help="Original ID column in mapping CSV",
    )
    parser.add_argument(
        "--mapping-anon-col",
        type=str,
        default="anonymous_id",
        help="Anonymous ID column in mapping CSV",
    )
    parser.add_argument(
        "--scanner-id-col",
        type=str,
        default="PatientName",
        help="ID column in scanner metadata CSV",
    )
    parser.add_argument(
        "--output-radiomics",
        type=Path,
        default=None,
        help="Output anonymized radiomics CSV (features only)",
    )
    parser.add_argument(
        "--output-radiomics-with-outcomes",
        type=Path,
        default=None,
        help="Output anonymized radiomics CSV joined to outcomes",
    )
    parser.add_argument(
        "--output-scanner-metadata",
        type=Path,
        default=None,
        help="Output anonymized scanner metadata CSV",
    )
    parser.add_argument(
        "--delete-identifying",
        action="store_true",
        help="Delete identifying CSVs after outputs are written",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    radiomics_path = Path(args.radiomics_csv)
    mapping_path = Path(args.mapping_csv)
    outcomes_path = Path(args.outcomes_csv)
    scanner_path = Path(args.scanner_metadata_csv)

    out_radiomics = args.output_radiomics or (output_dir / "radiomics_cohort_anonymized.csv")
    out_radiomics_outcomes = args.output_radiomics_with_outcomes or (
        output_dir / "radiomics_cohort_anonymized_with_outcomes.csv"
    )
    out_scanner = args.output_scanner_metadata or (
        output_dir / f"{scanner_path.stem}_anonymized.csv"
    )

    # Load inputs
    radiomics_df = pd.read_csv(radiomics_path)
    mapping_df = pd.read_csv(mapping_path)
    outcomes_df = pd.read_csv(outcomes_path)
    scanner_df = pd.read_csv(scanner_path)

    _require_columns(radiomics_df, [args.radiomics_id_col], radiomics_path)
    _require_columns(mapping_df, [args.mapping_original_col, args.mapping_anon_col], mapping_path)
    _require_columns(outcomes_df, ["patient_id"], outcomes_path)
    _require_columns(scanner_df, [args.scanner_id_col], scanner_path)

    # Build ID map
    id_map = dict(
        zip(mapping_df[args.mapping_original_col], mapping_df[args.mapping_anon_col])
    )

    # Anonymize radiomics IDs
    radiomics_df = radiomics_df.copy()
    radiomics_df[args.radiomics_id_col] = radiomics_df[args.radiomics_id_col].map(id_map)
    missing = radiomics_df[args.radiomics_id_col].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} radiomics rows could not be mapped to anonymous IDs."
        )

    # Write anonymized radiomics
    radiomics_df.to_csv(out_radiomics, index=False)
    LOGGER.info("Wrote anonymized radiomics: %s", out_radiomics)

    # Join outcomes (inner join -> only patients with outcomes)
    radiomics_outcomes = radiomics_df.merge(outcomes_df, on="patient_id", how="inner")
    LOGGER.info(
        "Outcomes matched: %d/%d",
        len(radiomics_outcomes),
        len(radiomics_df),
    )
    radiomics_outcomes.to_csv(out_radiomics_outcomes, index=False)
    LOGGER.info("Wrote radiomics + outcomes: %s", out_radiomics_outcomes)

    # Anonymize scanner metadata
    scanner_df = scanner_df.copy()
    scanner_df["patient_id"] = scanner_df[args.scanner_id_col].map(id_map)
    missing_scan = scanner_df["patient_id"].isna().sum()
    if missing_scan:
        raise ValueError(
            f"{missing_scan} scanner metadata rows could not be mapped to anonymous IDs."
        )
    scanner_df = scanner_df.drop(columns=[args.scanner_id_col])
    scanner_df.to_csv(out_scanner, index=False)
    LOGGER.info("Wrote anonymized scanner metadata: %s", out_scanner)

    # Optionally delete identifying inputs
    if args.delete_identifying:
        for path in [radiomics_path, mapping_path, scanner_path]:
            if path.exists():
                path.unlink()
                LOGGER.info("Deleted identifying file: %s", path)


if __name__ == "__main__":
    main()

