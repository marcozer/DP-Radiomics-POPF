#!/usr/bin/env python3
"""
Deployment smoke tests (no external test runner required).

These checks are meant to be run manually (or in CI) to ensure that:
- CLI contracts across repos still hold
- an optional end-to-end run works when you provide CT+seg inputs (no sample imaging is shipped)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RADPANC deployment smoke tests")
    parser.add_argument("--contracts-only", action="store_true", help="Only run CLI contract checks")
    parser.add_argument("--patient-id", default="test", help="Patient ID for the E2E run")
    parser.add_argument("--ct-nifti", type=Path, default=None, help="Path to CT NIfTI (.nii/.nii.gz) for optional E2E run")
    parser.add_argument("--pancreas-seg", type=Path, default=None, help="Path to pancreas mask NIfTI (.nii/.nii.gz) for optional E2E run")
    parser.add_argument("--head-x", type=float, default=4.7)
    parser.add_argument("--output-root", type=Path, default=Path("runs/smoke_tests"))
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    # 1) Contracts
    subprocess.run(
        [
            sys.executable,
            str(root / "deployment" / "radpanc_runner.py"),
            "--patient-id",
            args.patient_id,
            "--ct-nifti",
            "x",
            "--pancreas-seg",
            "y",
            "--head-x",
            "0",
            "--check-contracts",
        ],
        cwd=str(root),
        check=True,
    )
    print("OK: CLI contracts satisfied.")

    if args.contracts_only or args.ct_nifti is None or args.pancreas_seg is None:
        print("SKIP: end-to-end run (no CT/seg provided).")
        return

    # 2) End-to-end
    if not args.ct_nifti.exists():
        raise FileNotFoundError(f"CT not found: {args.ct_nifti}")
    if not args.pancreas_seg.exists():
        raise FileNotFoundError(f"Seg not found: {args.pancreas_seg}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (root / args.output_root / f"{args.patient_id}_{stamp}").resolve()

    subprocess.run(
        [
            sys.executable,
            str(root / "deployment" / "radpanc_runner.py"),
            "--patient-id",
            args.patient_id,
            "--ct-nifti",
            str(args.ct_nifti.resolve()),
            "--pancreas-seg",
            str(args.pancreas_seg.resolve()),
            "--head-x",
            str(args.head_x),
            "--run-dir",
            str(run_dir),
            "--skip-combat",
        ],
        cwd=str(root),
        check=True,
    )

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    pred = Path(manifest["outputs"]["prediction_csv"])
    if not pred.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {pred}")

    print(f"OK: end-to-end run succeeded ({pred})")


if __name__ == "__main__":
    run()
