#!/usr/bin/env python3
"""
RADPANC end-to-end runner (composition layer).

This tool composes the two sibling repos via CLI contracts:
- `radiomics pipeline/` for head mask extraction + radiomics (+ optional ComBat)
- `primary analysis/` for calibrated POPF risk inference

Design goal: keep the deployment surface stable even when internal scripts evolve.
We do this by:
- Calling scripts as subprocesses (no cross-repo imports)
- Checking required CLI flags before running (contract smoke check)
- Writing a run manifest (commands, inputs, outputs) per case
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts import CONTRACTS


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    radiomics_repo: Path
    analysis_repo: Path


def _repo_paths(args: argparse.Namespace) -> RepoPaths:
    root = Path(__file__).resolve().parents[1]
    radiomics_repo = (root / args.radiomics_repo).resolve()
    analysis_repo = (root / args.analysis_repo).resolve()
    return RepoPaths(root=root, radiomics_repo=radiomics_repo, analysis_repo=analysis_repo)


def _run(
    cmd: list[str],
    *,
    cwd: Path | None,
    env: dict[str, str] | None,
    dry_run: bool,
) -> None:
    cmd_str = " ".join([shlex_quote(c) for c in cmd])
    if dry_run:
        print(f"[DRY RUN] {cmd_str}")
        return
    print(f"[RADPANC] $ {cmd_str}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=True)


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


def _read_text(cmd: list[str], *, cwd: Path | None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return proc.stdout


def _assert_cli_contract(script: Path, required_tokens: list[str], *, cwd: Path | None) -> None:
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    out = _read_text([sys.executable, str(script), "--help"], cwd=cwd)
    missing = [tok for tok in required_tokens if tok not in out]
    if missing:
        raise RuntimeError(
            "CLI contract check failed for "
            f"{script} (missing tokens: {missing}).\n"
            "If you changed this script, update the deployment runner contract accordingly."
        )


def _copy_or_link(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _load_coordinates_subset(
    coordinates_json: Path, *, patient_id: str, output_path: Path
) -> Path:
    coords = json.loads(coordinates_json.read_text())
    if patient_id not in coords:
        raise KeyError(f"Patient '{patient_id}' not found in {coordinates_json}")
    _write_json(output_path, {patient_id: coords[patient_id]})
    return output_path


def _write_coordinates_from_args(args: argparse.Namespace, *, output_path: Path) -> Path:
    entry: dict[str, Any] = {"x_coordinate": float(args.head_x)}
    if args.head_y is not None:
        entry["y_coordinate"] = float(args.head_y)
    if args.head_z_limit is not None:
        entry["z_limit"] = int(args.head_z_limit)
    entry["timestamp"] = datetime.now().isoformat()
    _write_json(output_path, {args.patient_id: entry})
    return output_path


def _latest_matching(path: Path, pattern: str) -> Path:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matching {pattern!r} in {path}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RADPANC end-to-end runner (radiomics → optional ComBat → calibrated POPF risk)")

    parser.add_argument("--patient-id", required=True, help="Patient identifier (used for file naming)")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run directory (default: runs/<patient>_<timestamp>)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--link-inputs", action="store_true", help="Symlink inputs into run dir instead of copying")
    parser.add_argument("--check-contracts", action="store_true", help="Only check required CLIs, then exit")

    # Repo paths (relative to publication_export root by default)
    parser.add_argument("--radiomics-repo", default="radiomics pipeline", help="Path to the radiomics pipeline repo")
    parser.add_argument("--analysis-repo", default="primary analysis", help="Path to the primary analysis repo")

    # Inputs (provide either --ct-nifti or a DICOM input in a future extension)
    parser.add_argument("--ct-nifti", type=Path, required=True, help="CT NIfTI (.nii or .nii.gz)")
    parser.add_argument("--pancreas-seg", type=Path, required=True, help="Pancreas segmentation NIfTI (.nii or .nii.gz)")

    # Head coordinates (either pass --coordinates-json from viewer OR pass --head-x to generate one)
    parser.add_argument("--coordinates-json", type=Path, default=None, help="Viewer coordinate JSON (x_coordinate_selections.json)")
    parser.add_argument("--head-x", type=float, default=None, help="Head X coordinate (world space) to generate a coordinate JSON")
    parser.add_argument("--head-y", type=float, default=None, help="Optional head Y coordinate (world space)")
    parser.add_argument("--head-z-limit", type=int, default=None, help="Optional Z limit (slice index) for tail retention")

    # Radiomics extraction
    parser.add_argument(
        "--radiomics-config",
        type=Path,
        default=None,
        help="PyRadiomics YAML config (default: radiomics pipeline/code/configs/radiomics_config_2mm.yaml)",
    )

    # ComBat (optional)
    parser.add_argument("--combat-estimates", type=Path, default=None, help="ComBat estimates .pkl from combat_fit.py")
    parser.add_argument("--combat-scanner-id", type=str, default=None, help="Scanner/batch label for the new patient")
    parser.add_argument("--skip-combat", action="store_true", help="Do not apply ComBat even if estimates are provided")

    # Prediction (primary analysis)
    parser.add_argument("--model-pkl", type=Path, default=None, help="Exported model bundle (.pkl)")
    parser.add_argument("--calibration-json", type=Path, default=None, help="Optional calibration JSON; defaults to analysis configs if present")
    parser.add_argument("--no-calibration", action="store_true", help="Disable calibration at prediction time")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _repo_paths(args)

    radiomics_scripts = {
        "extract_head": paths.radiomics_repo / "code" / "extract_pancreatic_head_from_viewer_coordinates.py",
        "extract_ct": paths.radiomics_repo / "code" / "extract_ct_from_head_segmentations.py",
        "extract_radiomics": paths.radiomics_repo / "code" / "extract_radiomics_yaml.py",
        "combat_apply": paths.radiomics_repo / "code" / "combat_apply.py",
    }
    analysis_scripts = {
        "predict": paths.analysis_repo / "code" / "predict_popf_risk.py",
    }

    # Contract smoke checks
    for contract in CONTRACTS:
        repo_root = paths.radiomics_repo if contract.repo == "radiomics" else paths.analysis_repo
        _assert_cli_contract(
            repo_root / contract.rel_path,
            list(contract.required_tokens),
            cwd=repo_root,
        )

    if args.check_contracts:
        print("OK: CLI contracts satisfied.")
        return

    if args.coordinates_json is None and args.head_x is None:
        raise ValueError("Provide either --coordinates-json or --head-x (to generate a coordinate JSON).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (paths.root / "runs" / f"{args.patient_id}_{timestamp}")
    run_dir = run_dir.resolve()

    # Headless-safe matplotlib defaults (prevents macOS crashes when matplotlib is used from threads).
    base_env = os.environ.copy()
    base_env.setdefault("MPLBACKEND", "Agg")
    base_env.setdefault("MPLCONFIGDIR", str(run_dir / ".mplcache"))

    # Layout within run dir (intentionally mirrors radiomics pipeline repo conventions)
    ct_dir = run_dir / "niftii"
    seg_root = run_dir / "data" / "pancreas" / args.patient_id
    seg_dir = run_dir / "data" / "pancreas"

    outputs_dir = run_dir / "outputs"
    out_head_dir = outputs_dir / "pancreatic_heads_manual_extracted"
    out_ct_head_dir = outputs_dir / "ct_head_data"
    out_radiomics_dir = outputs_dir / "radiomics_yaml"
    out_pred_dir = outputs_dir / "prediction"

    manifest_path = run_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "patient_id": args.patient_id,
        "created_at": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "inputs": {},
        "steps": [],
        "outputs": {},
    }

    # Stage inputs
    ct_src = args.ct_nifti.resolve()
    seg_src = args.pancreas_seg.resolve()
    if not ct_src.exists():
        raise FileNotFoundError(f"CT not found: {ct_src}")
    if not seg_src.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_src}")

    ct_dst = ct_dir / f"{args.patient_id}{ct_src.suffix}"
    # Preserve .nii.gz suffix if present
    if ct_src.name.endswith(".nii.gz"):
        ct_dst = ct_dir / f"{args.patient_id}.nii.gz"
    seg_dst = seg_root / "pancreas.nii.gz"
    if seg_src.name.endswith(".nii.gz"):
        seg_dst = seg_root / "pancreas.nii.gz"
    else:
        seg_dst = seg_root / f"pancreas{seg_src.suffix}"

    _copy_or_link(ct_src, ct_dst, link=args.link_inputs)
    _copy_or_link(seg_src, seg_dst, link=args.link_inputs)

    manifest["inputs"]["ct_nifti"] = str(ct_dst)
    manifest["inputs"]["pancreas_seg"] = str(seg_dst)

    # Prepare coordinates JSON
    coordinates_path = run_dir / "head_coordinates.json"
    if args.coordinates_json is not None:
        _load_coordinates_subset(args.coordinates_json.resolve(), patient_id=args.patient_id, output_path=coordinates_path)
        manifest["inputs"]["coordinates_json_source"] = str(args.coordinates_json.resolve())
    else:
        _write_coordinates_from_args(args, output_path=coordinates_path)
        manifest["inputs"]["coordinates_json_source"] = "generated_from_args"
    manifest["inputs"]["coordinates_json"] = str(coordinates_path)

    # Step 1: head mask extraction
    print("[RADPANC] Step 1/5 · Extract pancreatic head mask", flush=True)
    cmd1 = [
        sys.executable,
        str(radiomics_scripts["extract_head"]),
        "--ct-dir",
        str(ct_dir),
        "--seg-dir",
        str(seg_dir),
        "--coordinates-file",
        str(coordinates_path),
        "--patient-id",
        args.patient_id,
        "--output-dir",
        str(out_head_dir),
    ]
    manifest["steps"].append({"name": "extract_pancreatic_head", "cmd": cmd1})
    _run(cmd1, cwd=paths.radiomics_repo, env=base_env, dry_run=args.dry_run)

    # Step 2: crop CT to head ROI
    print("[RADPANC] Step 2/5 · Crop CT to head ROI", flush=True)
    cmd2 = [
        sys.executable,
        str(radiomics_scripts["extract_ct"]),
        "--ct-dir",
        str(ct_dir),
        "--head-dir",
        str(out_head_dir),
        "--output-dir",
        str(out_ct_head_dir),
    ]
    manifest["steps"].append({"name": "extract_ct_head", "cmd": cmd2})
    _run(cmd2, cwd=paths.radiomics_repo, env=base_env, dry_run=args.dry_run)

    # Step 3: radiomics extraction (YAML)
    print("[RADPANC] Step 3/5 · Extract radiomics (PyRadiomics YAML)", flush=True)
    default_config = paths.radiomics_repo / "code" / "configs" / "radiomics_config_2mm.yaml"
    radiomics_config = args.radiomics_config.resolve() if args.radiomics_config else default_config.resolve()
    cmd3 = [
        sys.executable,
        str(radiomics_scripts["extract_radiomics"]),
        "--input-dir",
        str(out_ct_head_dir),
        "--output-dir",
        str(out_radiomics_dir),
        "--config",
        str(radiomics_config),
    ]
    manifest["steps"].append({"name": "extract_radiomics", "cmd": cmd3})
    _run(cmd3, cwd=paths.radiomics_repo, env=base_env, dry_run=args.dry_run)

    if args.dry_run:
        _write_json(manifest_path, manifest)
        print(f"[DRY RUN] Wrote manifest: {manifest_path}")
        return

    features_csv = _latest_matching(out_radiomics_dir, "radiomics_yaml_*.csv")
    manifest["outputs"]["radiomics_csv"] = str(features_csv)

    # Optional step 4: ComBat apply
    features_for_model = features_csv
    if not args.skip_combat and args.combat_estimates and args.combat_scanner_id:
        print("[RADPANC] Step 4/5 · Apply ComBat harmonization", flush=True)
        out_combat_csv = out_radiomics_dir / "radiomics_yaml_combat.csv"
        cmd4 = [
            sys.executable,
            str(radiomics_scripts["combat_apply"]),
            "--features-csv",
            str(features_csv),
            "--estimates-pkl",
            str(args.combat_estimates.resolve()),
            "--scanner-id",
            str(args.combat_scanner_id),
            "--output-csv",
            str(out_combat_csv),
        ]
        manifest["steps"].append({"name": "combat_apply", "cmd": cmd4})
        try:
            _run(cmd4, cwd=paths.radiomics_repo, env=base_env, dry_run=False)
            features_for_model = out_combat_csv
            manifest["outputs"]["radiomics_combat_csv"] = str(out_combat_csv)
        except subprocess.CalledProcessError as e:
            manifest["outputs"]["combat_error"] = f"exit_code={e.returncode}"
            # Proceed without ComBat (unknown scanner/batch or other issue)
            features_for_model = features_csv

    # Step 5: prediction (raw + calibrated)
    print("[RADPANC] Step 5/5 · Predict POPF risk (raw + calibrated)", flush=True)
    model_pkl = args.model_pkl.resolve() if args.model_pkl else (paths.analysis_repo / "configs" / "exported_model.pkl").resolve()
    out_pred_csv = out_pred_dir / "popf_predictions.csv"
    out_pred_meta = out_pred_dir / "popf_predictions_meta.json"

    cmd5 = [
        sys.executable,
        str(analysis_scripts["predict"]),
        "--model-pkl",
        str(model_pkl),
        "--features-csv",
        str(features_for_model),
        "--id-col",
        "patient_id",
        "--patient-id",
        args.patient_id,
        "--output-csv",
        str(out_pred_csv),
        "--output-meta",
        str(out_pred_meta),
    ]
    if args.no_calibration:
        cmd5.append("--no-calibration")
    if args.calibration_json is not None:
        cmd5.extend(["--calibration-json", str(args.calibration_json.resolve())])

    manifest["steps"].append({"name": "predict_popf", "cmd": cmd5})
    _run(cmd5, cwd=paths.analysis_repo, env=base_env, dry_run=False)

    manifest["outputs"]["prediction_csv"] = str(out_pred_csv)
    manifest["outputs"]["prediction_meta"] = str(out_pred_meta)

    _write_json(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Prediction: {out_pred_csv}")


if __name__ == "__main__":
    main()
