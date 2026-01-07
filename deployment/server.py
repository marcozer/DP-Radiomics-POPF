#!/usr/bin/env python3
"""
RADPANC deployment server (MVP).

Goals:
- Provide a minimal web UI + API to run the end-to-end pipeline on new patients.
- Treat `radiomics pipeline/` and `primary analysis/` as black boxes, running them
  through `deployment/radpanc_runner.py` (CLI contract-checked).
- Store uploads in a shared storage layout so the existing head-selection viewer
  can be pointed at the same data via env vars:
    RADPANC_CT_DIR=<storage>/niftii
    RADPANC_SEG_DIR=<storage>/data/pancreas
    RADPANC_COORDINATES_PATH=<storage>/coordinates/x_coordinate_selections.json
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, url_for

import logging


LOGGER = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    val = os.environ.get(name)
    return Path(val).expanduser().resolve() if val else default.resolve()


PATIENT_ID_ALLOWED = re.compile(r"[^A-Za-z0-9_.-]+")


def normalize_patient_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("patient_id is required")
    normalized = PATIENT_ID_ALLOWED.sub("_", raw)
    return normalized.strip("_") or "patient"


def _latest_glob(path: Path, pattern: str) -> Path | None:
    matches = sorted(path.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _tail_text_file(path: Path, *, max_bytes: int = 20_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _format_dir_listing(path: Path, *, limit: int = 50) -> str:
    if not path.exists():
        return f"{path} (missing)"
    entries: list[tuple[str, int]] = []
    for p in sorted(path.rglob("*")):
        if not p.exists() or p.is_dir():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            size = -1
        try:
            rel = str(p.relative_to(path))
        except Exception:
            rel = str(p)
        entries.append((rel, size))
        if len(entries) >= limit:
            break
    if not entries:
        return f"{path} (no files)"
    body = "\n".join(f"  - {rel} ({size} bytes)" for rel, size in entries)
    return f"{path} (showing up to {limit} files)\n{body}"


def _sitk_convert_dicom_to_nifti(*, dicom_dir: Path, output_nii_gz: Path) -> dict[str, Any]:
    """
    Fallback DICOM→NIfTI conversion using SimpleITK (GDCM).

    This is used when `dcm2niix` fails to produce NIfTI outputs for a given export.
    """
    import SimpleITK as sitk

    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise FileNotFoundError(f"No DICOM series IDs found in {dicom_dir}")

    best_series_id = None
    best_files: list[str] = []
    for series_id in series_ids:
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_dir), series_id)
        if len(files) > len(best_files):
            best_series_id = series_id
            best_files = list(files)

    if not best_series_id or not best_files:
        raise FileNotFoundError(f"No DICOM files found for any series in {dicom_dir}")

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()

    output_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(output_nii_gz), True)

    def _tag(key: str) -> str | None:
        if reader.HasMetaDataKey(0, key):
            val = reader.GetMetaData(0, key).strip()
            return val or None
        return None

    return {
        "converter": "SimpleITK",
        "dicom_dir": str(dicom_dir),
        "series_id": best_series_id,
        "file_count": len(best_files),
        # Useful DICOM tags for auditability / scanner mapping
        "dicom_tags": {
            "modality": _tag("0008|0060"),
            "manufacturer": _tag("0008|0070"),
            "model": _tag("0008|1090"),
            "series_description": _tag("0008|103e"),
            "study_date": _tag("0008|0020"),
            "series_number": _tag("0020|0011"),
        },
        # Useful image attributes (SimpleITK direction is LPS-based)
        "image": {
            "size": list(image.GetSize()),
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
        },
        "output_nifti": str(output_nii_gz),
    }


def _guess_scanner_group_collapsed(
    *,
    manufacturer: str | None,
    model: str | None,
    scanner_groups_config: dict[str, Any] | None,
) -> str | None:
    """
    Heuristic mapping that mirrors `radiomics pipeline/data/scanner_groups_config_*.json`.

    Output is the *collapsed* batch label (e.g., GE_Revolution_HD, Siemens_All).
    """
    man = (manufacturer or "").strip()
    mdl = (model or "").strip()
    man_l = man.lower()
    mdl_l = mdl.lower()

    # Prefer explicit GE Revolution mapping if present
    if "ge" in man_l:
        if mdl == "Revolution HD":
            return "GE_Revolution_HD"
        if mdl in {
            "Revolution EVO",
            "Revolution Apex",
            "Revolution CT",
            "Revolution Frontier",
            "Revolution Maxima",
        }:
            return "GE_Revolution_Other"
        if "revolution" in mdl_l:
            return "GE_Revolution_Other"
        if "optima" in mdl_l or "discovery" in mdl_l:
            return "Other"
        return "Other"

    if "siemens" in man_l:
        return "Siemens_All"
    if "philips" in man_l:
        return "Philips_All"
    if "canon" in man_l or "toshiba" in man_l:
        return "Canon_Toshiba_All"

    # Fallback to "Other" if the config suggests it exists.
    if scanner_groups_config and isinstance(scanner_groups_config.get("batch_structure"), dict):
        if "Other" in scanner_groups_config["batch_structure"]:
            return "Other"

    return None


def _case_page_extras() -> dict[str, Any]:
    """
    Shared template context for `case.html` (used by both normal and error renders).
    """
    scanner_groups: list[str] = []
    try:
        cfg_path = _latest_glob(((_root() / SETTINGS.radiomics_repo / "data").resolve()), "scanner_groups_config_*.json")
        cfg = _read_json(cfg_path) if cfg_path else None
        if isinstance(cfg, dict):
            bs = cfg.get("batch_structure")
            if isinstance(bs, dict):
                scanner_groups = sorted([str(k) for k in bs.keys()])
    except Exception:
        scanner_groups = []

    default_combat_estimates = None
    try:
        candidate = ((_root() / SETTINGS.radiomics_repo / "outputs_combat" / "combat_estimates.pkl").resolve())
        if candidate.exists():
            default_combat_estimates = str(candidate)
    except Exception:
        default_combat_estimates = None

    return {
        "scanner_groups": scanner_groups,
        "default_combat_estimates": default_combat_estimates,
    }

@dataclass(frozen=True)
class Settings:
    storage_root: Path
    runs_root: Path
    viewer_url: str
    radiomics_repo: str
    analysis_repo: str
    max_workers: int
    autorun_on_coordinate: bool
    autorun_interval_seconds: int
    dcm2niix_bin: str
    totalseg_bin: str
    totalseg_task: str
    totalseg_roi_subset: str
    totalseg_fast: bool
    totalseg_robust_crop: bool
    totalseg_device: str
    totalseg_nr_thr_resamp: int
    totalseg_nr_thr_saving: int
    totalseg_force_split: bool
    totalseg_allow_fast_fallback: bool
    totalseg_fast_fallback_mode: str

    @property
    def ct_dir(self) -> Path:
        return self.storage_root / "niftii"

    @property
    def seg_dir(self) -> Path:
        return self.storage_root / "data" / "pancreas"

    @property
    def coordinates_path(self) -> Path:
        return self.storage_root / "coordinates" / "x_coordinate_selections.json"

    @property
    def dicom_dir(self) -> Path:
        return self.storage_root / "dicom"

    @property
    def cases_dir(self) -> Path:
        return self.storage_root / "cases"

    @property
    def jobs_dir(self) -> Path:
        return self.storage_root / "jobs"


def load_settings() -> Settings:
    root = _root()
    storage_root = _env_path("RADPANC_STORAGE_ROOT", root / "deployment" / "storage")
    runs_root = _env_path("RADPANC_RUNS_ROOT", root / "runs")
    viewer_url = os.environ.get("RADPANC_VIEWER_URL", "http://localhost:5000/").strip() or "http://localhost:5000/"
    radiomics_repo = os.environ.get("RADPANC_RADIOMICS_REPO", "radiomics pipeline")
    analysis_repo = os.environ.get("RADPANC_ANALYSIS_REPO", "primary analysis")
    max_workers = int(os.environ.get("RADPANC_MAX_JOBS", "1"))
    autorun_on_coordinate = os.environ.get("RADPANC_AUTORUN_ON_COORDINATE", "0").strip().lower() in {"1", "true", "yes"}
    autorun_interval_seconds = int(os.environ.get("RADPANC_AUTORUN_INTERVAL_SECONDS", "2"))
    dcm2niix_bin = os.environ.get("RADPANC_DCM2NIIX_BIN", "dcm2niix").strip() or "dcm2niix"
    totalseg_bin = os.environ.get("RADPANC_TOTALSEG_BIN", "TotalSegmentator").strip() or "TotalSegmentator"
    totalseg_task = os.environ.get("RADPANC_TOTALSEG_TASK", "total").strip() or "total"
    totalseg_roi_subset = os.environ.get("RADPANC_TOTALSEG_ROI_SUBSET", "pancreas").strip() or "pancreas"
    totalseg_fast = os.environ.get("RADPANC_TOTALSEG_FAST", "0").strip().lower() in {"1", "true", "yes"}
    totalseg_robust_crop = os.environ.get("RADPANC_TOTALSEG_ROBUST_CROP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    totalseg_device = os.environ.get("RADPANC_TOTALSEG_DEVICE", "gpu").strip() or "gpu"
    totalseg_nr_thr_resamp = int(os.environ.get("RADPANC_TOTALSEG_NR_THR_RESAMP", "1"))
    totalseg_nr_thr_saving = int(os.environ.get("RADPANC_TOTALSEG_NR_THR_SAVING", "6"))
    totalseg_force_split = os.environ.get("RADPANC_TOTALSEG_FORCE_SPLIT", "0").strip().lower() in {"1", "true", "yes"}
    totalseg_allow_fast_fallback = os.environ.get("RADPANC_TOTALSEG_ALLOW_FAST_FALLBACK", "1").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    totalseg_fast_fallback_mode = os.environ.get("RADPANC_TOTALSEG_FAST_FALLBACK_MODE", "fast").strip().lower() or "fast"
    if totalseg_fast_fallback_mode not in {"fast", "fastest"}:
        totalseg_fast_fallback_mode = "fast"
    return Settings(
        storage_root=storage_root,
        runs_root=runs_root,
        viewer_url=viewer_url,
        radiomics_repo=radiomics_repo,
        analysis_repo=analysis_repo,
        max_workers=max(1, max_workers),
        autorun_on_coordinate=autorun_on_coordinate,
        autorun_interval_seconds=max(1, autorun_interval_seconds),
        dcm2niix_bin=dcm2niix_bin,
        totalseg_bin=totalseg_bin,
        totalseg_task=totalseg_task,
        totalseg_roi_subset=totalseg_roi_subset,
        totalseg_fast=totalseg_fast,
        totalseg_robust_crop=totalseg_robust_crop,
        totalseg_device=totalseg_device,
        totalseg_nr_thr_resamp=max(1, totalseg_nr_thr_resamp),
        totalseg_nr_thr_saving=max(1, totalseg_nr_thr_saving),
        totalseg_force_split=totalseg_force_split,
        totalseg_allow_fast_fallback=totalseg_allow_fast_fallback,
        totalseg_fast_fallback_mode=totalseg_fast_fallback_mode,
    )


SETTINGS = load_settings()

APP = Flask(__name__)
APP.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("RADPANC_MAX_UPLOAD_BYTES", str(2 * 1024**3)))  # 2GB default

EXECUTOR = ThreadPoolExecutor(max_workers=SETTINGS.max_workers)
JOBS_LOCK = threading.Lock()
JOBS: dict[str, dict[str, Any]] = {}


def _ensure_storage_layout() -> None:
    SETTINGS.ct_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.seg_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.dicom_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.cases_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.jobs_dir.mkdir(parents=True, exist_ok=True)
    SETTINGS.coordinates_path.parent.mkdir(parents=True, exist_ok=True)
    if not SETTINGS.coordinates_path.exists():
        SETTINGS.coordinates_path.write_text("{}")

    # Some tools (e.g., TotalSegmentator) use Path.home() to determine cache/model dirs.
    # In Docker Compose we set HOME to a path under the shared storage volume, but we
    # must ensure it exists (TotalSegmentator creates ~/.totalsegmentator without parents).
    home = os.environ.get("HOME")
    if home:
        Path(home).expanduser().mkdir(parents=True, exist_ok=True)


def _case_meta_path(patient_id: str) -> Path:
    return SETTINGS.cases_dir / f"{patient_id}.json"


def _load_case_meta(patient_id: str) -> dict[str, Any] | None:
    path = _case_meta_path(patient_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_case_meta(patient_id: str, payload: dict[str, Any]) -> None:
    path = _case_meta_path(patient_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _job_path(job_id: str) -> Path:
    return SETTINGS.jobs_dir / f"{job_id}.json"


def _write_job(job: dict[str, Any]) -> None:
    path = _job_path(job["job_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, indent=2))


def _update_job_fields(job_id: str, **updates: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job.update(updates)
        job["updated_at"] = _now_iso()
        _write_job(job)


def _read_jobs_from_disk() -> None:
    if not SETTINGS.jobs_dir.exists():
        return
    for path in sorted(SETTINGS.jobs_dir.glob("*.json")):
        try:
            job = json.loads(path.read_text())
            job_id = job.get("job_id")
            if isinstance(job_id, str) and job_id:
                JOBS[job_id] = job
        except Exception:
            continue


def _ct_path(patient_id: str) -> Path:
    return SETTINGS.ct_dir / f"{patient_id}.nii.gz"


def _seg_path(patient_id: str) -> Path:
    return SETTINGS.seg_dir / patient_id / "pancreas.nii.gz"


def _save_nifti_upload(file_storage, *, dst_path: Path) -> None:
    filename = (file_storage.filename or "").lower()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if filename.endswith(".nii.gz"):
        file_storage.save(dst_path)
        return

    if filename.endswith(".nii"):
        with gzip.open(dst_path, "wb") as gz_out:
            shutil.copyfileobj(file_storage.stream, gz_out)
        return

    raise ValueError("Only .nii or .nii.gz uploads are supported.")


def _save_dicom_zip_upload(file_storage, *, dst_path: Path) -> None:
    filename = (file_storage.filename or "").lower()
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not filename.endswith(".zip"):
        raise ValueError("Only .zip uploads are supported for DICOM ingest.")
    file_storage.save(dst_path)


def _largest_nifti(path: Path) -> Path:
    candidates = list(path.rglob("*.nii.gz")) + list(path.rglob("*.nii"))
    if not candidates:
        raise FileNotFoundError(f"No NIfTI files produced in {path}")
    return max(candidates, key=lambda p: p.stat().st_size)


def _gzip_to(src: Path, dst_nii_gz: Path) -> None:
    dst_nii_gz.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as f_in, gzip.open(dst_nii_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def _assert_affine_and_shape_match(ct_path: Path, seg_path: Path) -> None:
    import nibabel as nib
    import numpy as np

    ct = nib.load(ct_path)
    seg = nib.load(seg_path)
    if ct.shape != seg.shape:
        raise ValueError(f"CT/seg shape mismatch: ct={ct.shape}, seg={seg.shape}")
    if not np.allclose(ct.affine, seg.affine):
        raise ValueError("CT/seg affine mismatch (alignment would be compromised)")


def _extract_scanner_from_dcm2niix_sidecar(json_path: Path) -> tuple[str | None, str | None]:
    """
    Returns (Manufacturer, ManufacturerModelName) when available.
    dcm2niix JSON key names can vary slightly across versions.
    """
    payload = _read_json(json_path)
    manufacturer = payload.get("Manufacturer")
    model = payload.get("ManufacturersModelName") or payload.get("ManufacturerModelName")
    if isinstance(manufacturer, str):
        manufacturer = manufacturer.strip() or None
    else:
        manufacturer = None
    if isinstance(model, str):
        model = model.strip() or None
    else:
        model = None
    return manufacturer, model


def _iter_nonhidden_files(root: Path) -> Any:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__MACOSX"]
        for name in filenames:
            if name.startswith("."):
                continue
            yield Path(dirpath) / name


def _sample_nonhidden_files(root: Path, *, limit: int = 25) -> list[Path]:
    out: list[Path] = []
    for p in _iter_nonhidden_files(root):
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _select_dicom_input_dir(extracted_root: Path) -> Path:
    """
    Pick a sensible directory to point dcm2niix at.

    Many PACS exports zip a *single top-level folder* that contains the DICOM series.
    dcm2niix can miss data when invoked on the wrong level, so we unwrap and then pick
    the subtree with the most files (excluding hidden / __MACOSX).
    """
    root = extracted_root
    ignore_files = {"DICOMDIR", "VERSION", "LOCKFILE"}

    # Unwrap single top-level folder wrappers (common with Finder zips).
    for _ in range(10):
        children = [p for p in root.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]
        files = [p for p in children if p.is_file()]
        dirs = [p for p in children if p.is_dir()]
        if files:
            break
        if len(dirs) != 1:
            break
        root = dirs[0]

    # If the selected root has relevant files directly (not just DICOMDIR/VERSION wrappers), use it.
    direct_files = [
        p for p in root.iterdir() if p.is_file() and not p.name.startswith(".") and p.name not in ignore_files
    ]
    if direct_files:
        return root

    # Otherwise, pick the directory with the most files.
    best_dir = root
    best_count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and d != "__MACOSX"]
        count = sum(1 for f in filenames if f and not f.startswith(".") and f not in ignore_files)
        if count > best_count:
            best_count = count
            best_dir = Path(dirpath)

    return best_dir


def _clear_directory(path: Path) -> None:
    if path.exists():
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    continue
    else:
        path.mkdir(parents=True, exist_ok=True)


def _get_saved_coordinate(patient_id: str) -> dict[str, Any] | None:
    if not SETTINGS.coordinates_path.exists():
        return None
    try:
        all_coords = json.loads(SETTINGS.coordinates_path.read_text())
    except Exception:
        return None
    if not isinstance(all_coords, dict):
        return None
    entry = all_coords.get(patient_id)
    return entry if isinstance(entry, dict) else None


def _has_active_job(patient_id: str) -> bool:
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.get("patient_id") == patient_id and job.get("status") in {"queued", "running"}:
                return True
    return False


def _create_job(
    *,
    patient_id: str,
    ct_path: Path,
    seg_path: Path,
    use_viewer_coordinate: bool,
    head_x: float | None,
    head_y: float | None,
    head_z_limit: int | None,
    skip_combat: bool,
    combat_estimates: str | None,
    combat_scanner_id: str | None,
    no_calibration: bool,
    calibration_json: str | None,
) -> str:
    job_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SETTINGS.runs_root / f"{patient_id}_{timestamp}_{job_id[:8]}"

    job: dict[str, Any] = {
        "kind": "predict",
        "job_id": job_id,
        "patient_id": patient_id,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "queued",
        "phase": "queued",
        "run_dir": str(run_dir),
        "inputs": {"ct_nifti": str(ct_path), "pancreas_seg": str(seg_path)},
        "use_viewer_coordinate": bool(use_viewer_coordinate),
        "head_x": head_x,
        "head_y": head_y,
        "head_z_limit": head_z_limit,
        "skip_combat": bool(skip_combat),
        "combat_estimates": combat_estimates,
        "combat_scanner_id": combat_scanner_id,
        "no_calibration": bool(no_calibration),
        "calibration_json": calibration_json,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job
        _write_job(job)

    EXECUTOR.submit(_run_job, job_id)
    return job_id


def _create_preprocess_job(*, patient_id: str, dicom_zip: Path) -> str:
    job_id = uuid.uuid4().hex
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = SETTINGS.runs_root / f"{patient_id}_{timestamp}_{job_id[:8]}_preprocess"

    job: dict[str, Any] = {
        "kind": "preprocess",
        "job_id": job_id,
        "patient_id": patient_id,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "status": "queued",
        "phase": "queued",
        "run_dir": str(run_dir),
        "inputs": {"dicom_zip": str(dicom_zip)},
        "outputs": {},
    }

    with JOBS_LOCK:
        JOBS[job_id] = job
        _write_job(job)

    EXECUTOR.submit(_run_job, job_id)
    return job_id


def _coordinate_fingerprint(entry: dict[str, Any]) -> str:
    relevant = {k: entry.get(k) for k in ("x_coordinate", "y_coordinate", "z_limit", "timestamp")}
    return json.dumps(relevant, sort_keys=True, default=str)


def _autorun_tick() -> None:
    if not SETTINGS.coordinates_path.exists():
        return

    try:
        coords = json.loads(SETTINGS.coordinates_path.read_text())
    except Exception:
        return

    if not isinstance(coords, dict):
        return

    for raw_patient_id, entry in coords.items():
        if not isinstance(raw_patient_id, str) or not isinstance(entry, dict):
            continue

        patient_id = normalize_patient_id(raw_patient_id)

        ct_path = _ct_path(patient_id)
        seg_path = _seg_path(patient_id)
        if not ct_path.exists() or not seg_path.exists():
            continue

        if _has_active_job(patient_id):
            continue

        fingerprint = _coordinate_fingerprint(entry)

        meta = _load_case_meta(patient_id) or {"patient_id": patient_id, "created_at": _now_iso()}
        if meta.get("last_autorun_coordinate_fingerprint") == fingerprint:
            continue

        combat_scanner_id = None
        combat_enabled = bool(meta.get("combat_enabled"))
        combat_estimates = meta.get("combat_estimates") if isinstance(meta.get("combat_estimates"), str) else None
        # Prefer explicit ComBat scanner_id, else fall back to the case scanner_id.
        scanner_id = meta.get("combat_scanner_id") if isinstance(meta.get("combat_scanner_id"), str) else meta.get("scanner_id")
        if isinstance(scanner_id, str) and scanner_id:
            combat_scanner_id = scanner_id
        if not combat_enabled or not combat_estimates or not combat_scanner_id:
            combat_enabled = False

        job_id = _create_job(
            patient_id=patient_id,
            ct_path=ct_path,
            seg_path=seg_path,
            use_viewer_coordinate=True,
            head_x=None,
            head_y=None,
            head_z_limit=None,
            skip_combat=not combat_enabled,
            combat_estimates=combat_estimates if combat_enabled else None,
            combat_scanner_id=combat_scanner_id if combat_enabled else None,
            no_calibration=False,
            calibration_json=None,
        )

        meta["updated_at"] = _now_iso()
        meta["last_autorun_coordinate_fingerprint"] = fingerprint
        meta["last_autorun_job_id"] = job_id
        _write_case_meta(patient_id, meta)
        LOGGER.info("Autorun queued for %s (job %s)", patient_id, job_id[:8])


def _autorun_loop() -> None:  # pragma: no cover
    while True:
        try:
            _autorun_tick()
        except Exception as exc:
            LOGGER.warning("Autorun tick failed: %s", exc)
        time.sleep(SETTINGS.autorun_interval_seconds)


def _run_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        return

    kind = str(job.get("kind", "predict"))
    if kind == "preprocess":
        _run_preprocess_job(job_id)
        return
    if kind == "predict":
        _run_predict_job(job_id)
        return

    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "failed"
        job["updated_at"] = _now_iso()
        job["error"] = {"type": "ValueError", "message": f"Unknown job kind: {kind}"}
        _write_job(job)


def _run_preprocess_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["phase"] = "starting"
        job["phase_detail"] = "Starting preprocess"
        job["started_at"] = _now_iso()
        job["updated_at"] = job["started_at"]
        _write_job(job)

    patient_id = job["patient_id"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "job.log"

    dicom_zip = Path(job["inputs"]["dicom_zip"])
    dicom_extract_dir = run_dir / "dicom_extracted"
    dcm2niix_out = run_dir / "dcm2niix_out"

    ct_dst = _ct_path(patient_id)
    seg_dst = _seg_path(patient_id)
    seg_dst.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        try:
            log_file.write(f"[{_now_iso()}] Preprocess job for {patient_id}\n")
            log_file.write(f"DICOM zip: {dicom_zip}\n\n")

            if not dicom_zip.exists():
                raise FileNotFoundError(f"DICOM zip not found: {dicom_zip}")

            _update_job_fields(job_id, phase="dicom_extract", phase_detail="Extracting DICOM zip")
            dicom_extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(dicom_zip, "r") as zf:
                zf.extractall(dicom_extract_dir)

            # Convert DICOM → NIfTI (choose largest output as CT)
            dicom_in_dir = _select_dicom_input_dir(dicom_extract_dir)
            log_file.write(f"Selected dcm2niix input dir: {dicom_in_dir}\n")

            dcm2niix_out.mkdir(parents=True, exist_ok=True)
            _clear_directory(dcm2niix_out)
            _update_job_fields(job_id, phase="dcm2niix", phase_detail="Converting DICOM → NIfTI (dcm2niix)")
            cmd_dcm2niix = [
                SETTINGS.dcm2niix_bin,
                "-z",
                "i",
                "-f",
                f"{patient_id}_%s",
                "-o",
                str(dcm2niix_out),
                str(dicom_in_dir),
            ]
            cmd_dcm2niix_attempts = [cmd_dcm2niix]
            log_file.write("Running dcm2niix (attempt 1):\n  " + " ".join(cmd_dcm2niix) + "\n\n")
            subprocess.run(cmd_dcm2niix, stdout=log_file, stderr=log_file, check=True)
            log_file.write("\nDcm2niix output listing (attempt 1):\n" + _format_dir_listing(dcm2niix_out) + "\n\n")

            ct_candidate: Path | None = None
            try:
                ct_candidate = _largest_nifti(dcm2niix_out)
            except FileNotFoundError:
                # Retry without an explicit filename pattern (some PACS exports can trigger name conflicts).
                _clear_directory(dcm2niix_out)
                cmd_dcm2niix = [
                    SETTINGS.dcm2niix_bin,
                    "-z",
                    "i",
                    "-o",
                    str(dcm2niix_out),
                    str(dicom_in_dir),
                ]
                cmd_dcm2niix_attempts.append(cmd_dcm2niix)
                log_file.write("\nNo NIfTI found after attempt 1; retrying dcm2niix (attempt 2):\n  " + " ".join(cmd_dcm2niix) + "\n\n")
                subprocess.run(cmd_dcm2niix, stdout=log_file, stderr=log_file, check=True)
                log_file.write("\nDcm2niix output listing (attempt 2):\n" + _format_dir_listing(dcm2niix_out) + "\n\n")

            try:
                ct_candidate = _largest_nifti(dcm2niix_out)
            except FileNotFoundError:
                # Fallback: some users may upload a zip that already contains NIfTI(s).
                nifti_in_zip = list(dicom_extract_dir.rglob("*.nii.gz")) + list(dicom_extract_dir.rglob("*.nii"))
                if nifti_in_zip:
                    ct_candidate = max(nifti_in_zip, key=lambda p: p.stat().st_size)
                    log_file.write(
                        "\nWARNING: dcm2niix produced no NIfTI outputs; using NIfTI found inside the uploaded zip:\n"
                        f"  {ct_candidate}\n"
                    )
                else:
                    # Final fallback: SimpleITK DICOM→NIfTI conversion.
                    log_file.write(
                        "\nNo NIfTI produced by dcm2niix; falling back to SimpleITK (GDCM) DICOM→NIfTI conversion.\n"
                    )
                    _clear_directory(dcm2niix_out)
                    sitk_out = dcm2niix_out / f"{patient_id}_sitk.nii.gz"
                    sitk_report = _sitk_convert_dicom_to_nifti(dicom_dir=dicom_in_dir, output_nii_gz=sitk_out)
                    log_file.write("SimpleITK conversion report:\n" + json.dumps(sitk_report, indent=2) + "\n\n")
                    ct_candidate = sitk_out

            assert ct_candidate is not None
            log_file.write(f"\nSelected CT NIfTI: {ct_candidate}\n")
            _update_job_fields(job_id, phase="store_ct", phase_detail="Storing CT NIfTI")

            # Attempt to capture scanner metadata from dcm2niix sidecar JSON.
            sidecar = None
            name = ct_candidate.name
            if name.endswith(".nii.gz"):
                sidecar = ct_candidate.with_name(name[:-7] + ".json")
            elif name.endswith(".nii"):
                sidecar = ct_candidate.with_name(name[:-4] + ".json")

            manufacturer = None
            model = None
            scanner_group = None

            if sidecar and sidecar.exists():
                manufacturer, model = _extract_scanner_from_dcm2niix_sidecar(sidecar)
            else:
                # If we didn't get a dcm2niix JSON sidecar (or we used the SimpleITK fallback),
                # attempt to extract scanner metadata from DICOM tags via SimpleITK.
                try:
                    import SimpleITK as sitk

                    series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dicom_in_dir))
                    if series_ids:
                        series_files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dicom_in_dir), series_ids[0])
                        if series_files:
                            rdr = sitk.ImageFileReader()
                            rdr.SetFileName(series_files[0])
                            rdr.LoadPrivateTagsOn()
                            rdr.ReadImageInformation()
                            if rdr.HasMetaDataKey("0008|0070"):
                                manufacturer = rdr.GetMetaData("0008|0070").strip() or None
                            if rdr.HasMetaDataKey("0008|1090"):
                                model = rdr.GetMetaData("0008|1090").strip() or None
                except Exception:
                    pass

            if manufacturer or model:
                cfg_path = _latest_glob(
                    (_root() / SETTINGS.radiomics_repo / "data").resolve(), "scanner_groups_config_*.json"
                )
                scanner_groups_cfg = _read_json(cfg_path) if cfg_path else None
                scanner_group = _guess_scanner_group_collapsed(
                    manufacturer=manufacturer, model=model, scanner_groups_config=scanner_groups_cfg
                )

                meta = _load_case_meta(patient_id) or {"patient_id": patient_id, "created_at": _now_iso()}
                meta["updated_at"] = _now_iso()
                meta["dicom_scanner"] = {
                    "manufacturer": manufacturer,
                    "model": model,
                    "scanner_group_collapsed_auto": scanner_group,
                }
                # Only set `scanner_id` automatically if the user hasn't provided one.
                if not meta.get("scanner_id") and scanner_group:
                    meta["scanner_id"] = scanner_group
                _write_case_meta(patient_id, meta)

            if ct_candidate.name.endswith(".nii.gz"):
                ct_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ct_candidate, ct_dst)
            else:
                _gzip_to(ct_candidate, ct_dst)

            # Segment pancreas with TotalSegmentator (high quality = default, no --fast)
            _update_job_fields(job_id, phase="totalseg", phase_detail="Segmenting pancreas (TotalSegmentator)")
            seg_out_dir = SETTINGS.seg_dir / patient_id
            seg_out_dir.mkdir(parents=True, exist_ok=True)
            # Split on whitespace and/or commas (e.g. "pancreas liver" or "pancreas,liver").
            roi_subset = [v for v in re.split(r"[\s,]+", SETTINGS.totalseg_roi_subset) if v]
            primary_roi = roi_subset[0] if roi_subset else SETTINGS.totalseg_roi_subset

            wrapper = (_root() / "deployment" / "totalseg_wrapper.py").resolve()
            log_file.write(f"\nTotalSegmentator wrapper: {wrapper} (exists={wrapper.exists()})\n")
            log_file.write(f"TotalSegmentator device: {SETTINGS.totalseg_device}\n")
            log_file.write(
                "TotalSegmentator threads: "
                f"nr_thr_resamp={SETTINGS.totalseg_nr_thr_resamp} "
                f"nr_thr_saving={SETTINGS.totalseg_nr_thr_saving}\n"
            )
            log_file.write(f"TotalSegmentator force_split: {SETTINGS.totalseg_force_split}\n")
            if wrapper.exists():
                cmd_totalseg = [
                    sys.executable,
                    str(wrapper),
                    "-i",
                    str(ct_dst),
                    "-o",
                    str(seg_out_dir),
                    "--task",
                    SETTINGS.totalseg_task,
                    "--device",
                    SETTINGS.totalseg_device,
                    "--nr-thr-resamp",
                    str(SETTINGS.totalseg_nr_thr_resamp),
                    "--nr-thr-saving",
                    str(SETTINGS.totalseg_nr_thr_saving),
                ]
                if roi_subset:
                    cmd_totalseg.extend(["--roi-subset", *roi_subset])
                if SETTINGS.totalseg_fast:
                    cmd_totalseg.append("--fast")
                if SETTINGS.totalseg_robust_crop:
                    cmd_totalseg.append("--robust-crop")
                if SETTINGS.totalseg_force_split:
                    cmd_totalseg.append("--force-split")
            else:
                cmd_totalseg = [
                    SETTINGS.totalseg_bin,
                    "-i",
                    str(ct_dst),
                    "-o",
                    str(seg_out_dir),
                    "--task",
                    SETTINGS.totalseg_task,
                    "--roi_subset",
                    primary_roi,
                    "--nr_thr_resamp",
                    str(SETTINGS.totalseg_nr_thr_resamp),
                    "--nr_thr_saving",
                    str(SETTINGS.totalseg_nr_thr_saving),
                    "--device",
                    SETTINGS.totalseg_device,
                ]
                if SETTINGS.totalseg_fast:
                    cmd_totalseg.append("--fast")
                if SETTINGS.totalseg_robust_crop:
                    cmd_totalseg.append("--robust_crop")
                if SETTINGS.totalseg_force_split:
                    cmd_totalseg.append("--force_split")

            def _is_sigkill(returncode: int) -> bool:
                return returncode in {-9, 137} or (returncode < 0 and abs(returncode) == 9)

            totalseg_env = os.environ.copy()
            # Reduce the risk of oversubscription / OOM on CPU-only Docker setups.
            totalseg_env.setdefault("OMP_NUM_THREADS", "1")
            totalseg_env.setdefault("MKL_NUM_THREADS", "1")
            totalseg_env.setdefault("OPENBLAS_NUM_THREADS", "1")
            totalseg_env.setdefault("NUMEXPR_NUM_THREADS", "1")
            totalseg_env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "1")

            def _run_totalseg(cmd: list[str], *, label: str) -> None:
                log_file.write(f"\nRunning TotalSegmentator ({label}):\n  " + " ".join(cmd) + "\n\n")
                subprocess.run(cmd, stdout=log_file, stderr=log_file, check=True, env=totalseg_env)

            try:
                _run_totalseg(cmd_totalseg, label="attempt 1")
            except subprocess.CalledProcessError as exc:
                should_retry = (
                    _is_sigkill(exc.returncode)
                    and SETTINGS.totalseg_allow_fast_fallback
                    and not SETTINGS.totalseg_fast
                )
                if not should_retry:
                    raise

                mode = SETTINGS.totalseg_fast_fallback_mode
                fallback_cmd = [c for c in cmd_totalseg if c not in {"--fast", "--fastest"}]
                fallback_cmd.append("--fastest" if mode == "fastest" else "--fast")
                log_file.write(
                    "\nTotalSegmentator was killed (SIGKILL). "
                    "This is typically an out-of-memory condition. "
                    f"Retrying once with {mode} mode.\n"
                )
                _run_totalseg(fallback_cmd, label=f"fallback ({mode})")
                cmd_totalseg = fallback_cmd

            expected = seg_out_dir / f"{primary_roi}.nii.gz"
            if expected.exists() and expected != seg_dst:
                shutil.copy2(expected, seg_dst)
            if not seg_dst.exists() and expected.exists():
                # If expected path already matches seg_dst (same file), we're done.
                pass
            if not seg_dst.exists():
                # Fallback: single output nifti in seg_out_dir
                candidates = list(seg_out_dir.glob("*.nii.gz"))
                if len(candidates) == 1:
                    shutil.copy2(candidates[0], seg_dst)
                else:
                    raise FileNotFoundError(
                        f"Pancreas segmentation not found at {seg_dst}. "
                        f"Expected {expected}. Outputs: {[p.name for p in candidates][:20]}"
                    )

            _update_job_fields(job_id, phase="alignment_check", phase_detail="Checking CT/seg alignment")
            _assert_affine_and_shape_match(ct_dst, seg_dst)

            outputs = {
                "ct_nifti": str(ct_dst),
                "pancreas_seg": str(seg_dst),
                "dcm2niix_cmd": cmd_dcm2niix,
                "dcm2niix_cmd_attempts": cmd_dcm2niix_attempts,
                "totalseg_cmd": cmd_totalseg,
                "dicom_manufacturer": manufacturer,
                "dicom_model": model,
                "scanner_group_collapsed_auto": scanner_group,
            }
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "completed"
                job["phase"] = "completed"
                job["phase_detail"] = "Preprocess complete"
                job["updated_at"] = _now_iso()
                job["outputs"] = outputs
                job["log"] = str(log_path)
                _write_job(job)
            return
        except subprocess.CalledProcessError as exc:
            message = None
            if exc.returncode in (-9, 137) or (exc.returncode < 0 and abs(exc.returncode) == 9):
                message = (
                    "A subprocess was killed (SIGKILL). In Docker this is usually an out-of-memory condition. "
                    "Increase Docker memory and/or set RADPANC_TOTALSEG_FAST=1 (or RADPANC_TOTALSEG_FORCE_SPLIT=1)."
                )
            if message is None:
                message = str(exc)
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "failed"
                job["phase"] = "failed"
                job["updated_at"] = _now_iso()
                job["error"] = {
                    "type": "CalledProcessError",
                    "returncode": exc.returncode,
                    "log": str(log_path),
                    "message": message,
                }
                job["log"] = str(log_path)
                _write_job(job)
            return
        except Exception as exc:
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "failed"
                job["phase"] = "failed"
                job["updated_at"] = _now_iso()
                job["error"] = {"type": type(exc).__name__, "message": str(exc), "log": str(log_path)}
                job["log"] = str(log_path)
                _write_job(job)
            return


def _run_predict_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "running"
        job["phase"] = "pipeline"
        job["phase_detail"] = "Running head→radiomics→prediction pipeline"
        job["started_at"] = _now_iso()
        job["updated_at"] = job["started_at"]
        _write_job(job)

    patient_id = job["patient_id"]
    ct_path = Path(job["inputs"]["ct_nifti"])
    seg_path = Path(job["inputs"]["pancreas_seg"])
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "job.log"

    cmd = [
        sys.executable,
        str((_root() / "deployment" / "radpanc_runner.py").resolve()),
        "--patient-id",
        patient_id,
        "--ct-nifti",
        str(ct_path),
        "--pancreas-seg",
        str(seg_path),
        "--run-dir",
        str(run_dir),
        "--radiomics-repo",
        SETTINGS.radiomics_repo,
        "--analysis-repo",
        SETTINGS.analysis_repo,
        "--link-inputs",
    ]

    if job.get("head_x") is not None:
        cmd.extend(["--head-x", str(job["head_x"])])
        if job.get("head_y") is not None:
            cmd.extend(["--head-y", str(job["head_y"])])
        if job.get("head_z_limit") is not None:
            cmd.extend(["--head-z-limit", str(job["head_z_limit"])])
    else:
        cmd.extend(["--coordinates-json", str(SETTINGS.coordinates_path)])

    if job.get("skip_combat"):
        cmd.append("--skip-combat")
    else:
        combat_estimates = job.get("combat_estimates")
        combat_scanner_id = job.get("combat_scanner_id")
        if combat_estimates and combat_scanner_id:
            cmd.extend(["--combat-estimates", str(combat_estimates), "--combat-scanner-id", str(combat_scanner_id)])

    if job.get("no_calibration"):
        cmd.append("--no-calibration")
    calibration_json = job.get("calibration_json")
    if calibration_json:
        cmd.extend(["--calibration-json", str(calibration_json)])

    with open(log_path, "w") as log_file:
        try:
            subprocess.run(cmd, cwd=str(_root()), stdout=log_file, stderr=log_file, check=True)
        except subprocess.CalledProcessError as exc:
            # Provide a compact hint for common inference failures.
            log_file.flush()
            tail = _tail_text_file(log_path)
            message = None
            if "AttributeError: 'SimpleImputer' object has no attribute '_fill_dtype'" in tail:
                message = (
                    "Prediction failed due to a scikit-learn version mismatch with the exported model bundle. "
                    "Rebuild the container (deployment image pins scikit-learn==1.5.2)."
                )
            elif "InconsistentVersionWarning" in tail and "unpickle estimator" in tail:
                message = (
                    "Prediction failed after loading a model trained with a different scikit-learn version. "
                    "Rebuild the container to use the pinned scikit-learn version for inference."
                )
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "failed"
                job["phase"] = "failed"
                job["updated_at"] = _now_iso()
                job["error"] = {
                    "type": "CalledProcessError",
                    "returncode": exc.returncode,
                    "log": str(log_path),
                    "message": message,
                }
                _write_job(job)
            return
        except Exception as exc:
            with JOBS_LOCK:
                job = JOBS[job_id]
                job["status"] = "failed"
                job["phase"] = "failed"
                job["updated_at"] = _now_iso()
                job["error"] = {"type": type(exc).__name__, "message": str(exc), "log": str(log_path)}
                _write_job(job)
            return

    manifest_path = run_dir / "manifest.json"
    outputs: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
        except Exception:
            outputs = {}

    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = "completed"
        job["phase"] = "completed"
        job["phase_detail"] = "Prediction complete"
        job["updated_at"] = _now_iso()
        job["outputs"] = outputs
        job["log"] = str(log_path)
        job["manifest"] = str(manifest_path)
        _write_job(job)


_ensure_storage_layout()
_read_jobs_from_disk()
if SETTINGS.autorun_on_coordinate:
    threading.Thread(target=_autorun_loop, daemon=True).start()


def _list_cases() -> list[dict[str, Any]]:
    cases = []
    for path in sorted(SETTINGS.cases_dir.glob("*.json")):
        try:
            meta = json.loads(path.read_text())
        except Exception:
            continue
        patient_id = meta.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            continue
        cases.append(meta)
    return cases


def _list_jobs() -> list[dict[str, Any]]:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    return sorted(jobs, key=lambda j: j.get("created_at", ""), reverse=True)


@APP.get("/api/cases")
def api_cases() -> Any:
    """
    List cases with lightweight status for the UI (case switcher).
    """
    out: list[dict[str, Any]] = []
    for meta in _list_cases():
        patient_id = meta.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id:
            continue
        latest_predict = _latest_job_for_patient(patient_id, "predict")
        has_prediction = bool(latest_predict and latest_predict.get("status") == "completed")
        out.append(
            {
                "patient_id": patient_id,
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "scanner_id": meta.get("scanner_id"),
                "ct_present": _ct_path(patient_id).exists(),
                "seg_present": _seg_path(patient_id).exists(),
                "has_coordinate": _get_saved_coordinate(patient_id) is not None,
                "has_prediction": has_prediction,
                "last_dicom_zip": meta.get("last_dicom_zip"),
                "dicom_zip_count": len(meta.get("dicom_zips", [])) if isinstance(meta.get("dicom_zips"), list) else 0,
            }
        )
    out.sort(key=lambda c: str(c.get("updated_at") or c.get("created_at") or ""), reverse=True)
    return jsonify({"cases": out})

def _latest_job_for_patient(patient_id: str, kind: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        jobs = [
            j
            for j in JOBS.values()
            if j.get("patient_id") == patient_id and j.get("kind") == kind and isinstance(j.get("created_at"), str)
        ]
    if not jobs:
        return None
    return max(jobs, key=lambda j: j.get("created_at", ""))


def _job_public(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job:
        return None
    return {
        "job_id": job.get("job_id"),
        "kind": job.get("kind"),
        "patient_id": job.get("patient_id"),
        "status": job.get("status"),
        "phase": job.get("phase"),
        "phase_detail": job.get("phase_detail"),
        "created_at": job.get("created_at"),
        "started_at": job.get("started_at"),
        "updated_at": job.get("updated_at"),
        "error": job.get("error"),
        "outputs": job.get("outputs"),
    }


def _prediction_from_job(job: dict[str, Any] | None) -> dict[str, Any] | None:
    if not job or job.get("status") != "completed":
        return None
    outputs = job.get("outputs") if isinstance(job.get("outputs"), dict) else {}
    pred_path_raw = outputs.get("prediction_csv")
    if not isinstance(pred_path_raw, str):
        return None
    pred_path = Path(pred_path_raw)
    if not pred_path.exists():
        return None
    try:
        import pandas as pd
        import numpy as np

        df = pd.read_csv(pred_path)
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        cleaned: dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                cleaned[k] = None
                continue
            # pandas uses NaN for missing; jsonify can't encode it
            try:
                if isinstance(v, float) and np.isnan(v):
                    cleaned[k] = None
                    continue
            except Exception:
                pass
            # numpy scalars -> python scalars
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                cleaned[k] = v.item()
                continue
            cleaned[k] = v
        return cleaned
    except Exception:
        return None


@APP.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "time": _now_iso()})


@APP.get("/")
def index() -> Any:
    cases = _list_cases()
    if cases:
        # Default to the most recently created/updated case to keep the UI "single-page" focused.
        def _key(c: dict[str, Any]) -> str:
            return str(c.get("updated_at") or c.get("created_at") or "")

        latest = max(cases, key=_key)
        pid = latest.get("patient_id")
        if isinstance(pid, str) and pid:
            return redirect(url_for("case_detail", patient_id=pid))

    return render_template("index.html", settings=SETTINGS, cases=cases, jobs=[])


@APP.post("/cases")
def create_case() -> Any:
    patient_id = normalize_patient_id(request.form.get("patient_id", ""))
    scanner_id = (request.form.get("scanner_id") or "").strip() or None

    existing = _load_case_meta(patient_id)
    if existing is None:
        meta: dict[str, Any] = {
            "patient_id": patient_id,
            "created_at": _now_iso(),
        }
        if scanner_id:
            meta["scanner_id"] = scanner_id
        _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/quickstart")
def quickstart() -> Any:
    """
    Single form: create/open case + upload DICOM zip + start preprocess, then redirect to the case page.
    """
    patient_id = normalize_patient_id(request.form.get("patient_id", ""))
    scanner_id = (request.form.get("scanner_id") or "").strip() or None

    meta = _load_case_meta(patient_id)
    if meta is None:
        meta = {"patient_id": patient_id, "created_at": _now_iso()}
    meta["updated_at"] = _now_iso()
    if scanner_id:
        meta["scanner_id"] = scanner_id
    _write_case_meta(patient_id, meta)

    dicom_zip = request.files.get("dicom_zip")
    if not dicom_zip or not dicom_zip.filename:
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=["Missing dicom_zip upload (.zip)."],
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = SETTINGS.dicom_dir / patient_id / f"{timestamp}_{uuid.uuid4().hex[:8]}.zip"
    try:
        _save_dicom_zip_upload(dicom_zip, dst_path=dst)
    except Exception as exc:
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=[f"DICOM upload failed: {exc}"],
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    meta = _load_case_meta(patient_id) or meta
    meta["updated_at"] = _now_iso()
    meta["last_dicom_zip"] = str(dst)
    zips = meta.get("dicom_zips")
    if not isinstance(zips, list):
        zips = []
    zips.append(
        {
            "path": str(dst),
            "uploaded_at": _now_iso(),
            "filename": dicom_zip.filename,
            "size_bytes": dst.stat().st_size if dst.exists() else None,
        }
    )
    meta["dicom_zips"] = zips
    _write_case_meta(patient_id, meta)

    job_id = _create_preprocess_job(patient_id=patient_id, dicom_zip=dst)
    meta = _load_case_meta(patient_id) or meta
    meta["updated_at"] = _now_iso()
    meta["last_preprocess_job_id"] = job_id
    _write_case_meta(patient_id, meta)

    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.get("/cases/<patient_id>")
def case_detail(patient_id: str) -> Any:
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    ct_present = _ct_path(patient_id).exists()
    seg_present = _seg_path(patient_id).exists()
    saved_coordinate = _get_saved_coordinate(patient_id)

    return render_template(
        "case.html",
        settings=SETTINGS,
        case=meta,
        ct_present=ct_present,
        seg_present=seg_present,
        saved_coordinate=saved_coordinate,
        **_case_page_extras(),
    )


@APP.post("/cases/<patient_id>/scanner")
def update_scanner(patient_id: str) -> Any:
    """
    Update the scanner/batch label for a case.

    This is used for ComBat (if enabled) and for documenting the acquisition source.
    """
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    scanner_id = (request.form.get("scanner_id") or "").strip() or None
    meta["updated_at"] = _now_iso()
    if scanner_id is None:
        meta.pop("scanner_id", None)
    else:
        meta["scanner_id"] = scanner_id
    _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/cases/<patient_id>/combat")
def update_combat(patient_id: str) -> Any:
    """
    Update ComBat settings for a case.

    These settings are used by autorun (after Save Selection) and manual prediction runs.
    """
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    enabled = request.form.get("combat_enabled") == "on"
    estimates = (request.form.get("combat_estimates") or "").strip() or None
    scanner_id = (request.form.get("combat_scanner_id") or "").strip() or None

    meta["updated_at"] = _now_iso()
    meta["combat_enabled"] = bool(enabled)
    if estimates is None:
        meta.pop("combat_estimates", None)
    else:
        meta["combat_estimates"] = estimates
    if scanner_id is None:
        meta.pop("combat_scanner_id", None)
    else:
        meta["combat_scanner_id"] = scanner_id
        # Keep the case-level scanner label in sync for UI display and defaults.
        meta["scanner_id"] = scanner_id

    _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/cases/<patient_id>/delete")
def delete_case(patient_id: str) -> Any:
    """
    Delete a case and its stored artifacts from `deployment/storage/`.

    This is best-effort and intentionally conservative: it removes CT/seg/DICOM zips
    and the case metadata entry; it does not delete run outputs under `runs/`.
    """
    patient_id = normalize_patient_id(patient_id)
    path = _case_meta_path(patient_id)
    if not path.exists():
        abort(404)

    # Remove from coordinates file (best-effort).
    try:
        if SETTINGS.coordinates_path.exists():
            coords = json.loads(SETTINGS.coordinates_path.read_text())
            if isinstance(coords, dict):
                coords.pop(patient_id, None)
                SETTINGS.coordinates_path.write_text(json.dumps(coords, indent=2))
    except Exception:
        pass

    # Remove stored artifacts (best-effort).
    try:
        _ct_path(patient_id).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        shutil.rmtree(_seg_path(patient_id).parent, ignore_errors=True)
    except Exception:
        pass
    try:
        shutil.rmtree(SETTINGS.dicom_dir / patient_id, ignore_errors=True)
    except Exception:
        pass

    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass

    return redirect(url_for("index"))


@APP.post("/cases/<patient_id>/upload")
def upload_case_files(patient_id: str) -> Any:
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    ct_file = request.files.get("ct_nifti")
    seg_file = request.files.get("pancreas_seg")

    errors: list[str] = []
    if ct_file and ct_file.filename:
        try:
            _save_nifti_upload(ct_file, dst_path=_ct_path(patient_id))
        except Exception as exc:
            errors.append(f"CT upload failed: {exc}")

    if seg_file and seg_file.filename:
        try:
            _save_nifti_upload(seg_file, dst_path=_seg_path(patient_id))
        except Exception as exc:
            errors.append(f"Seg upload failed: {exc}")

    # Alignment guardrail: if both exist, verify shape+affine match.
    ct_path = _ct_path(patient_id)
    seg_path = _seg_path(patient_id)
    if ct_path.exists() and seg_path.exists():
        try:
            _assert_affine_and_shape_match(ct_path, seg_path)
        except Exception as exc:
            errors.append(f"Alignment check failed: {exc}")

    if errors:
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=errors,
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/cases/<patient_id>/upload_dicom")
def upload_dicom_zip(patient_id: str) -> Any:
    """
    Upload a zipped DICOM series. The server will convert to NIfTI and run
    TotalSegmentator pancreas segmentation in a background job.
    """
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    dicom_zip = request.files.get("dicom_zip")
    if not dicom_zip or not dicom_zip.filename:
        abort(400, description="Missing dicom_zip upload (.zip).")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = SETTINGS.dicom_dir / patient_id / f"{timestamp}_{uuid.uuid4().hex[:8]}.zip"
    try:
        _save_dicom_zip_upload(dicom_zip, dst_path=dst)
    except Exception as exc:
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=[f"DICOM upload failed: {exc}"],
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    meta["updated_at"] = _now_iso()
    meta["last_dicom_zip"] = str(dst)
    zips = meta.get("dicom_zips")
    if not isinstance(zips, list):
        zips = []
    zips.append(
        {
            "path": str(dst),
            "uploaded_at": _now_iso(),
            "filename": dicom_zip.filename,
            "size_bytes": dst.stat().st_size if dst.exists() else None,
        }
    )
    meta["dicom_zips"] = zips
    _write_case_meta(patient_id, meta)

    job_id = _create_preprocess_job(patient_id=patient_id, dicom_zip=dst)
    meta = _load_case_meta(patient_id) or meta
    meta["updated_at"] = _now_iso()
    meta["last_preprocess_job_id"] = job_id
    _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/cases/<patient_id>/rerun_preprocess")
def rerun_preprocess(patient_id: str) -> Any:
    """
    Re-run preprocess for an existing case using the most recently uploaded DICOM zip.

    This avoids re-uploading large PACS exports when troubleshooting failures.
    """
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    if _has_active_job(patient_id):
        return redirect(url_for("case_detail", patient_id=patient_id))

    last_zip = meta.get("last_dicom_zip")
    if not isinstance(last_zip, str) or not last_zip:
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=["No previous DICOM upload found for this case."],
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    zip_path = Path(last_zip)
    if not zip_path.exists():
        return (
            render_template(
                "case.html",
                settings=SETTINGS,
                case=meta,
                errors=[f"Last DICOM zip not found on disk: {zip_path}"],
                ct_present=_ct_path(patient_id).exists(),
                seg_present=_seg_path(patient_id).exists(),
                saved_coordinate=_get_saved_coordinate(patient_id),
                **_case_page_extras(),
            ),
            400,
        )

    job_id = _create_preprocess_job(patient_id=patient_id, dicom_zip=zip_path)
    meta = _load_case_meta(patient_id) or meta
    meta["updated_at"] = _now_iso()
    meta["last_preprocess_job_id"] = job_id
    _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.post("/cases/<patient_id>/run")
def run_case(patient_id: str) -> Any:
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)

    ct_path = _ct_path(patient_id)
    seg_path = _seg_path(patient_id)
    if not ct_path.exists() or not seg_path.exists():
        abort(400, description="CT and pancreas segmentation must be uploaded before running.")

    use_viewer = request.form.get("use_viewer_coordinate") == "on"
    head_x_raw = (request.form.get("head_x") or "").strip()
    head_y_raw = (request.form.get("head_y") or "").strip()
    head_z_raw = (request.form.get("head_z_limit") or "").strip()

    head_x: float | None = None
    head_y: float | None = None
    head_z_limit: int | None = None

    if not use_viewer:
        if not head_x_raw:
            abort(400, description="Provide head_x or enable 'use viewer coordinate'.")
        head_x = float(head_x_raw)
        if head_y_raw:
            head_y = float(head_y_raw)
        if head_z_raw:
            head_z_limit = int(head_z_raw)
    else:
        saved = _get_saved_coordinate(patient_id)
        if saved is None:
            abort(400, description=f"No saved coordinate for {patient_id} found in {SETTINGS.coordinates_path}")

    # ComBat defaults: use case-level config unless explicitly overridden by the form.
    skip_combat: bool
    if "skip_combat" in request.form:
        skip_combat = request.form.get("skip_combat") == "on"
    else:
        skip_combat = not bool(meta.get("combat_enabled"))

    combat_estimates = (request.form.get("combat_estimates") or "").strip() or meta.get("combat_estimates") or None
    combat_scanner_id = (
        (request.form.get("combat_scanner_id") or "").strip()
        or meta.get("combat_scanner_id")
        or meta.get("scanner_id")
        or None
    )

    job_id = _create_job(
        patient_id=patient_id,
        ct_path=ct_path,
        seg_path=seg_path,
        use_viewer_coordinate=use_viewer,
        head_x=head_x,
        head_y=head_y,
        head_z_limit=head_z_limit,
        skip_combat=bool(skip_combat),
        combat_estimates=combat_estimates,
        combat_scanner_id=combat_scanner_id,
        no_calibration=False,
        calibration_json=None,
    )
    meta["updated_at"] = _now_iso()
    meta["last_predict_job_id"] = job_id
    _write_case_meta(patient_id, meta)
    return redirect(url_for("case_detail", patient_id=patient_id))


@APP.get("/jobs/<job_id>")
def job_detail(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)

    patient_id = job.get("patient_id")
    if isinstance(patient_id, str) and patient_id:
        return redirect(url_for("case_detail", patient_id=patient_id) + f"?job={job_id}")
    abort(404)


@APP.get("/api/jobs/<job_id>")
def api_job(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


def _extract_tqdm_percent(log_tail: str) -> int | None:
    """
    Heuristic: pull the last "NN%|" token used by tqdm progress bars.
    """
    matches = re.findall(r"(\\d{1,3})%\\|", log_tail)
    if not matches:
        return None
    try:
        val = int(matches[-1])
    except Exception:
        return None
    return val if 0 <= val <= 100 else None


def _extract_latest_stage_line(log_tail: str) -> str | None:
    """
    Extract the latest stage line printed by the deployment runner.
    """
    lines = [ln.strip() for ln in log_tail.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.startswith("[RADPANC] Step"):
            return ln
    return None


@APP.get("/api/jobs/<job_id>/log")
def api_job_log(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)

    tail_bytes = request.args.get("tail_bytes", "20000")
    try:
        tail_bytes_i = max(1_000, min(200_000, int(tail_bytes)))
    except Exception:
        tail_bytes_i = 20_000

    log_tail = ""
    log_path_raw = job.get("log")
    if isinstance(log_path_raw, str):
        path = Path(log_path_raw)
        if path.exists():
            log_tail = _tail_text_file(path, max_bytes=tail_bytes_i)

    return jsonify(
        {
            "job_id": job.get("job_id"),
            "kind": job.get("kind"),
            "status": job.get("status"),
            "phase": job.get("phase"),
            "phase_detail": job.get("phase_detail"),
            "progress_percent": _extract_tqdm_percent(log_tail),
            "stage_hint": _extract_latest_stage_line(log_tail),
            "log_tail": log_tail,
        }
    )


@APP.get("/api/cases/<patient_id>")
def api_case(patient_id: str) -> Any:
    patient_id = normalize_patient_id(patient_id)
    meta = _load_case_meta(patient_id)
    if meta is None:
        abort(404)
    latest_preprocess = _latest_job_for_patient(patient_id, "preprocess")
    latest_predict = _latest_job_for_patient(patient_id, "predict")
    return jsonify(
        {
            "case": meta,
            "ct_nifti": str(_ct_path(patient_id)),
            "pancreas_seg": str(_seg_path(patient_id)),
            "ct_present": _ct_path(patient_id).exists(),
            "seg_present": _seg_path(patient_id).exists(),
            "saved_coordinate": _get_saved_coordinate(patient_id),
            "latest_preprocess_job": _job_public(latest_preprocess),
            "latest_predict_job": _job_public(latest_predict),
            "latest_prediction": _prediction_from_job(latest_predict),
        }
    )


@APP.get("/jobs/<job_id>/files/manifest.json")
def download_manifest(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)
    manifest = job.get("manifest")
    if not isinstance(manifest, str):
        abort(404)
    path = Path(manifest)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="application/json", as_attachment=True, download_name="manifest.json")


@APP.get("/jobs/<job_id>/files/popf_predictions.csv")
def download_predictions(job_id: str) -> Any:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        abort(404)
    outputs = job.get("outputs")
    if not isinstance(outputs, dict):
        abort(404)
    pred = outputs.get("prediction_csv")
    if not isinstance(pred, str):
        abort(404)
    path = Path(pred)
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="text/csv", as_attachment=True, download_name="popf_predictions.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    host = os.environ.get("RADPANC_HOST", "127.0.0.1")
    port = int(os.environ.get("RADPANC_PORT", "8000"))
    APP.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
