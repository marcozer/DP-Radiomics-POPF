#!/usr/bin/env python3
"""
TotalSegmentator wrapper for RADPANC deployment.

Why this exists
---------------
TotalSegmentator's `python_api.totalsegmentator()` currently downloads weights for *all*
sub-models of the 1.5mm `task=total` pipeline (tasks 291–295) even when `roi_subset`
would only require one part (e.g. pancreas is in the "organs" part only).

In network-restricted / unstable environments this can fail (large downloads), even
though inference would only run a subset of models. This wrapper:

1) Skips weight downloads for model parts that are not needed for the requested `roi_subset`.
2) Replaces the internal downloader with a resumable + retrying implementation to reduce
   failures on interrupted connections.

It preserves the core TotalSegmentator behaviour and does not resample or modify outputs.
Alignment between CT and segmentation must remain identical (affine + shape).
"""

from __future__ import annotations

import argparse
import os
import re
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RADPANC TotalSegmentator wrapper")
    p.add_argument("-i", "--input", required=True, type=Path, help="Input CT NIfTI (.nii/.nii.gz)")
    p.add_argument("-o", "--output", required=True, type=Path, help="Output directory for masks")
    p.add_argument("--task", default="total", help="TotalSegmentator task (default: total)")
    p.add_argument("--roi-subset", nargs="+", default=None, help="ROI subset (space-separated)")
    p.add_argument("--roi-subset-robust", nargs="+", default=None, help="Use robust crop model for these ROIs")
    p.add_argument("--fast", action="store_true", help="Use 3mm fast model")
    p.add_argument("--fastest", action="store_true", help="Use 6mm fastest model")
    p.add_argument("--robust-crop", action="store_true", help="Use 3mm model for cropping instead of 6mm")
    p.add_argument(
        "--nr-thr-resamp",
        type=int,
        default=int(os.environ.get("RADPANC_TOTALSEG_NR_THR_RESAMP", "1")),
        help="Threads for resampling (default: RADPANC_TOTALSEG_NR_THR_RESAMP or 1).",
    )
    p.add_argument(
        "--nr-thr-saving",
        type=int,
        default=int(os.environ.get("RADPANC_TOTALSEG_NR_THR_SAVING", "6")),
        help="Threads for saving (default: RADPANC_TOTALSEG_NR_THR_SAVING or 6).",
    )
    p.add_argument(
        "--force-split",
        action="store_true",
        default=os.environ.get("RADPANC_TOTALSEG_FORCE_SPLIT", "0").strip().lower() in {"1", "true", "yes"},
        help="Force split large volumes to reduce memory usage (default: RADPANC_TOTALSEG_FORCE_SPLIT).",
    )
    p.add_argument(
        "--device",
        default=os.environ.get("RADPANC_TOTALSEG_DEVICE", "gpu"),
        help="Device for TotalSegmentator (gpu/cpu/mps/gpu:X). Default: RADPANC_TOTALSEG_DEVICE or 'gpu'",
    )
    p.add_argument(
        "--download-retries",
        type=int,
        default=int(os.environ.get("RADPANC_TOTALSEG_DOWNLOAD_RETRIES", "5")),
        help="Max retries for weight downloads (default: RADPANC_TOTALSEG_DOWNLOAD_RETRIES or 5)",
    )
    p.add_argument(
        "--download-timeout",
        type=float,
        default=float(os.environ.get("RADPANC_TOTALSEG_DOWNLOAD_TIMEOUT", "300")),
        help="Socket read timeout (seconds) for weight downloads (default: RADPANC_TOTALSEG_DOWNLOAD_TIMEOUT or 300)",
    )
    p.add_argument("--quiet", action="store_true", help="Reduce TotalSegmentator output")
    return p.parse_args()


@contextmanager
def _file_lock(lock_path: Path):
    """
    Simple inter-process file lock (POSIX).

    TotalSegmentator weights are cached in a shared volume; concurrent downloads can
    corrupt the temporary zip. This lock keeps downloads/extraction serialized.
    """
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _parse_total_from_content_range(value: str) -> int | None:
    # Example: "bytes 0-1023/2048"
    m = re.match(r"^bytes\\s+\\d+-\\d+/(\\d+)$", value.strip())
    return int(m.group(1)) if m else None


def _download_with_resume(*, url: str, dst_part: Path, retries: int, timeout_seconds: float) -> None:
    import requests

    dst_part.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    for attempt in range(1, max(1, retries) + 1):
        existing = dst_part.stat().st_size if dst_part.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}

        try:
            with session.get(url, stream=True, headers=headers, timeout=(30, timeout_seconds)) as r:
                if existing and r.status_code == 200:
                    # Server ignored Range; restart from scratch.
                    existing = 0
                    dst_part.unlink(missing_ok=True)
                r.raise_for_status()

                total = None
                if r.status_code == 206:
                    total = _parse_total_from_content_range(r.headers.get("Content-Range", ""))
                    if total is None:
                        # Content-Length is the remaining bytes.
                        try:
                            total = existing + int(r.headers.get("Content-Length", "0"))
                        except ValueError:
                            total = None
                else:
                    try:
                        total = int(r.headers.get("Content-Length", "0")) or None
                    except ValueError:
                        total = None

                # Progress bar is optional; avoid hard-failing if tqdm is absent.
                progress = None
                try:
                    from tqdm import tqdm

                    progress = tqdm(
                        total=total,
                        initial=existing,
                        unit="B",
                        unit_scale=True,
                        desc="Downloading",
                        leave=False,
                    )
                except Exception:
                    progress = None

                mode = "ab" if existing else "wb"
                with open(dst_part, mode) as f:
                    for chunk in r.iter_content(chunk_size=8192 * 16):
                        if not chunk:
                            continue
                        f.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))

                if progress is not None:
                    progress.close()

            if total is not None and dst_part.exists() and dst_part.stat().st_size < total:
                raise OSError(f"Incomplete download: have {dst_part.stat().st_size} bytes, expected {total} bytes")

            return
        except Exception as exc:
            print(f"Download attempt {attempt}/{retries} failed: {type(exc).__name__}: {exc}")
            if attempt >= retries:
                raise
            time.sleep(min(60.0, 3.0 * attempt))


def _resumable_download_url_and_unpack(
    url: str,
    config_dir: Path,
    *,
    retries: int,
    timeout_seconds: float,
    lock_path: Path,
) -> None:
    """
    Replacement for `totalsegmentator.libs.download_url_and_unpack`.
    """
    config_dir = Path(config_dir)
    config_dir.mkdir(exist_ok=True, parents=True)

    # Use a deterministic filename so partial downloads can be resumed.
    zip_path = config_dir / "tmp_download_file.zip"
    zip_part = config_dir / "tmp_download_file.zip.part"

    with _file_lock(lock_path):
        _download_with_resume(url=url, dst_part=zip_part, retries=retries, timeout_seconds=timeout_seconds)

        # Validate and extract.
        zip_path.unlink(missing_ok=True)
        zip_part.replace(zip_path)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                bad = zf.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Corrupt zip member: {bad}")
                zf.extractall(config_dir)
        except Exception:
            # If extraction fails, remove zip to force a clean re-download.
            zip_path.unlink(missing_ok=True)
            raise
        finally:
            zip_path.unlink(missing_ok=True)


def _compute_required_totalseg_task_ids(
    *,
    task: str,
    roi_subset: list[str] | None,
    roi_subset_robust: list[str] | None,
    fast: bool,
    fastest: bool,
    robust_crop: bool,
) -> set[int] | None:
    """
    Returns a set of task IDs that are actually required for the requested configuration.
    If `None`, no filtering is applied (download everything Totalsegmentator asks for).
    """
    if not task.startswith("total"):
        return None

    # total_mr uses different task ids; handle minimally to avoid surprises.
    is_mr = task.endswith("_mr")

    if fast:
        required = {852 if is_mr else 297}
    elif fastest:
        required = {853 if is_mr else 298}
    else:
        if is_mr:
            required = {850, 851}
        else:
            required = {291, 292, 293, 294, 295}

        # For CT total (multimodel), `roi_subset` runs only the model parts that contain those ROIs.
        if roi_subset and not is_mr:
            from totalsegmentator.map_to_binary import class_map_5_parts, map_taskid_to_partname_ct

            map_partname_to_taskid = {v: k for k, v in map_taskid_to_partname_ct.items()}
            needed: set[int] = set()
            for part_name, part_map in class_map_5_parts.items():
                if any(roi in roi_subset for roi in part_map.values()):
                    needed.add(map_partname_to_taskid[part_name])
            if needed:
                required = needed

    # When `roi_subset` is used, TotalSegmentator generates a rough segmentation for cropping.
    if roi_subset or roi_subset_robust:
        if is_mr:
            required.add(852)  # MR always uses 3mm model for cropping
        else:
            use_robust = robust_crop or bool(roi_subset_robust)
            required.add(297 if use_robust else 298)

    return required


def main() -> int:
    args = _parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()

    if args.fast and args.fastest:
        raise SystemExit("Only one of --fast / --fastest can be set.")

    args.output.mkdir(parents=True, exist_ok=True)

    import totalsegmentator.libs as ts_libs
    import totalsegmentator.python_api as ts_api

    required_task_ids = _compute_required_totalseg_task_ids(
        task=args.task,
        roi_subset=args.roi_subset,
        roi_subset_robust=args.roi_subset_robust,
        fast=args.fast,
        fastest=args.fastest,
        robust_crop=args.robust_crop,
    )

    # Patch downloader to be resumable + retrying.
    lock_path = Path(os.environ.get("TOTALSEG_HOME_DIR", str(Path.home()))) / ".radpanc_totalseg_download.lock"
    ts_libs.download_url_and_unpack = lambda url, config_dir: _resumable_download_url_and_unpack(
        url,
        Path(config_dir),
        retries=max(1, args.download_retries),
        timeout_seconds=args.download_timeout,
        lock_path=lock_path,
    )

    # Patch python_api's imported function ref (it was imported as a symbol).
    original_download = ts_api.download_pretrained_weights

    def _download_pretrained_weights_filtered(task_id: int):
        tid = int(task_id)
        if required_task_ids is not None and tid not in required_task_ids:
            if not args.quiet:
                print(f"Skipping weight download for Task {tid} (not required for roi_subset)")
            return None
        return original_download(tid)

    ts_api.download_pretrained_weights = _download_pretrained_weights_filtered  # type: ignore[assignment]

    # Run TotalSegmentator via python API (same behaviour as CLI).
    ts_api.totalsegmentator(
        args.input,
        args.output,
        task=args.task,
        roi_subset=args.roi_subset,
        roi_subset_robust=args.roi_subset_robust,
        fast=args.fast,
        fastest=args.fastest,
        robust_crop=args.robust_crop,
        nr_thr_resamp=args.nr_thr_resamp,
        nr_thr_saving=args.nr_thr_saving,
        force_split=args.force_split,
        device=args.device,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
