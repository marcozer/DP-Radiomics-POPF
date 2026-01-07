#!/usr/bin/env python3
"""
Prefetch TotalSegmentator weights into the local cache.

This is meant for deployment environments where the pipeline must run
*offline* (or where downloads are flaky). It downloads only the weights
needed for the configured task/roi_subset, using the same resumable downloader
as `deployment/totalseg_wrapper.py`.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch TotalSegmentator weights for RADPANC")
    p.add_argument("--task", default=os.environ.get("RADPANC_TOTALSEG_TASK", "total"))
    p.add_argument(
        "--roi-subset",
        nargs="+",
        default=None,
        help="ROI subset (space-separated). Defaults to RADPANC_TOTALSEG_ROI_SUBSET if set.",
    )
    p.add_argument("--roi-subset-robust", nargs="+", default=None)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--fastest", action="store_true")
    p.add_argument("--robust-crop", action="store_true")
    p.add_argument(
        "--download-retries",
        type=int,
        default=int(os.environ.get("RADPANC_TOTALSEG_DOWNLOAD_RETRIES", "10")),
    )
    p.add_argument(
        "--download-timeout",
        type=float,
        default=float(os.environ.get("RADPANC_TOTALSEG_DOWNLOAD_TIMEOUT", "300")),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.fast and args.fastest:
        raise SystemExit("Only one of --fast / --fastest can be set.")

    if args.roi_subset is None:
        env_val = os.environ.get("RADPANC_TOTALSEG_ROI_SUBSET", "pancreas").strip()
        args.roi_subset = [v for v in env_val.replace(",", " ").split() if v]

    # Reuse the wrapper's logic to keep behaviour consistent.
    # NOTE: this script is executed as `python deployment/prefetch_totalseg_weights.py`,
    # so `sys.path[0]` is the `deployment/` directory (not repo root). Import the wrapper
    # as a sibling module to avoid requiring `deployment` to be a package.
    from totalseg_wrapper import _compute_required_totalseg_task_ids, _resumable_download_url_and_unpack

    required = _compute_required_totalseg_task_ids(
        task=args.task,
        roi_subset=args.roi_subset,
        roi_subset_robust=args.roi_subset_robust,
        fast=args.fast,
        fastest=args.fastest,
        robust_crop=args.robust_crop,
    )
    if required is None or not required:
        print("No task filtering applied; refusing to prefetch an unknown weight set.")
        return 2

    required_all = set(required)

    # If the deployment server is configured to auto-fallback to --fast/--fastest on OOM,
    # prefetch those weights too so inference can run offline after container startup.
    allow_fallback = os.environ.get("RADPANC_TOTALSEG_ALLOW_FAST_FALLBACK", "1").strip().lower() in {"1", "true", "yes"}
    fallback_mode = os.environ.get("RADPANC_TOTALSEG_FAST_FALLBACK_MODE", "fast").strip().lower()
    if allow_fallback and not args.fast and not args.fastest and fallback_mode in {"fast", "fastest"}:
        fallback_required = _compute_required_totalseg_task_ids(
            task=args.task,
            roi_subset=args.roi_subset,
            roi_subset_robust=args.roi_subset_robust,
            fast=fallback_mode == "fast",
            fastest=fallback_mode == "fastest",
            robust_crop=args.robust_crop,
        )
        if fallback_required:
            required_all |= set(fallback_required)

    import totalsegmentator.libs as ts_libs

    lock_path = Path(os.environ.get("TOTALSEG_HOME_DIR", str(Path.home()))) / ".radpanc_totalseg_download.lock"
    ts_libs.download_url_and_unpack = lambda url, config_dir: _resumable_download_url_and_unpack(
        url,
        Path(config_dir),
        retries=max(1, args.download_retries),
        timeout_seconds=args.download_timeout,
        lock_path=lock_path,
    )

    print(f"Prefetching TotalSegmentator weights for task={args.task} roi_subset={args.roi_subset}")
    print(f"Required task IDs: {sorted(required_all)}")
    for tid in sorted(required_all):
        print(f"Ensuring weights for Task {tid} ...")
        ts_libs.download_pretrained_weights(int(tid))

    print("OK: weights present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
