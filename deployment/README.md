# RADPANC Deployment Tooling (Design + Runner)

This folder provides a **composition layer** that connects the two repos:

- `radiomics pipeline/`: CT + segmentation → pancreatic head ROI → radiomics (and optional ComBat).
- `primary analysis/`: frozen 7-rad model + frozen reportable-probability/risk-strata configuration → POPF risk.

Key requirement: **updates inside either repo must not silently break deployment**.
We achieve this by treating both repos as black boxes with **stable CLI contracts**.

## What exists today (MVP)

- `deployment/radpanc_runner.py`: an end-to-end CLI that:
  1) stages CT + pancreas segmentation into a run directory,
  2) extracts the pancreatic head mask using head coordinates,
  3) crops the CT to the head ROI,
  4) extracts radiomics using the frozen YAML config,
  5) optionally applies frozen ComBat estimates,
  6) computes raw and reportable 7-rad POPF risk using the exported model bundle.

It writes a `manifest.json` with commands/inputs/outputs so runs are auditable.

- `deployment/server.py`: a minimal web UI + API that lets you upload CT/seg, reuse the head-selection viewer, and run the same pipeline in a background job.

- `deployment/smoke_test.py`: no-dependency smoke tests (contract check + optional end-to-end run on the bundled `test` example).

## CLI contracts (anti-breakage)

`deployment/radpanc_runner.py` runs a help-based contract check before execution.
If any required flag disappears (e.g., a script is refactored), the runner fails early with a clear error.

Contracts are defined in `deployment/contracts.py`. Currently enforced:
- `radiomics pipeline/code/extract_pancreatic_head_from_viewer_coordinates.py` must accept:
  `--ct-dir`, `--seg-dir`, `--coordinates-file`, `--output-dir`, `--patient-id`
- `radiomics pipeline/code/extract_ct_from_head_segmentations.py` must accept:
  `--ct-dir`, `--head-dir`, `--output-dir`
- `radiomics pipeline/code/extract_radiomics_yaml.py` must accept:
  `--config`, `--input-dir`, `--output-dir`
- `radiomics pipeline/code/combat_apply.py` must accept:
  `--features-csv`, `--estimates-pkl`, `--output-csv`
- `primary analysis/code/predict_popf_risk.py` must accept:
  `--model-pkl`, `--features-csv`, `--output-csv`

## Run directory structure (artifact contract)

Each case is executed in an isolated run directory (default: `runs/<patient>_<timestamp>`):

```
runs/<case>/
  niftii/<patient>.nii.gz
  data/pancreas/<patient>/pancreas.nii.gz
  head_coordinates.json
  outputs/
    pancreatic_heads_manual_extracted/<patient>/pancreatic_head.nii.gz
    ct_head_data/<patient>/ct_head.nii.gz
    radiomics_yaml/radiomics_yaml_<timestamp>.csv
    prediction/popf_predictions.csv
  manifest.json
```

## Quickstart (single patient)

Prerequisites:
- CT NIfTI and pancreas segmentation NIfTI are aligned (same affine + shape).
- Head coordinate is available either as:
  - `--coordinates-json` from the viewer, or
  - `--head-x` (and optional Y / Z-limit).

Example:

```bash
python deployment/radpanc_runner.py \
  --patient-id "<PATIENT_ID>" \
  --ct-nifti "/path/to/<PATIENT_ID>.nii.gz" \
  --pancreas-seg "/path/to/pancreas.nii.gz" \
  --head-x 4.7 \
  --combat-estimates "radiomics pipeline/outputs_combat/combat_estimates.pkl" \
  --combat-scanner-id "GE_Revolution_Other"
```

This produces:
- `runs/test_<timestamp>/outputs/prediction/popf_predictions.csv`
- `runs/test_<timestamp>/manifest.json`

To only validate that the runner won’t break due to missing flags:

```bash
python deployment/radpanc_runner.py --patient-id test --ct-nifti x --pancreas-seg y --head-x 0 --check-contracts
```

## Web server (MVP)

This is a lightweight local UI intended for demo / internal use (not hardened).

### 1) Start the head-selection viewer (pointed at deployment storage)

```bash
export RADPANC_CT_DIR="deployment/storage/niftii"
export RADPANC_SEG_DIR="deployment/storage/data/pancreas"
export RADPANC_COORDINATES_PATH="deployment/storage/coordinates/x_coordinate_selections.json"
export RADPANC_VIEWER_HOST="127.0.0.1"
export RADPANC_VIEWER_PORT="5003"

python "radiomics pipeline/pancreas_head_delimiter/app.py"
```

Viewer: `http://localhost:5003`

### 2) Start the RADPANC server

```bash
export RADPANC_VIEWER_URL="http://localhost:5003/"
export RADPANC_AUTORUN_ON_COORDINATE=1
python deployment/server.py
```

Server UI: `http://localhost:8000`

With `RADPANC_AUTORUN_ON_COORDINATE=1`, saving a head coordinate in the viewer will automatically queue a job and compute POPF risk (no manual “Run” click required).

### Single-page workflow

Open a case page: `http://localhost:8000/cases/<patient_id>`

That page includes:
- Central embedded head-selection viewer (the “main screen”).
- Compact side panels for upload, pipeline progress, live logs, and the final 7-rad POPF risk.
- Live **progress + log tail** for preprocess and prediction (no need to open job pages).

### One-page quickstart (minimum clicks)

Open `http://localhost:8000/` and use the **“One-page quickstart”** form (it redirects you to the case page immediately):

1) Enter a patient ID and choose a DICOM `.zip`.
2) Click **“Upload DICOM → segment → open case”**.
3) On the case page, use the embedded viewer and click **Save Selection**.
4) With `RADPANC_AUTORUN_ON_COORDINATE=1`, prediction starts automatically and the risk appears on the same page.

### Storage layout (shared between viewer + server)

```
deployment/storage/
  dicom/<patient>/<timestamp>_<id>.zip
  niftii/<patient>.nii.gz
  data/pancreas/<patient>/pancreas.nii.gz
  coordinates/x_coordinate_selections.json
  cases/<patient>.json
  jobs/<job_id>.json
```

### DICOM → NIfTI → pancreas segmentation (all-in-one)

From the server UI (`http://localhost:8000`):

1) Create/open a case (patient ID).
2) Upload a **zipped DICOM series** (`.zip`) under “Upload DICOM → convert + segment”.
   - The server queues a **preprocess** job:
     - runs `dcm2niix -z i -f <patient>_%s` to create `deployment/storage/niftii/<patient>.nii.gz`
     - runs TotalSegmentator (via `deployment/totalseg_wrapper.py`) to create `deployment/storage/data/pancreas/<patient>/pancreas.nii.gz`
   - The job fails if CT and segmentation do not share the same affine+shape (alignment guardrail).
   - Notes:
     - The `.zip` can contain a single top-level folder (common in Finder/PACS exports); the server auto-selects the best DICOM input subdirectory.
     - PACS exports that include `DICOMDIR` / `VERSION` wrapper files are handled (the server ignores these when selecting the conversion directory).
     - If `dcm2niix` produces no NIfTI outputs on the first try, the server automatically retries once with a simpler `dcm2niix` invocation (no explicit filename pattern).
     - If `dcm2niix` still produces no NIfTI outputs, the server falls back to **SimpleITK (GDCM)** DICOM→NIfTI conversion.
     - If conversion still fails, open the job page and read “Log (tail)” to see the exact converter output and file listings.
   - If preprocessing fails and you don't want to re-upload the zip, use the case page button: **“Re-run preprocess (use last uploaded DICOM)”**.
3) Use the embedded viewer on the case page (or open the viewer directly) and save the head coordinate.
4) With `RADPANC_AUTORUN_ON_COORDINATE=1`, the server automatically queues a **predict** job and writes:
   `runs/<patient>_.../outputs/prediction/popf_predictions.csv`

Optional: set the case “Scanner ID” (batch label) for documentation and ComBat (if you later enable it). If you upload DICOM, the server will try to auto-detect manufacturer/model and suggest a collapsed batch label (e.g. `GE_Revolution_Other`) based on `radiomics pipeline/data/scanner_groups_config_*.json`.

Preprocess settings can be controlled via env vars (defaults shown):

- `RADPANC_DCM2NIIX_BIN=dcm2niix`
- `RADPANC_TOTALSEG_BIN=TotalSegmentator`
- `RADPANC_TOTALSEG_TASK=total`
- `RADPANC_TOTALSEG_ROI_SUBSET=pancreas`
- `RADPANC_TOTALSEG_FAST=0` (keep `0` for high quality; set `1` for faster, lower-quality inference)
- `RADPANC_TOTALSEG_ROBUST_CROP=0` (set `1` to use the 3mm model for cropping instead of 6mm)
- `RADPANC_TOTALSEG_DEVICE=gpu` (or `cpu`, `mps`, `gpu:X`)
- `RADPANC_TOTALSEG_NR_THR_RESAMP=1` (threads for resampling)
- `RADPANC_TOTALSEG_NR_THR_SAVING=6` (threads for saving masks; lowering can help on small Docker memory limits)
- `RADPANC_TOTALSEG_FORCE_SPLIT=0` (set `1` to split large volumes and reduce peak memory)
- `RADPANC_TOTALSEG_ALLOW_FAST_FALLBACK=1` (set `1` to retry once with `--fast/--fastest` when TotalSegmentator is SIGKILLed)
- `RADPANC_TOTALSEG_FAST_FALLBACK_MODE=fast` (or `fastest`)
- `RADPANC_TOTALSEG_DOWNLOAD_RETRIES=5` (resumable download retries for model weights)
- `RADPANC_TOTALSEG_DOWNLOAD_TIMEOUT=300` (seconds; socket read timeout during weight download)

### Troubleshooting: TotalSegmentator gets killed (return code `-9` / `137`)

This usually means Docker killed the process due to **out-of-memory**.

Recommended fixes (pick one):
- Increase Docker Desktop memory (e.g., 8–16GB).
- Set `RADPANC_TOTALSEG_FAST=1` (lower resolution).
- Set `RADPANC_TOTALSEG_FORCE_SPLIT=1` (lower peak memory).
- Keep `RADPANC_TOTALSEG_ALLOW_FAST_FALLBACK=1` to automatically retry once in `fast` mode when SIGKILL happens.

## Docker Compose (local)

```bash
docker compose -f deployment/docker-compose.yml up --build
```

- Note (arm64/aarch64): PyRadiomics is built from source (C extensions), so the Docker image installs compiler toolchains via `build-essential` during build.
- Note: PyRadiomics is installed with `--no-build-isolation` in the Dockerfile because it imports `numpy` during build-time metadata discovery.
- Note: The inference image pins `scikit-learn==1.5.2` because the exported model bundle was pickled with that version. Newer sklearn versions can break unpickling/inference (e.g. `SimpleImputer` attribute errors).
- Note: This image is intended for the end-to-end inference workflow (DICOM→CT NIfTI→pancreas segmentation→head cut→radiomics→prediction). ComBat *fitting* is not part of the container build; you can still apply frozen ComBat estimates if you already have them.
- Note: TotalSegmentator model weights are **prefetched on container startup** and cached under `HOME` (in this compose file: `/app/deployment/storage/home`), so subsequent cases run offline (no downloads during patient processing).
- Note: `deployment/totalseg_wrapper.py` reduces the initial TotalSegmentator downloads for `roi_subset` by skipping model parts that are not needed (e.g., pancreas only needs the “organs” part) and uses a resumable downloader to tolerate flaky connections.
- Note: weights for the configured `RADPANC_TOTALSEG_FAST_FALLBACK_MODE` are also prefetched so the SIGKILL fallback can run offline.
- If you previously attempted a build, use `docker compose -f deployment/docker-compose.yml build --no-cache` to ensure the fixed dependencies are applied.

- Viewer: `http://localhost:5003`
- Server: `http://localhost:8000`

## Smoke tests

Contracts only:

```bash
python deployment/smoke_test.py --contracts-only
```

End-to-end on the bundled `test` example:

```bash
python deployment/smoke_test.py
```

## Reportable Probability

The current manuscript uses the fixed unweighted L2 7-rad probability directly as the reportable probability.

- Intercept-only recalibration was audited but not retained for the manuscript.
- A frozen identity-calibration artifact is stored under:
  `primary analysis/configs/calibration/radiomics_calibration.json`.
- `primary analysis/code/predict_popf_risk.py` still writes `popf_risk_calibrated` for backward compatibility; with the current identity artifact, it equals `popf_risk_raw`.

## Risk stratification (Low / Intermediate / High)

- The deployment assigns a **frozen** risk group based on the reportable 7-rad probability.
- Thresholds are stored under:
  `primary analysis/configs/calibration/radiomics_risk_stratification.json` and are taken from the final threshold
  diagnostics figure in `primary analysis/configs/calibration/radiomics_threshold_diagnostics.svg`.
- The resulting columns appear in `popf_predictions.csv` (e.g., `risk_group`, `risk_threshold_low`, `risk_threshold_high`).

## ComBat deployment rule

Frozen ComBat can only be applied if the patient’s batch label exists in the batches used during ComBat fitting.
If the scanner/batch is unseen, you must either:
- skip ComBat and flag out-of-distribution (OOD), or
- collect an adaptation set from the new batch and update the harmonization strategy (requires re-validation).

## Online deployment plan (PACS-like)

For a production “upload → segment → select head → risk” experience, implement a job-based service:

1) **DICOM ingest**
   - DICOMweb (preferred) or DICOM C-STORE gateway (e.g., Orthanc).
   - Convert to NIfTI (e.g., `dcm2niix`) while preserving geometry.

2) **Pancreas segmentation**
   - TotalSegmentator (GPU service), outputting `pancreas.nii.gz` in CT geometry.

3) **Head selection UI**
   - Reuse `radiomics pipeline/pancreas_head_delimiter/` or integrate an OHIF extension.
   - Store the head cut coordinate as an API payload; do not rely on shared files in multi-user mode.

4) **Radiomics + inference**
   - Call the same CLIs as `deployment/radpanc_runner.py` does.
   - Persist outputs + manifest; return risk to the UI.

5) **PACS write-back (optional)**
   - Store head mask as DICOM SEG and risk report as DICOM SR.

### Stability strategy (do not break on script updates)

- Treat repo scripts as versioned APIs:
  - any breaking CLI/output change requires a major version bump + contract update.
- Keep an integration test / contract check in CI:
  - run `--help` contract checks,
  - run the pipeline on the `test` example with small data.
- Pin dependencies (especially PyRadiomics, SimpleITK, scikit-learn, neuroCombat) in container builds.
