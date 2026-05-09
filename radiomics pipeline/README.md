# Radiomics Pipeline (Pancreatic Head → Radiomics → ComBat → Ready to process dataset for post-operative pancreatic fistula prediction)

This repository contains the imaging pipeline: manual pancreatic head localization, head mask extraction, CT cropping, radiomics extraction, and ComBat harmonization. It is independent from the primary analysis code; handoff happens via CSV outputs.

## Repository layout

- `pancreas_head_delimiter/`: Flask viewer for manual head coordinates.
- `code/`: extraction + harmonization scripts.
- `niftii/`: CT NIfTI files (`<patient>.nii.gz`).
- `data/pancreas/`: pancreas segmentations (see naming conventions below).
- `outputs_*`: generated outputs.

## Dependencies

- Viewer: `pip install -r pancreas_head_delimiter/requirements.txt`
- Pipeline: Python 3.9+ with `nibabel`, `SimpleITK`, `pyradiomics`, `pandas`, `numpy`, `scikit-learn`, `neuroCombat`.

## Data conventions

CTs must be stored as:
- `niftii/<patient>.nii.gz`

Pancreas segmentations are resolved in this order:
- `data/pancreas/<patient>/pancreas.nii.gz`
- `data/pancreas/<patient>_pancreas.nii.gz`
- `data/pancreas/<patient>.nii.gz`
- `data/pancreas/pancreas.nii.gz` (single-test case)

If CT and segmentation live in the same directory, point `--ct-dir` and `--seg-dir` to the same path.

## Alignment requirement (critical)

CT and pancreas segmentation must share the **same affine and shape**. Do not resample either independently. Quick check:

```bash
python - <<'PY'
import nibabel as nib
import numpy as np
ct = nib.load("/path/to/ct.nii.gz")
seg = nib.load("/path/to/pancreas_mask.nii.gz")
print("shape:", ct.shape, seg.shape)
print("affine match:", np.allclose(ct.affine, seg.affine))
PY
```

## End-to-end workflow (single patient)

### 1) Launch the viewer and save coordinates

```bash
cd "radiomics pipeline"
export RADPANC_CT_DIR="niftii"
export RADPANC_SEG_DIR="data/pancreas"
export RADPANC_COORDINATES_PATH="pancreas_head_delimiter/x_coordinate_selections.json"
export RADPANC_VIEWER_HOST="127.0.0.1"
export RADPANC_VIEWER_PORT="5003"
python pancreas_head_delimiter/app.py
```

Open the viewer, select the patient, save the head coordinates. The file is written to:
`pancreas_head_delimiter/x_coordinate_selections.json` (or the path set in `RADPANC_COORDINATES_PATH`).

Optional deep-linking (useful when embedding in another UI):
- Auto-select patient: `http://localhost:5003/?patient=<PATIENT_ID>`
- Embedded mode (hides patient list): `http://localhost:5003/?patient=<PATIENT_ID>&embedded=1`

### 2) Extract pancreatic head mask from viewer coordinates

```bash
RUN_DIR="outputs_run"
PATIENT_ID="test"

python code/extract_pancreatic_head_from_viewer_coordinates.py \
  --ct-dir "$RADPANC_CT_DIR" \
  --seg-dir "$RADPANC_SEG_DIR" \
  --coordinates-file pancreas_head_delimiter/x_coordinate_selections.json \
  --patient-id "$PATIENT_ID" \
  --output-dir "$RUN_DIR/pancreatic_heads_manual_extracted"
```

Omit `--patient-id` to process all entries in the coordinates file.

### 3) Crop CT to the head ROI

```bash
python code/extract_ct_from_head_segmentations.py \
  --ct-dir "$RADPANC_CT_DIR" \
  --head-dir "$RUN_DIR/pancreatic_heads_manual_extracted" \
  --output-dir "$RUN_DIR/ct_head_data"
```

This produces per-patient folders with:
- `ct_head.nii.gz`
- `head_mask_cropped.nii.gz`
- `ct_head_metadata.json`

### 4) Extract radiomics

Recommended (YAML config, matches HF3 feature conventions with LoG 3/5/7):

```bash
python code/extract_radiomics_yaml.py \
  --input-dir "$RUN_DIR/ct_head_data" \
  --config code/configs/radiomics_config_2mm.yaml
```

Outputs are written to `code/outputs_yaml/radiomics_yaml_*.csv`.

Alternate (optimized script with custom output dir):

```bash
python code/extract_radiomics_optimized.py \
  --input-dir "$RUN_DIR/ct_head_data" \
  --output-dir "$RUN_DIR/radiomics_optimized" \
  --log-sigma 3,5,7
```

If you omit `--log-sigma`, the default is `2,3,4`.

## Harmonization (ComBat)

### Fit ComBat once on the reference cohort

```bash
python code/combat_fit.py \
  --features-csv /path/to/radiomics_cohort.csv \
  --scanner-metadata /path/to/scanner_metadata_collapsed.csv \
  --patient-col patient_id \
  --metadata-id-col patient_id \
  --metadata-batch-col scanner_group_collapsed \
  --ref-batch auto \
  --output-estimates outputs_combat/combat_estimates.pkl
```

The scanner batch label **must** exist in the training cohort. New/unseen batches cannot be transformed with existing ComBat estimates.

### Apply ComBat to a new patient

```bash
NEW_FEATURES="$(ls -t "$RUN_DIR"/radiomics_optimized/radiomics_optimized_*.csv | head -1)"

python code/combat_apply.py \
  --features-csv "$NEW_FEATURES" \
  --estimates-pkl outputs_combat/combat_estimates.pkl \
  --scanner-id "<scanner_group_collapsed label>" \
  --output-csv "$RUN_DIR/radiomics_optimized/radiomics_optimized_combat.csv"
```

## Anonymization (optional but recommended for training cohorts)

```bash
python code/prepare_anonymized_cohort.py \
  --radiomics-csv /path/to/raw_cohort_radiomics.csv \
  --mapping-csv /path/to/patient_mapping.csv \
  --outcomes-csv /path/to/outcomes.csv \
  --scanner-metadata-csv /path/to/scanner_metadata.csv \
  --output-dir /path/to/output_dir \
  --delete-identifying
```

## Notes

- Use the YAML config (LoG 3/5/7, wavelet `coif1`) when you plan to apply the POPF model trained on HF3 features in the primary analysis repository.
- `code/harmonize_extended_features.py` is kept for legacy runs; prefer `combat_fit.py`/`combat_apply.py` for current harmonization.
- If you distribute synthetic templates, keep them under `templates/` and document that they contain no PHI.

## POPF risk prediction (primary analysis)

After exporting a radiomics CSV for the new patient, run the inference script in the analysis repository:

```bash
cd "../primary analysis"
python code/predict_popf_risk.py \
  --model-pkl configs/exported_model.pkl \
  --features-csv "../radiomics pipeline/code/outputs_yaml/radiomics_yaml_<timestamp>.csv" \
  --id-col patient_id \
  --patient-id test \
  --output-csv results/predictions_test.csv
```

The default calibration JSON is an identity mapping from the current manuscript analysis, so `popf_risk_calibrated` is retained as a backward-compatible reportable-risk column and equals `popf_risk_raw`.
