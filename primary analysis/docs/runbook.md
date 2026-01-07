# Runbook

All commands assume repository root and an activated virtualenv with `stabl` installed via pip (e.g., `pip install git+https://github.com/gregbellan/Stabl.git@<commit>`).

## 1. Alignment Audit
```bash
python code/scripts/check_alignment.py \
  --radiomics-path data/radiomics_filtered_unsupervised.csv \
  --matches-path data/POPF-SCANNER.csv \
  --out results/alignment_audit
```
Outputs: merged radiomics + outcomes table, unmatched IDs CSV.

## 2. STABL Discovery (full settings)
```bash
python "code/main analysis/popf_stabl_ultra_optimized.py" \
  --radiomics-path data/radiomics_filtered_unsupervised.csv \
  --matches-path data/POPF-SCANNER.csv \
  --model lr --ensemble-runs 20 --n-bootstraps 800 \
  --consensus-threshold 0.65 --artificial-type knockoff \
  --n-features 5 --no-corr-grouping \
  --fdr-start 0.10 --fdr-end 0.80 --fdr-step 0.01 \
  --discovery-only --export-panel configs/panels/publish_lr_panel.txt \
  --output-dir results_reference/publish_discovery
```
Artifacts: frozen panel, feature frequencies, discovery plots.

### Dev-mode sanity (fast)
Use `--ensemble-runs 2 --n-bootstraps 50 --output-dir results_reference/dev_run/discovery` for smoke testing.

## 3. Fixed-Panel Evaluation (primary, the one used for the publication)
```bash
python "code/main analysis/popf_stabl_corrected_parallel_enhanced_v3.py" \
    --radiomics-path data/HF3.csv \
    --matches-path data/POPF-SCANNER.csv \
    --model enlr \
    --ensemble-runs 20 \
    --n-bootstraps 500 \
    --consensus-threshold 0.8 \
    --n-features 100 \
    --positive-grades B,C \
    --n-workers 12 \
    --validation-bootstrap 2000 \
    --variance-threshold 0.01 \
    --max-nan-fraction 0.2 \
    --impute-strategy median \
    --artificial-type random_permutation \
    --corr-group-threshold 99 \
    --fdr-start 0.8 --fdr-end 0.99 --fdr-step 0.001 \
    --n-lambda 30 --lambda-grid auto \
    --val-methods bootstrap632+ repeated-cv loocv simple-bootstrap \
    --cv-splits 4 --cv-repeats 20 \
    --eval-consensus-only \
    --stabl-penalty l1 --stabl-l1-ratio 0.5 \
    --output-dir results/publish_github
```
Artifacts: AUROC distribution JSON, percentile CI, calibration and ROC plots, per-fold logs.

### Optional supportive analyses
- `.632+ bootstrap`: `--validation-bootstrap 200 --bootstrap-method 632plus`.
- Temporal holdout: add `--temporal-holdout --scanner-metadata-path data/StudyDates.csv --scanner-id-col scanner_patient_name --date-col StudyDate`.

## 4. Post-selection Tuning (optional)
```bash
python code/scripts/optimize_lr_postselection.py \
  --results-dir results_reference/publish_eval \
  --penalty l1 --cs 0.1 0.25 0.5 1 2.5 --cv-folds 5
```
Records tuned coefficients for sensitivity checks.

## 5. Packaging
- Copy `results_reference/publish_discovery` and `results_reference/publish_eval` into the export archive.
- Include command logs (`command.txt`) and environment snapshot from `docs/setup_env.md` step 4.

## 6. Inference (single new external patient)

1) Run the radiomics pipeline to produce a features CSV for the new patient (HF3 feature schema):
- See `radiomics pipeline/README.md` (recommended: `extract_radiomics_yaml.py` with `code/configs/radiomics_config_2mm.yaml`).

2) Predict POPF risk using the exported model bundle:

```bash
cd "primary analysis"
FEATURES_CSV="/path/to/radiomics_features.csv"

python code/predict_popf_risk.py \
  --model-pkl configs/exported_model.pkl \
  --features-csv "$FEATURES_CSV" \
  --id-col patient_id \
  --patient-id "<PATIENT_ID>" \
  --output-csv results/predictions_test.csv
```

### Apply post-hoc calibration (final reportable probability)

Calibration for the frozen signature is produced by:
`code/models/comparative_risk_stratification_v2.py`.

```bash
cd "primary analysis"
CAL_DIR="results/comparative_risk_stratification_v2"

python code/models/comparative_risk_stratification_v2.py \
  --output-dir "$CAL_DIR" \
  --calibration-method auto

python code/predict_popf_risk.py \
  --model-pkl configs/exported_model.pkl \
  --features-csv "$FEATURES_CSV" \
  --id-col patient_id \
  --patient-id "<PATIENT_ID>" \
  --calibration-json "$CAL_DIR/radiomics_calibration.json" \
  --output-csv results/predictions_test_calibrated.csv
```
