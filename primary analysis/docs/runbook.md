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

## 3. R0_v2 Elastic-Net Fixed-Panel Evaluation (current manuscript analysis)
```bash
python "code/models/r0_v2_elasticnet_7rad_mpd_thickness.py" \
  --radiomics-path data/HF3.csv \
  --clinical-path data/final_clinical_db.csv \
  --output-dir results/r0_v2_elasticnet_7rad_mpd_thickness \
  --export-model-pkl configs/exported_model.pkl
```
Artifacts: bootstrap `.632+` metrics, repeated OOF metrics, paired AUC comparison, standalone DP-FRS/DISPAIR benchmarks, manuscript Figure 6, and a deployable non-patient-level model bundle.

The current manuscript does not use temporal hold-out or fixed-L2 as primary estimators. Those older scripts remain in `code/main analysis/` and `code/models/` only for audit/sensitivity work.

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

### Calibration and risk grouping

The R0_v2 deployment bundle uses an identity calibration JSON by default. Risk groups, if requested, use internal descriptive OOF tertile cut points and require external validation before clinical use.

```bash
cd "primary analysis"

python code/predict_popf_risk.py \
  --model-pkl configs/exported_model.pkl \
  --features-csv "$FEATURES_CSV" \
  --id-col patient_id \
  --patient-id "<PATIENT_ID>" \
  --calibration-json configs/calibration/radiomics_calibration.json \
  --output-csv results/predictions_test_calibrated.csv
```
