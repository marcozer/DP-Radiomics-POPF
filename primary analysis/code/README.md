# Code Bundle Instructions

Copy the finalized scripts and packages listed in `../docs/code_manifest.md` into this folder, preserving their relative import paths.

Minimum required items:
- `popf_stabl_corrected_parallel_enhanced_v3.py`
- `popf_stabl_ultra_optimized.py`
- `models/r0_v2_elasticnet_7rad_mpd_thickness.py`
- `figures/generate_figure3_model_development_internal_validation.py`
- `scripts/` helper modules (alignment checks, fixed-panel eval, post-selection tuner)
- `utils/` helpers referenced by the scripts (plotting, data utils)
- `stabl` installed via pip (see `../docs/setup_env.md`)

The public export bundles de-identified model-ready radiomics features and covariates under `../data_anonymized/`. Imaging files, segmentations, source clinical databases, notes, direct identifiers, and patient-level prediction outputs are not bundled. To run against a private local source table, pass absolute paths at runtime.

Current R0_v2 manuscript/deployment analysis for the locked seven-feature radiomics (`7-rad`) signature:
```bash
python "code/models/r0_v2_elasticnet_7rad_mpd_thickness.py" \
  --radiomics-path "data_anonymized/radiomics_features_anonymized.csv" \
  --clinical-path "data_anonymized/model_covariates_anonymized.csv" \
  --output-dir "results/r0_v2_elasticnet_7rad_mpd_thickness" \
  --export-model-pkl "configs/exported_model.pkl"
```

The R0_v2 analysis writes pseudonymous row-index predictions under `results/` for local auditability. The `results/` folder is ignored by git and should not be committed.

Deployment inference uses `../configs/exported_model.pkl`, a final all-cohort refit of the locked 7-rad feature panel with standardized unweighted elastic-net logistic regression. This refit is for prospective inference only; manuscript performance is reported with bootstrap `.632+` and repeated out-of-fold validation.

Before exporting, run:
```bash
shasum -a 256 code/**/*.py > docs/code_hashes.txt
```
to capture integrity hashes.
