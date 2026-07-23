# Code Bundle Instructions

Copy the finalized scripts and packages listed in `../docs/code_manifest.md` into this folder, preserving their relative import paths.

Minimum required items:
- `popf_stabl_corrected_parallel_enhanced_v3.py`
- `popf_stabl_ultra_optimized.py`
- `models/r0_v3_apparent_model_comparison.py`
- `models/r0_v2_elasticnet_7rad_mpd_thickness.py`
- `models/locked_panel_candidate_632plus.py`
- `models/bootstrap_oob_cutpoints.py`
- `scripts/` helper modules (alignment checks, fixed-panel eval, post-selection tuner)
- `utils/` helpers referenced by the scripts (plotting, data utils)
- `stabl` installed via pip (see `../docs/setup_env.md`)

The public export bundles de-identified model-ready radiomics features and covariates under `../data_anonymized/`. Imaging files, segmentations, source clinical databases, notes, direct identifiers, and patient-level prediction outputs are not bundled. To run against a private local source table, pass absolute paths at runtime.

Current apparent manuscript comparison:
```bash
python "code/models/r0_v3_apparent_model_comparison.py" \
  --export-model "configs/exported_model.pkl"
```

Runtime outputs are written under `results/`, which is ignored by git. Committed reference outputs are aggregate and contain no patient-level predictions.

Deployment inference uses `../configs/exported_model.pkl`, a final all-cohort refit of the locked 7-rad feature panel with standardized unweighted elastic-net logistic regression. This refit is for prospective inference only; the primary manuscript performance estimate is bootstrap `.632+`, with strict nested STABL reported separately for feature-selection sensitivity.

Before exporting, run:
```bash
shasum -a 256 code/**/*.py > docs/code_hashes.txt
```
to capture integrity hashes.
