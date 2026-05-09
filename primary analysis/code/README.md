# Code Bundle Instructions

Copy the finalized scripts and packages listed in `../docs/code_manifest.md` into this folder, preserving their relative import paths.

Minimum required items:
- `popf_stabl_corrected_parallel_enhanced_v3.py`
- `popf_stabl_ultra_optimized.py`
- `models/nested_unweighted_calibration.py`
- `models/create_v2_style_figures_from_clean_analysis.py`
- `scripts/` helper modules (alignment audit, fixed-panel eval, post-selection tuner)
- `utils/` helpers referenced by the scripts (plotting, data utils)
- `stabl` installed via pip (see `../docs/setup_env.md`)

No clinical database, radiomics matrix, imaging file, or patient-level prediction file is bundled in this public export. Place local non-versioned inputs under `primary analysis/data/` or pass absolute paths at runtime.

Clean manuscript/deployment calibration audit for the locked seven-feature radiomics (`7-rad`) signature:
```bash
python "code/models/nested_unweighted_calibration.py" \
  --radiomics-path "data/HF3.csv" \
  --clinical-path "data/POPF_SCANNER_complete_clinical_db_filled.csv" \
  --output-dir "results/nested_unweighted_calibration"
```

Optional nested Optuna sensitivity, kept separate from the primary fixed L2
estimate and not used as the main manuscript probability estimate:
```bash
python "code/models/nested_unweighted_calibration.py" \
  --radiomics-path "data/HF3.csv" \
  --clinical-path "data/POPF_SCANNER_complete_clinical_db_filled.csv" \
  --output-dir "results/nested_unweighted_calibration_optuna" \
  --include-nested-optuna \
  --optuna-trials 150
```

To regenerate the same figure family as
`comparative_risk_stratification_v2_hard_optuna_500` using the clean fixed-L2
OOF probabilities:
```bash
python "code/models/create_v2_style_figures_from_clean_analysis.py" \
  --radiomics-path "data/HF3.csv" \
  --clinical-path "data/POPF_SCANNER_complete_clinical_db_filled.csv" \
  --output-dir "results/v2_style_figures_fixed_l2"
```

Patient-level prediction CSVs are disabled by default in the public scripts. Add `--write-patient-predictions` only for local, non-public audit runs.

Deployment inference uses `../configs/exported_model.pkl`, a final all-cohort refit of the locked 7-rad feature panel with unweighted fixed-L2 logistic regression. This refit is for prospective inference only; manuscript performance remains based on out-of-fold predictions.

Before exporting, run:
```bash
shasum -a 256 code/**/*.py > docs/code_hashes.txt
```
to capture integrity hashes.
