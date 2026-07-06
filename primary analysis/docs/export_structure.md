# Export Structure Plan

## Code Bundle
- `code/popf_stabl_corrected_parallel_enhanced_v3.py`: fixed-panel pipeline with preprocessing, alignment, and evaluation. Copy verbatim from repo root. Update `data_dir` defaults to `data/` relative path.
- `code/popf_stabl_ultra_optimized.py`: discovery-oriented STABL selection script. Include README comments warning that AUCs are exploratory only.
- `code/scripts/`: include `scripts/eval_fixed_panel.py`, `scripts/eval_fixed_panel_combat.py`, `scripts/optimize_lr_postselection.py`, `scripts/check_alignment.py`, and any helper modules under `utils/` (`plotting_utils.py`, `data.py`, etc.). Preserve package-relative imports by keeping directory names identical.
- Install `stabl` via pip (see `docs/setup_env.md`). Record the commit SHA in your environment capture notes for reproducibility.

## Configurations
- Provide CLI presets in YAML (e.g., `configs/stabl_v3_publish.yaml`) capturing discovery + evaluation flags.
- Store panel files (`configs/panels/*.txt`) and clinical feature lists referenced by comparative scripts.

## Data Templates
- `data_templates/radiomics_schema.md`: column description, ID alignment rules, mandatory numeric formats.
- `data_templates/clinical_schema.md`: definitions for clinical covariates (MPD mm, BMI kg/m², etc.).
- Synthetic CSVs (no PHI) that illustrate header order and accepted values. Use randomly generated values and note they are illustrative only.

## Results Reference
- Re-run discovery + fixed-panel CV with reduced iterations (e.g., `--ensemble-runs 2`, `--n-bootstraps 50`) and store the output folder under `results_reference/dev_run/`.
- Capture JSON summaries, panel text files, AUROC distribution plots, and calibration figures to demonstrate output naming.
- Include log excerpts showing runtime, command used, and environment hash.

## Documentation
- `docs/setup_env.md`: environment + dependency capture instructions.
- `docs/runbook.md`: sequence of commands (alignment check → discovery → fixed-panel evaluation → optional bootstrap) plus expected runtime.
- `docs/results_packaging_checklist.md`: verification steps before shipping (hash scripts, confirm panel file, ensure no PHI).
