# Reference Results

This directory contains aggregate manuscript-ready figures only. It must not contain patient-level databases, imaging files, segmentations, or OOF prediction tables.

## Manuscript Figure Assets

The files in `manuscript_figures/` were exported from the current manuscript figure DOCX:

- `figure1_study_design_radiomics_workflow.svg` / `.png`
- `figure2_radiomics_feature_selection.svg` / `.png`
- `figure3_model_development_internal_validation.svg` / `.png`
- `figure4_signature_values_predicted_risk.svg` / `.png`
- `figure5_published_clinical_score_benchmarks.svg` / `.png`
- `figure6_elasticnet_7rad_mpd_thickness.svg` / `.png`

## Regeneration

Use `../code/models/r0_v2_elasticnet_7rad_mpd_thickness.py` for the current R0_v2 model comparison and Figure 6. Outputs should be written under `primary analysis/results/`, which is ignored by git.

Use `../code/figures/generate_figure3_model_development_internal_validation.py` to regenerate Figure 3 from local R0_v2 outputs. Only aggregate SVG/PNG figure assets are committed.

`nested_feature_selection_summary.json` contains the aggregate nested STABL feature-selection sensitivity result used to support feature-selection robustness. It does not contain patient-level data.
