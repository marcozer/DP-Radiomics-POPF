# Reference Results

This directory contains aggregate manuscript-ready figures only. It must not contain patient-level databases, imaging files, segmentations, or OOF prediction tables.

## Manuscript Figure Assets

The files in `manuscript_figures/` were exported from the current manuscript figure DOCX:

- `figure1_study_design_radiomics_workflow.svg` / `.png`
- `figure2_radiomics_feature_selection.svg` / `.png`
- `figure3_model_development_internal_validation.svg` / `.png`
- `figure4_signature_values_predicted_risk.svg` / `.png`
- `figure5_published_clinical_score_benchmarks.svg` / `.png`
- `figure6_7rad_vs_dfrs_preop_oof.svg` / `.png`

Additional model-comparison references:

- `figure7_7rad_vs_dfrs_preop_direct_comparison.svg` / `.png`
- `risk_group_event_rates_7rad_vs_dfrs_preop.svg` / `.png`

## Regeneration

Use `../code/models/nested_unweighted_calibration.py` for the primary clean OOF probability audit and `../code/models/create_v2_style_figures_from_clean_analysis.py` for the v2-style figure family. Outputs should be written under `primary analysis/results/`, which is ignored by git.
