# Reference Results

This directory contains aggregate manuscript-ready figures and non-patient-level validation summaries. It must not contain patient-level databases, imaging files, segmentations, or OOF prediction tables.

## Manuscript Figure Assets

The current editable SVG files in `manuscript_figures/` are:

- `figure1_study_flow_and_radiomics_pipeline.svg`
- `figure2_segmentation_and_head_isthmus_volume.svg`
- `figure3_radiomics_feature_selection_lasso.svg`
- `figure4_locked_panel_model_screening.svg`
- `figure5_apparent_radiomics_heatmap.svg`
- `figure6_apparent_roc_and_fitted_model_calibration.svg`

The `manuscript_figures/supplementary/` directory contains the three current
supplementary SVG figures.

## Regeneration

Use `../code/models/r0_v3_apparent_model_comparison.py` for the apparent model comparison and Figure 6.

Use `../code/models/locked_panel_candidate_632plus.py` to reproduce the 2,000-resample paired bootstrap `.632+` model-family estimates.

Use `../code/models/bootstrap_oob_cutpoints.py` to reproduce the 2,000-resample out-of-bag cutpoint validation.

Runtime outputs should be written under `primary analysis/results/`, which is ignored by git. The corresponding directories here contain only aggregate reference outputs and editable SVG assets.

## Aggregate Reference Outputs

- `r0_v3_apparent_model_comparison/`: apparent AUCs, bootstrap confidence intervals, calibration points, DeLong tests, tuning parameters, and elastic-net coefficients.
- `locked_panel_candidate_632plus/`: paired 2,000-resample `.632+` model-family screening.
- `bootstrap_oob_cutpoints/`: paired 2,000-resample `.632+` AUC and out-of-bag validation of constrained-MCC operating points.

`nested_feature_selection_summary.json` contains the aggregate nested STABL feature-selection sensitivity result used to support feature-selection robustness. It does not contain patient-level data.
