# DP-Radiomics-POPF

End-to-end radiomics workflow for predicting clinically relevant postoperative pancreatic fistula (CR-POPF) after distal pancreatectomy from preoperative CT imaging of the pancreatic remnant/head-isthmus region.

This repository is organized as three components that can be used independently or together:
- `radiomics pipeline/`: segmentation viewer + pancreatic head delimitation + radiomics extraction + (optional) harmonization utilities.
- `primary analysis/`: feature selection, model development, calibration, evaluation, and manuscript-ready figures.
- `deployment/`: a lightweight local web app (Docker) that runs the workflow for new cases.

## What’s included / what’s not
- Included: code, configs, manuscript reference figures, and de-identified model-ready CSV files for reproducing the primary analysis.
- Not included: patient imaging, segmentations, operative notes, hospital records, dates, direct identifiers, source-system identifiers, or any PHI/PII.

## Current manuscript analysis
The current manuscript uses a locked seven-feature radiomics signature (`7-rad`) and evaluates:
- the final `7-rad` model as a standardized, unweighted elastic-net logistic regression refitted on the full 195-patient cohort;
- paired class-stratified bootstrap `.632+` AUC with 2,000 resamples as the primary optimism-corrected estimate;
- strict nested feature-selection sensitivity in which STABL is repeated within each outer training fold;
- apparent full-cohort comparisons with refitted `7-rad + MPD/thickness` and `7-rad + DISPAIR-FRS 2025 covariates` elastic-net models;
- published preoperative DP-FRS and 2025 DISPAIR-FRS equations as standalone, non-refitted benchmarks;
- constrained-MCC rule-out and rule-in cutpoints validated out of bag over 2,000 class-stratified bootstrap resamples.

This repository includes only de-identified model-ready covariates and radiomics features under `primary analysis/data_anonymized/`. It does not include clinical source databases, imaging files, segmentations, direct identifiers, or patient-level prediction outputs. Runtime outputs under `primary analysis/results/` are ignored by git; only aggregate/reference manuscript figures are kept under `primary analysis/results_reference/`.

For Docker/local inference, `primary analysis/configs/exported_model.pkl` is a final all-cohort refit of the locked 7-rad panel with standardized unweighted elastic-net logistic regression. Apparent deployment probabilities are not the primary manuscript performance estimate; the manuscript reports bootstrap `.632+` AUC and strict nested STABL separately.

## Quickstart (Docker deployment)
From the repo root:

```bash
docker compose -f deployment/docker-compose.yml up --build
```

Then open:
- API/UI: `http://localhost:8000`
- (If enabled) viewer: `http://localhost:5003`

Data you upload and all runtime outputs are written to Docker volumes / local bind mounts configured in `deployment/docker-compose.yml` and are ignored by git.

## Local runs (non-Docker)
See:
- `deployment/README.md`
- `radiomics pipeline/README.md`
- `primary analysis/docs/runbook.md`

## Code map (where things live)
**Radiomics pipeline**
- `radiomics pipeline/pancreas_head_delimiter/`: lightweight viewer used to select the pancreatic head delimiter (interactive coordinate selection).
- `radiomics pipeline/code/`: scripts for head extraction, CT cropping, radiomics extraction, and ComBat utilities.

**Statistical analysis**
- `primary analysis/code/main analysis/`: core modeling and statistical evaluation scripts (feature selection, cross-validation, ablations, sensitivity analyses).
- `primary analysis/code/models/`: model comparison, calibration, and risk-stratification utilities.
- `primary analysis/code/models/r0_v3_apparent_model_comparison.py`: reproduces the apparent radiomics/radioclinical comparison, published-score benchmarks, calibration, coefficients, and paired DeLong tests.
- `primary analysis/code/models/locked_panel_candidate_632plus.py`: reproduces the six-estimator, 2,000-resample paired bootstrap `.632+` screening.
- `primary analysis/code/models/bootstrap_oob_cutpoints.py`: reproduces the 2,000-resample out-of-bag validation of constrained-MCC cutpoints.
- `primary analysis/code/models/r0_v2_elasticnet_7rad_mpd_thickness.py`: retained earlier elastic-net analysis and deployable model export.
- `primary analysis/code/utils/`: shared helpers (e.g., figure styling via `primary analysis/code/utils/plotting_utils.py`).
- `primary analysis/results_reference/`: manuscript-ready reference figures exported for the paper.

**Deployment**
- `deployment/`: Dockerized local app wiring upload → preprocessing → segmentation → prediction.

## Manuscript figures
These are the current SVG assets embedded in the manuscript figure DOCX. They are stored in `primary analysis/results_reference/manuscript_figures/` and render directly on GitHub.

**Figure 1. Study flow and radiomics pipeline**

![Figure 1](primary%20analysis/results_reference/manuscript_figures/figure1_study_flow_and_radiomics_pipeline.svg)

**Figure 2. Head-isthmus segmentation and volume distribution**

![Figure 2](primary%20analysis/results_reference/manuscript_figures/figure2_segmentation_and_head_isthmus_volume.svg)

**Figure 3. Radiomics feature reduction and LASSO coefficient paths**

![Figure 3](primary%20analysis/results_reference/manuscript_figures/figure3_radiomics_feature_selection_lasso.svg)

**Figure 4. Locked-panel candidate-estimator screening and internal validation**

![Figure 4](primary%20analysis/results_reference/manuscript_figures/figure4_locked_panel_model_screening.svg)

**Figure 5. Apparent seven-feature radiomics signature across the cohort**

![Figure 5](primary%20analysis/results_reference/manuscript_figures/figure5_apparent_radiomics_heatmap.svg)

**Figure 6. Apparent discrimination and calibration of radiomics and radioclinical models**

![Figure 6](primary%20analysis/results_reference/manuscript_figures/figure6_apparent_roc_and_fitted_model_calibration.svg)

### Supplementary figures

- [Supplementary Figure 1. STABL consensus](primary%20analysis/results_reference/manuscript_figures/supplementary/supplementary_figure1_stabl_consensus.svg)
- [Supplementary Figure 2. Bootstrap and nested validation](primary%20analysis/results_reference/manuscript_figures/supplementary/supplementary_figure2_bootstrap_and_nested_validation.svg)
- [Supplementary Figure 3. Complete 195-patient heatmap](primary%20analysis/results_reference/manuscript_figures/supplementary/supplementary_figure3_complete_195_patient_heatmap.svg)

## Third-party sources and attribution
This project depends on several open-source tools and includes a vendored copy of `BeautifulFigures` for plotting style.

See `THIRD_PARTY.md` for:
- upstream repository links,
- licensing pointers (where applicable),
- and how each dependency is used in this work.
