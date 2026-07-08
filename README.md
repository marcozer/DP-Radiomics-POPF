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
- bootstrap `.632+` AUC as the primary optimism-corrected performance estimate for the final model;
- nested feature-selection sensitivity in which STABL is repeated within training folds to assess selection robustness;
- repeated out-of-fold validation of the locked feature panel for paired model comparison;
- a refitted `7-rad + MPD/thickness` elastic-net model as the comparative radioclinical sensitivity analysis;
- published DP-FRS and DISPAIR-FRS clinical scores exactly as published, as standalone benchmarks only.

This repository includes only de-identified model-ready covariates and radiomics features under `primary analysis/data_anonymized/`. It does not include clinical source databases, imaging files, segmentations, direct identifiers, or patient-level prediction outputs. Runtime outputs under `primary analysis/results/` are ignored by git; only aggregate/reference manuscript figures are kept under `primary analysis/results_reference/`.

For Docker/local inference, `primary analysis/configs/exported_model.pkl` is a final all-cohort refit of the locked 7-rad panel with standardized unweighted elastic-net logistic regression. Apparent deployment probabilities are not the manuscript performance estimate; the manuscript reports bootstrap `.632+` and repeated out-of-fold validation.

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
- `primary analysis/code/models/r0_v2_elasticnet_7rad_mpd_thickness.py`: current R0_v2 manuscript analysis, elastic-net `7-rad` with and without MPD/thickness, standalone DP-FRS/DISPAIR benchmarks, and deployable model export.
- `primary analysis/code/figures/generate_figure3_model_development_internal_validation.py`: regenerates manuscript Figure 3 with editable SVG text; uses local R0_v2 aggregate outputs and does not require patient-level data in the public repository.
- `primary analysis/code/utils/`: shared helpers (e.g., figure styling via `primary analysis/code/utils/plotting_utils.py`).
- `primary analysis/results_reference/`: manuscript-ready reference figures exported for the paper.

**Deployment**
- `deployment/`: Dockerized local app wiring upload → preprocessing → segmentation → prediction.

## Manuscript figures
These are the current SVG/PNG assets embedded in the manuscript figure DOCX, plus one direct-comparison model figure. They are stored in `primary analysis/results_reference/manuscript_figures/` and render directly on GitHub.

**Figure 1. Study design and radiomics workflow**

![Figure 1](primary%20analysis/results_reference/manuscript_figures/figure1_study_design_radiomics_workflow.svg)

**Figure 2. Radiomics feature selection**

![Figure 2](primary%20analysis/results_reference/manuscript_figures/figure2_radiomics_feature_selection.svg)

**Figure 3. Development and internal validation of the seven-feature radiomics model**

![Figure 3](primary%20analysis/results_reference/manuscript_figures/figure3_model_development_internal_validation.svg)

**Figure 4. Radiomics signature values and predicted CR-POPF risk**

![Figure 4](primary%20analysis/results_reference/manuscript_figures/figure4_signature_values_predicted_risk.svg)

**Figure 5. Published clinical reference scores applied without refitting**

![Figure 5](primary%20analysis/results_reference/manuscript_figures/figure5_published_clinical_score_benchmarks.svg)

**Figure 6. Elastic-net performance and calibration of the 7-rad signature with and without refit addition of MPD diameter and pancreatic neck thickness**

![Figure 6](primary%20analysis/results_reference/manuscript_figures/figure6_elasticnet_7rad_mpd_thickness.svg)

## Third-party sources and attribution
This project depends on several open-source tools and includes a vendored copy of `BeautifulFigures` for plotting style.

See `THIRD_PARTY.md` for:
- upstream repository links,
- licensing pointers (where applicable),
- and how each dependency is used in this work.
