# DP-Radiomics-POPF

End-to-end radiomics workflow for predicting clinically relevant postoperative pancreatic fistula (CR-POPF) after distal pancreatectomy from preoperative CT imaging of the pancreatic remnant/head-isthmus region.

This repository is organized as three components that can be used independently or together:
- `radiomics pipeline/`: segmentation viewer + pancreatic head delimitation + radiomics extraction + (optional) harmonization utilities.
- `primary analysis/`: feature selection, model development, calibration, evaluation, and manuscript-ready figures.
- `deployment/`: a lightweight local web app (Docker) that runs the workflow for new cases.

## What’s included / what’s not
- Included: code, configs, and reference figures for the manuscript.
- Not included: any patient imaging, segmentations, clinical tables, patient-level predictions, or PHI/PII. You must provide inputs locally.

## Current manuscript analysis
The current manuscript uses a locked seven-feature radiomics signature (`7-rad`) and evaluates:
- published DP-FRS and DISPAIR-FRS clinical scores exactly as published, without local coefficient refitting;
- unweighted fixed-L2 logistic out-of-fold radiomics models for the main probability audit;
- radiomics-plus-score comparisons against the 7-rad signature, especially 7-rad plus preoperative DP-FRS;
- DeLong testing, calibration diagnostics, and descriptive risk strata from paired out-of-fold predictions.

The public repository intentionally does not ship the enriched clinical database. Runtime outputs under `primary analysis/results/` are ignored by git; only aggregate/reference manuscript figures are kept under `primary analysis/results_reference/`.

For Docker/local inference, `primary analysis/configs/exported_model.pkl` is a final all-cohort refit of the locked 7-rad panel with unweighted fixed-L2 logistic regression. That deployment refit is not the manuscript performance estimate; manuscript discrimination and calibration are reported from out-of-fold predictions.

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
- `primary analysis/code/models/`: calibration and risk stratification utilities.
- `primary analysis/code/models/nested_unweighted_calibration.py`: primary clean 7-rad probability/calibration audit.
- `primary analysis/code/models/create_v2_style_figures_from_clean_analysis.py`: regenerates the manuscript-style figure family from the clean unweighted out-of-fold probabilities.
- `primary analysis/code/models/comparative_risk_stratification_v2.py`: updated comparative score/risk-stratification utility with corrected DP-FRS/DISPAIR formulas and unweighted radiomics/refit models.
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

**Figure 6. Out-of-fold performance, calibration, and risk stratification of the 7-rad signature with and without preoperative DP-FRS**

![Figure 6](primary%20analysis/results_reference/manuscript_figures/figure6_7rad_vs_dfrs_preop_oof.svg)

**Supplemental direct comparison. 7-rad signature vs 7-rad plus preoperative DP-FRS**

![Direct comparison](primary%20analysis/results_reference/manuscript_figures/figure7_7rad_vs_dfrs_preop_direct_comparison.svg)

## Third-party sources and attribution
This project depends on several open-source tools and includes a vendored copy of `BeautifulFigures` for plotting style.

See `THIRD_PARTY.md` for:
- upstream repository links,
- licensing pointers (where applicable),
- and how each dependency is used in this work.
