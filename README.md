# DP-Radiomics-POPF

End-to-end radiomics workflow for predicting postoperative pancreatic fistula (POPF) from CT imaging of the pancreatic head.

This repository is organized as three components that can be used independently or together:
- `radiomics pipeline/`: segmentation viewer + pancreatic head delimitation + radiomics extraction + (optional) harmonization utilities.
- `primary analysis/`: feature selection, model development, calibration, evaluation, and manuscript-ready figures.
- `deployment/`: a lightweight local web app (Docker) that runs the workflow for new cases.

## What’s included / what’s not
- Included: code, configs, and reference figures for the manuscript.
- Not included: any patient imaging, segmentations, clinical tables, or PHI/PII. You must provide your own inputs locally.

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
- `primary analysis/code/models/`: calibration and risk stratification utilities (see `primary analysis/code/models/comparative_risk_stratification_v2.py`).
- `primary analysis/code/utils/`: shared helpers (e.g., figure styling via `primary analysis/code/utils/plotting_utils.py`).
- `primary analysis/results_reference/`: manuscript-ready reference figures exported for the paper.

**Deployment**
- `deployment/`: Dockerized local app wiring upload → preprocessing → segmentation → prediction.

## Manuscript figures
These SVGs are stored in `primary analysis/results_reference/manuscript_figures/` and render directly on GitHub:

![AUC](primary%20analysis/results_reference/manuscript_figures/AUC.svg)
![Radiomics vs Clinical](primary%20analysis/results_reference/manuscript_figures/radiomics_clinical_performance.svg)
![Radiomics+Clinical Refit](primary%20analysis/results_reference/manuscript_figures/radiomics_clinical_refit_performance.svg)
![Head Volume Distribution](primary%20analysis/results_reference/manuscript_figures/head_volume_distribution.svg)
![Figure 7](primary%20analysis/results_reference/manuscript_figures/FIGURE_7.svg)

## Third-party sources and attribution
This project depends on several open-source tools and includes a vendored copy of `BeautifulFigures` for plotting style.

See `THIRD_PARTY.md` for:
- upstream repository links,
- licensing pointers (where applicable),
- and how each dependency is used in this work.
