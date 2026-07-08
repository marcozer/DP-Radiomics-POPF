# Publication Export Blueprint

This directory centralizes everything needed to ship the POPF STABL pipeline externally without manuscripts or exploratory notebooks. It documents the code entry points, resource expectations, dependency capture, and run commands so that we can assemble a self-contained archive for reviewers or collaborators.

## Contents
- `code/`: canonical scripts to include in the export plus a manifest mapping back to the working tree.
- `configs/`: JSON/YAML templates for CLI arguments, feature lists, or run metadata.
- `data_anonymized/`: de-identified model-ready radiomics features and clinical covariates used to reproduce the primary analysis.
- `results_reference/`: light-weight dev outputs (logs, plots, JSON summaries) created with reproducible seeds to illustrate folder structure.
- `docs/`: setup instructions, dependency lists, runbooks, and packaging checklists.

Populating these folders completes the export handoff: we copy the vetted scripts under `code/`, include the de-identified model-ready CSVs, drop the frozen STABL panel + figures into `results_reference/`, and provide the `docs/` artifacts to describe how to recreate full runs on a clean machine.

Notes:
- The modeling scripts expect `stabl` to be installed via pip (e.g., `pip install git+https://github.com/gregbellan/Stabl.git@<commit>`); no vendored copy is used.
- Run commands from the repository root, referencing scripts under `code/...` (see `docs/runbook.md`).
- Radiomics extraction/harmonization lives in the sibling `radiomics pipeline` repo; consume its exported CSVs via CLI arguments rather than cross-repo paths.
- For current R0_v2 manuscript analysis, use `code/models/r0_v2_elasticnet_7rad_mpd_thickness.py`; it runs on the bundled de-identified files by default.
- For single-patient inference with the frozen radiomics signature, use `code/predict_popf_risk.py` with `configs/exported_model.pkl` (see `docs/runbook.md` section 6). The default calibration JSON is identity for the R0_v2 elastic-net bundle.
