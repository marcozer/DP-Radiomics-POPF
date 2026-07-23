# De-identified analysis data

This folder contains the de-identified, model-ready CSV files used by the current manuscript analyses.

Files:
- `radiomics_features_anonymized.csv`: one row per analyzed patient, with `patient_id`, CR-POPF outcome, and 313 radiomics features.
- `model_covariates_anonymized.csv`: one row per analyzed patient, with `patient_id`, CR-POPF outcome, and covariates required for MPD/thickness and 2025 DISPAIR comparisons.

De-identification rules:
- `patient_id` is a non-reversible pseudonym created only for this public export.
- The name-to-`patient_id` crosswalk is not stored in this repository.
- No names, source-system identifiers, IPP/person identifiers, dates, clinical notes, imaging files, segmentations, or hospital source records are included.
- The dataset contains 195 analyzed patients and 36 CR-POPF events.

These files are sufficient to rerun the public apparent-model, locked-panel `.632+`, cutpoint, and nested STABL commands without private clinical databases.
