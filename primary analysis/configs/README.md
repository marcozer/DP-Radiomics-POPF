# Config Templates

Place frozen CLI presets, panel files, and metadata mappings here.

Recommended contents:
- `panels/publish_lr_panel.txt`: feature names selected via STABL discovery.
- `presets/stabl_v3_publish.yaml`: CLI arguments for `popf_stabl_corrected_parallel_enhanced_v3.py`.
- `presets/stabl_ultra_discovery.yaml`: CLI arguments for `popf_stabl_ultra_optimized.py`.
- `metadata/clinical_features.json`: documentation of clinical covariates used in comparative models.
- `exported_model.pkl`: deployable R0_v2 `7-rad` elastic-net model bundle with no patient-level data.
- `calibration/radiomics_calibration.json`: identity calibration metadata for the R0_v2 model bundle.
- `calibration/radiomics_risk_stratification.json`: internal descriptive OOF tertile thresholds; external validation required before clinical use.

Keep filenames descriptive; include the exact command string used to generate each panel/config at the top of the file as comments.
