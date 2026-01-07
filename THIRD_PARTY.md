# Third-party Sources and Attribution

This repository uses and/or depends on the following third-party projects. Please cite and comply with their licenses when using this work.

## Bundled in this repository

### BeautifulFigures
- Purpose here: figure styling inspiration and plotting conventions used by `primary analysis/code/utils/plotting_utils.py`.
- Location (vendored copy): `primary analysis/dependencies/BeautifulFigures/`
- License: `primary analysis/dependencies/BeautifulFigures/LICENSE`
- Author/project: BeautifulFigures by Dr. Andrey Churkin.
  - Background + examples: `primary analysis/dependencies/BeautifulFigures/README.md`
  - Author website: https://andreychurkin.ru/
  - YouTube channel: https://www.youtube.com/@chuscience

## External dependencies (not bundled)

### STABL
- Purpose here: stability selection / feature selection used by the primary analysis scripts and required to unpickle the exported preprocessing pipeline.
- Upstream: https://github.com/gregbellan/Stabl
- Install (example): `pip install git+https://github.com/gregbellan/Stabl.git@<commit>`

### PyRadiomics
- Purpose here: radiomics feature extraction (IBSI-oriented features).
- Upstream: https://github.com/AIM-Harvard/pyradiomics

### TotalSegmentator
- Purpose here: automated pancreas segmentation (deployment and preprocessing workflows).
- Upstream: https://github.com/wasserth/TotalSegmentator

### dcm2niix
- Purpose here: DICOM → NIfTI conversion in deployment workflows.
- Upstream: https://github.com/rordenlab/dcm2niix

### neurocombat-sklearn / ComBat
- Purpose here: optional ComBat harmonization utilities for radiomics features.
- Upstream: https://github.com/Warvito/neurocombat_sklearn

### Python scientific stack and framework libraries
- `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`
- `nibabel`, `SimpleITK`
- `Flask`, `flask-cors`

## Notes
- This repository does not ship any patient imaging, segmentations, or clinical outcome tables.
- If you add datasets locally, keep them outside version control and review third-party licenses before redistribution.
