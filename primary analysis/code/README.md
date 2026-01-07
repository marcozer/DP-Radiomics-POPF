# Code Bundle Instructions

Copy the finalized scripts and packages listed in `../docs/code_manifest.md` into this folder, preserving their relative import paths.

Minimum required items:
- `popf_stabl_corrected_parallel_enhanced_v3.py`
- `popf_stabl_ultra_optimized.py`
- `scripts/` helper modules (alignment audit, fixed-panel eval, post-selection tuner)
- `utils/` helpers referenced by the scripts (plotting, data utils)
- `stabl` installed via pip (see `../docs/setup_env.md`)

Before exporting, run:
```bash
shasum -a 256 code/**/*.py > docs/code_hashes.txt
```
to capture integrity hashes.
