# Code Manifest Template

| Export Path | Source Path | Notes |
|-------------|-------------|-------|
| `code/popf_stabl_corrected_parallel_enhanced_v3.py` | `popf_stabl_corrected_parallel_enhanced_v3.py` | Fixed-panel pipeline; update `data_dir` default to `data/`. |
| `code/popf_stabl_ultra_optimized.py` | `popf_stabl_ultra_optimized.py` | Discovery STABL script; includes ComBat helpers. |
| `code/scripts/check_alignment.py` | `scripts/check_alignment.py` | Required before any modeling. |
| `code/scripts/eval_fixed_panel.py` | `scripts/eval_fixed_panel.py` | Clean fixed-panel CV without selection. |
| `code/scripts/eval_fixed_panel_combat.py` | `scripts/eval_fixed_panel_combat.py` | Optional harmonization evaluation. |
| `code/scripts/optimize_lr_postselection.py` | `scripts/optimize_lr_postselection.py` | Optional LR C sensitvity. |
| `code/utils/plotting_utils.py` | `utils/plotting_utils.py` | Shared figure helpers. |
| `code/utils/data_utils.py` | `utils/data_utils.py` | (Add once finalized). |
| `stabl` | pip dependency | Install from the upstream repo and document the commit SHA in `docs/setup_env.md` notes. |

Fill in the remaining helper modules (e.g., `utils/preprocessing.py`, `scripts/comparative_risk_stratification.py`) once they are frozen for the export.
