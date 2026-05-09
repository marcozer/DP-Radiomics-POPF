# Code Manifest

| Export Path | Source Path | Notes |
|-------------|-------------|-------|
| `code/popf_stabl_corrected_parallel_enhanced_v3.py` | `popf_stabl_corrected_parallel_enhanced_v3.py` | Fixed-panel pipeline; update `data_dir` default to `data/`. |
| `code/popf_stabl_ultra_optimized.py` | `popf_stabl_ultra_optimized.py` | Discovery STABL script; includes ComBat helpers. |
| `code/scripts/check_alignment.py` | `scripts/check_alignment.py` | Required before any modeling. |
| `code/scripts/eval_fixed_panel.py` | `scripts/eval_fixed_panel.py` | Clean fixed-panel CV without selection. |
| `code/scripts/eval_fixed_panel_combat.py` | `scripts/eval_fixed_panel_combat.py` | Optional harmonization evaluation. |
| `code/scripts/optimize_lr_postselection.py` | `scripts/optimize_lr_postselection.py` | Optional LR C sensitvity. |
| `code/models/comparative_risk_stratification_v2.py` | `R0` / manuscript regeneration utility | Updated clinical-score formulas, unweighted radiomics/refit models, no-refit published score benchmarking. |
| `code/models/nested_unweighted_calibration.py` | `R0/nested_unweighted_20260505/run_nested_unweighted_calibration.py` | Primary clean unweighted fixed-L2 OOF probability audit for the locked 7-rad signature; patient-level predictions off by default. |
| `code/models/create_v2_style_figures_from_clean_analysis.py` | `R0/nested_unweighted_20260505/create_v2_style_figures_from_clean_analysis.py` | Recreates the `comparative_risk_stratification_v2` figure family from clean fixed-L2 OOF predictions; patient-level predictions off by default. |
| `code/utils/plotting_utils.py` | `utils/plotting_utils.py` | Shared figure helpers. |
| `code/utils/data_utils.py` | `utils/data_utils.py` | (Add once finalized). |
| `stabl` | pip dependency | Install from the upstream repo and document the commit SHA in `docs/setup_env.md` notes. |

The public export intentionally excludes all cohort databases, radiomics CSVs, imaging data, and patient-level prediction outputs.
