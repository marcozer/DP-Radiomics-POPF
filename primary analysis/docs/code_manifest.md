# Code Manifest

| Export Path | Source Path | Notes |
|-------------|-------------|-------|
| `code/popf_stabl_corrected_parallel_enhanced_v3.py` | `popf_stabl_corrected_parallel_enhanced_v3.py` | Fixed-panel pipeline; update `data_dir` default to `data/`. |
| `code/popf_stabl_ultra_optimized.py` | `popf_stabl_ultra_optimized.py` | Discovery STABL script; includes ComBat helpers. |
| `code/scripts/check_alignment.py` | `scripts/check_alignment.py` | Required before any modeling. |
| `code/scripts/eval_fixed_panel.py` | `scripts/eval_fixed_panel.py` | Clean fixed-panel CV without selection. |
| `code/scripts/eval_fixed_panel_combat.py` | `scripts/eval_fixed_panel_combat.py` | Optional harmonization evaluation. |
| `code/scripts/optimize_lr_postselection.py` | `scripts/optimize_lr_postselection.py` | Optional LR C sensitvity. |
| `code/models/r0_v2_elasticnet_7rad_mpd_thickness.py` | `R0_v2/analysis_elasticnet_7rad_mpd_thickness_20260616` | Current manuscript analysis: standardized unweighted elastic-net `7-rad`, refitted `7-rad + MPD/thickness`, standalone DP-FRS/DISPAIR benchmarks, bootstrap `.632+`, paired OOF comparison, and deployable model export. |
| `code/figures/generate_figure3_model_development_internal_validation.py` | `R0_v2` / local aggregate analysis outputs | Regenerates manuscript Figure 3 with the validation-method comparison retained and the ROC panel shown as the apparent final-fit curve annotated with bootstrap `.632+` AUC. |
| `code/source_snapshots/popf_stabl_corrected_parallel_enhanced_v3_publish_github_original.py` | `sept25/popf_stabl_corrected_parallel_enhanced_v3.py` | STABL model-development script for the 7-rad EN run used in Figure 3. |
| `code/source_snapshots/popf_stabl_nested_experiments_original.py` | `sept25/popf_stabl_nested_experiments.py` | Nested STABL feature-selection script. |
| `code/models/comparative_risk_stratification_v2.py` | `R0` / manuscript regeneration utility | Comparative score/risk-stratification utility retained for earlier figure generations. |
| `code/models/nested_unweighted_calibration.py` | `R0/nested_unweighted_20260505/run_nested_unweighted_calibration.py` | Clean unweighted fixed-L2 OOF probability utility; no longer the current R0_v2 primary estimator. |
| `code/models/create_v2_style_figures_from_clean_analysis.py` | `R0/nested_unweighted_20260505/create_v2_style_figures_from_clean_analysis.py` | V2-style figure utility from clean fixed-L2 OOF predictions. |
| `code/utils/plotting_utils.py` | `utils/plotting_utils.py` | Shared figure helpers. |
| `code/utils/data_utils.py` | `utils/data_utils.py` | (Add once finalized). |
| `stabl` | pip dependency | Install from the upstream repo and document the commit SHA in `docs/setup_env.md` notes. |

The public export includes only de-identified model-ready CSVs. It excludes source clinical databases, imaging data, segmentations, direct identifiers, and patient-level prediction outputs.
