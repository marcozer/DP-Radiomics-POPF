# Code Manifest

| Export Path | Source Path | Notes |
|-------------|-------------|-------|
| `code/popf_stabl_corrected_parallel_enhanced_v3.py` | `popf_stabl_corrected_parallel_enhanced_v3.py` | Fixed-panel pipeline; update `data_dir` default to `data/`. |
| `code/popf_stabl_ultra_optimized.py` | `popf_stabl_ultra_optimized.py` | Discovery STABL script; includes ComBat helpers. |
| `code/scripts/check_alignment.py` | `scripts/check_alignment.py` | Required before any modeling. |
| `code/scripts/eval_fixed_panel.py` | `scripts/eval_fixed_panel.py` | Clean fixed-panel CV without selection. |
| `code/scripts/eval_fixed_panel_combat.py` | `scripts/eval_fixed_panel_combat.py` | Optional harmonization evaluation. |
| `code/scripts/optimize_lr_postselection.py` | `scripts/optimize_lr_postselection.py` | Optional LR C sensitvity. |
| `code/models/r0_v2_elasticnet_7rad_mpd_thickness.py` | `R0_v2/analysis_elasticnet_7rad_mpd_thickness_20260616` | Earlier fixed-panel analysis retained for the deployable model export; it is not the current manuscript comparison entry point. |
| `code/models/r0_v3_apparent_model_comparison.py` | `R0_v3/hobeika_completion_20260715` | Reproduces apparent elastic-net radiomics/radioclinical comparisons, published preoperative DP-FRS and 2025 DISPAIR benchmarks, calibration, coefficients, and paired DeLong tests. |
| `code/models/locked_panel_candidate_632plus.py` | `R0_v3/model_family_2000_stratified_632plus_20260721` | Reproduces six-estimator screening with 2,000 paired class-stratified bootstrap `.632+` resamples. |
| `code/models/bootstrap_oob_cutpoints.py` | `R0_v3/bootstrap_oob_cutpoints_20260721` | Reproduces 2,000-resample in-bag constrained-MCC cutpoint derivation and unchanged out-of-bag evaluation. |
| `code/source_snapshots/popf_stabl_corrected_parallel_enhanced_v3_publish_github_original.py` | `sept25/popf_stabl_corrected_parallel_enhanced_v3.py` | STABL model-development script for the 7-rad EN run used in Figure 3. |
| `code/source_snapshots/popf_stabl_nested_experiments_original.py` | `sept25/popf_stabl_nested_experiments.py` | Nested STABL feature-selection script. |
| `code/models/comparative_risk_stratification_v2.py` | `R0` / manuscript regeneration utility | Comparative score/risk-stratification utility retained for earlier figure generations. |
| `code/models/nested_unweighted_calibration.py` | `R0/nested_unweighted_20260505/run_nested_unweighted_calibration.py` | Clean unweighted fixed-L2 OOF probability utility; no longer the current R0_v2 primary estimator. |
| `code/models/create_v2_style_figures_from_clean_analysis.py` | `R0/nested_unweighted_20260505/create_v2_style_figures_from_clean_analysis.py` | V2-style figure utility from clean fixed-L2 OOF predictions. |
| `code/utils/plotting_utils.py` | `utils/plotting_utils.py` | Shared figure helpers. |
| `code/utils/data_utils.py` | `utils/data_utils.py` | (Add once finalized). |
| `stabl` | pip dependency | Install from the upstream repo and document the commit SHA in `docs/setup_env.md` notes. |

The public export includes only de-identified model-ready CSVs. It excludes source clinical databases, imaging data, segmentations, direct identifiers, and patient-level prediction outputs.
