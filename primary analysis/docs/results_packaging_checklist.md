# Results Packaging Checklist

- [ ] Alignment check CSVs show zero unmatched IDs or include rationale for exclusions.
- [ ] `configs/panels/publish_lr_panel.txt` exists, matches discovery output hash, and is referenced in evaluation command logs.
- [ ] `code/` contains only the scripts required for reproduction and does not include third-party code copies unless explicitly licensed and documented.
- [ ] `requirements.txt`, `docs/setup_env.md`, and `docs/runbook.md` reflect the exact commands used; update version pins if any packages were upgraded.
- [ ] `results_reference/*/` folders include `command.txt`, `metrics.json`, `panel.json`, `plots/`, and `logs/` subfolders.
- [ ] No PHI-containing CSVs leave the repo: only schema docs or synthetic templates under `data_templates/`.
- [ ] Generated plots (ROC, calibration, feature importance) exist in both PNG and PDF/SVG when possible.
- [ ] Tests executed (`pytest -q`) and status recorded in `results_reference/test_status.txt`.
- [ ] SHA256 hashes recorded for each top-level script in `code/` (use `shasum -a 256`).
- [ ] README in `publication_export/` updated with final command references before sharing externally.
