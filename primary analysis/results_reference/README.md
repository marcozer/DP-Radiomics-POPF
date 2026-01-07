# Reference Results

Populate this directory after running the discovery and fixed-panel evaluation commands in `docs/runbook.md`.

Recommended layout:
```
results_reference/
  dev_run/
    discovery/
      command.txt
      metrics.json
      panel.txt
      plots/
    evaluation/
      command.txt
      metrics.json
      roc.png
      calibration.png
  publish_discovery/
  publish_eval/
```

Include a short `metadata.json` in each subfolder summarizing:
- Command string
- Git commit SHA
- Python version and platform
- Random seed(s)
- Panel file path

Compress large outputs (plots, JSON) only when distributing externally; keep raw files uncompressed in the repo for auditability.
