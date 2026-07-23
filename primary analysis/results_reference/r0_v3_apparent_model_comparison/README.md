# Apparent model comparison

These aggregate outputs reproduce the current full-cohort radiomics and
radioclinical comparison:

```bash
python "code/models/r0_v3_apparent_model_comparison.py" \
  --auc-bootstrap 5000
```

The directory contains apparent AUCs with simple-bootstrap confidence
intervals, paired DeLong comparisons, calibration points, elastic-net
coefficients, and tuning parameters. It contains no patient-level predictions.
