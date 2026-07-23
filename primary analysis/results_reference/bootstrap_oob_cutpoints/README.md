# Out-of-bag cutpoint validation

These aggregate outputs were generated from 2,000 class-stratified bootstrap
resamples:

```bash
python "code/models/bootstrap_oob_cutpoints.py" \
  --n-bootstrap 2000
```

The scaler and elastic-net coefficients are refitted in every sample.
Constrained-MCC cutpoints are selected in bag and evaluated unchanged out of
bag. No patient identifiers or patient-level predictions are included.
