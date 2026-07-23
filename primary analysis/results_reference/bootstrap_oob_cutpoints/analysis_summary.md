# Bootstrap .632+ AUC and OOB cutpoint validation

Cohort: 195 patients; 36 CR-POPF events.
Predictors: locked seven-feature radiomics panel.
Model: standardized, unweighted elastic-net logistic regression.
Post-hoc calibration: none.

Apparent AUC: 0.825.
Bootstrap .632+ AUC: 0.789 (0.697-0.856).

Cutpoints were selected in bag using constrained MCC and evaluated unchanged out of bag.

- Rule-out: full-cohort threshold 0.101662; mean OOB sensitivity 0.838; mean OOB specificity 0.527; mean OOB MCC 0.288.
- Rule-in: full-cohort threshold 0.368364; mean OOB sensitivity 0.334; mean OOB specificity 0.921; mean OOB MCC 0.301.

Full-cohort risk strata:

- Rule-out: 86 patients, 3 events (3.5%).
- Intermediate: 83 patients, 15 events (18.1%).
- Rule-in: 26 patients, 18 events (69.2%).
