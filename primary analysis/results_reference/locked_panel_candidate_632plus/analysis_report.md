# Paired stratified bootstrap .632+ model-family screening

- Locked radiomics panel: 7 features
- Cohort: 195 patients; 36 CR-POPF events
- Valid paired class-stratified bootstrap replicates: 2000
- Scaling and estimator coefficients refitted in every bootstrap sample
- Hyperparameters tuned once on the full cohort and held fixed

## Model estimates

| model                           | probability_source                                                                  |   apparent_auc |   inner_cv_auc_for_tuning |   auc_632plus |   auc_632plus_ci_low |   auc_632plus_ci_high |   mean_oob_auc |   valid_replicates | hyperparameters                            |
|:--------------------------------|:------------------------------------------------------------------------------------|---------------:|--------------------------:|--------------:|---------------------:|----------------------:|---------------:|-------------------:|:-------------------------------------------|
| Elastic-net logistic regression | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.825472 |                  0.790356 |      0.788636 |             0.696508 |              0.855625 |       0.774006 |               2000 | {"C": 3.0, "l1_ratio": 0.05}               |
| L2 logistic regression          | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.825821 |                  0.790356 |      0.788647 |             0.696298 |              0.855616 |       0.774016 |               2000 | {"C": 3.0}                                 |
| Support vector machine          | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.821454 |                  0.705975 |      0.717251 |             0.559141 |              0.821724 |       0.683503 |               2000 | {"C": 0.1, "gamma": 0.1}                   |
| Random forest                   | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.997205 |                  0.712788 |      0.746919 |             0.600473 |              0.862925 |       0.694725 |               2000 | {"max_depth": 5, "min_samples_leaf": 2}    |
| XGBoost                         | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.990217 |                  0.729036 |      0.76636  |             0.631798 |              0.873696 |       0.71303  |               2000 | {"learning_rate": 0.03, "max_depth": 3}    |
| LightGBM                        | apparent ROC from full-cohort fit; AUC from paired class-stratified bootstrap .632+ |       0.980433 |                  0.719602 |      0.759538 |             0.623218 |              0.865605 |       0.706537 |               2000 | {"min_child_samples": 10, "num_leaves": 3} |

## Paired differences

| comparison                                                   |   mean_delta_auc_632plus |      ci_low |    ci_high |   paired_replicates |
|:-------------------------------------------------------------|-------------------------:|------------:|-----------:|--------------------:|
| L2 logistic regression minus Elastic-net logistic regression |              1.06688e-05 | -0.00181634 | 0.00174124 |                2000 |
| LightGBM minus Elastic-net logistic regression               |             -0.0290987   | -0.144383   | 0.0704949  |                2000 |
| Random forest minus Elastic-net logistic regression          |             -0.0417177   | -0.158727   | 0.0509755  |                2000 |
| Support vector machine minus Elastic-net logistic regression |             -0.0713854   | -0.194148   | 0.0202253  |                2000 |
| XGBoost minus Elastic-net logistic regression                |             -0.0222766   | -0.135634   | 0.0716383  |                2000 |
