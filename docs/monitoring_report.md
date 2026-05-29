# Monitoring report

Daily target rows: `2189`
Training rows: `15309`
NWP archive rows: `3`
AWC live METAR rows: `2`
AWC live TAF rows: `2`
Forecast log rows: `14`
Forecast monitoring rows with outcomes: `0`
Forecast outcome status rows: `14`

## Active model

Model version: `quantile_mvp_issue_features_20260529`
Calibrator version: `quantile_mvp_issue_features_20260529.calibrator`
Model path: `data\models\quantile_mvp_issue_features_20260529.joblib`
Calibrator path: `data\models\quantile_mvp_issue_features_20260529.calibrator.joblib`
Model exists: `True`
Calibrator exists: `True`

## Registry health

Passed: `True`

## Archive freshness

METAR: `fresh`
TAF: `fresh`
NWP: `fresh`
Freshness gate passed: `True`

## Latest retraining

Model version: `quantile_mvp_issue_features_20260529`
Promoted: `True`
Rows: `15309`

## Outcome analysis

Status: `pending`
Rows: `0`

## Model artifacts

- `model_registry.json`
- `quantile_mvp.calibrator.joblib`
- `quantile_mvp.calibrator.metadata.json`
- `quantile_mvp.joblib`
- `quantile_mvp.metadata.json`
- `quantile_mvp_issue_features_20260529.calibrator.joblib`
- `quantile_mvp_issue_features_20260529.calibrator.metadata.json`
- `quantile_mvp_issue_features_20260529.joblib`
- `quantile_mvp_issue_features_20260529.metadata.json`
- `quantile_mvp_retrain_20260528.calibrator.joblib`
- `quantile_mvp_retrain_20260528.calibrator.metadata.json`
- `quantile_mvp_retrain_20260528.joblib`
- `quantile_mvp_retrain_20260528.metadata.json`
- `quantile_mvp_retrain_20260529.calibrator.joblib`
- `quantile_mvp_retrain_20260529.calibrator.metadata.json`
- `quantile_mvp_retrain_20260529.joblib`
- `quantile_mvp_retrain_20260529.metadata.json`

## Leakage audit

- `max_metar_knowledge_time_utc`: `0`
- `max_nwp_knowledge_time_utc`: `0`
- `latest_metar_time_utc`: `0`
- `forbidden_target_feature_columns`: `0`
- `rows`: `15309`

## Calibration comparison

| forecast_variant | rows | mean_nll | mean_crps | coverage_50 | coverage_80 | coverage_90 | brier_ge_20 | brier_ge_25 | brier_ge_30 | selected_for_production |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_spread | 360 | 2.4413893831 | 0.9323430385 | 0.4972222222 | 0.8083333333 | 0.9416666667 | 0.0650994867 | 0.0407192247 | 0.0153234201 | True |
| raw | 360 | 2.8937126366 | 0.936694904 | 0.425 | 0.7055555556 | 0.8138888889 | 0.0671631155 | 0.0406581371 | 0.0148311144 | False |
| calibrated_isotonic_cdf | 360 | 3.9277745753 | 0.9386782574 | 0.4333333333 | 0.7388888889 | 0.8055555556 | 0.0667672618 | 0.040724446 | 0.0144886117 | False |

## Rolling summary

| forecast_variant | issue_hour_utc | rows | mae_median | rmse_mean | mean_nll | mean_crps | coverage_50 | coverage_80 | coverage_90 | brier_ge_20 | brier_ge_25 | brier_ge_30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_isotonic_cdf | 6 | 360.0 | 2.4630555556 | 3.1145285609 | 5.2058537656 | 0.9362168013 | 0.4361111111 | 0.7611111111 | 0.8388888889 | 0.0697003317 | 0.0423343481 | 0.0153929035 |
| calibrated_isotonic_cdf | 18 | 360.0 | 0.4547222222 | 0.7716584646 | 3.471194209 | 0.9877505655 | 0.2833333333 | 0.45 | 0.4555555556 | 0.0244191882 | 0.0146602887 | 0.0046149585 |
| calibrated_spread | 6 | 360.0 | 2.4325 | 3.0970430426 | 2.4688542962 | 0.9324024101 | 0.4861111111 | 0.8166666667 | 0.9333333333 | 0.0649767702 | 0.0407550703 | 0.015447518 |
| calibrated_spread | 18 | 360.0 | 0.4386111111 | 0.9485946299 | 0.961798044 | 0.9727866212 | 0.3916666667 | 0.9055555556 | 0.9916666667 | 0.0182007771 | 0.010638837 | 0.004846489 |
| raw | 6 | 360.0 | 2.4502777778 | 3.0970430426 | 3.2052184174 | 0.936790674 | 0.4055555556 | 0.7 | 0.8361111111 | 0.0672588381 | 0.0418840889 | 0.0151271141 |
| raw | 18 | 360.0 | 0.4358333333 | 0.9485946299 | 1.0756073705 | 0.9754196 | 0.2694444444 | 0.6583333333 | 0.8027777778 | 0.0197289457 | 0.0109769643 | 0.0048506358 |

## Operational by model

No rows.

## Operational source mismatch

No rows.

## Operational availability

No rows.

## Operational acceptance

No rows.

## Operational forecast inventory

| model_version | airport | logged_forecasts | first_issue_time_utc | latest_issue_time_utc | metar_missing_rate | taf_missing_rate | nwp_missing_rate | accepted_rate | rejected_rate | unknown_acceptance_rate | metar_sources | taf_sources | nwp_sources | quality_statuses | acceptance_blocking_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| climatology_mvp | EDDM | 6 | 2026-05-28T09:01:15.275344+00:00 | 2026-07-15T06:00:00+00:00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |  |  |  | unknown |  |
| quantile_mvp_issue_features_20260529 | EDDM | 5 | 2026-05-29T15:00:00+00:00 | 2026-05-29T15:00:00+00:00 | 0.0 | 0.0 | 0.0 | 0.8 | 0.2 | 0.0 | awc.metar.live.EDDM | awc.taf.live.EDDM | open_meteo.live.icon_d2 | degraded, ok | quality_status_ok, has_no_hard_reasons |
| quantile_mvp_retrain_20260529 | EDDM | 3 | 2026-05-29T06:10:37.561494+00:00 | 2026-05-29T06:15:48.218553+00:00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | awc.metar.live.EDDM | awc.taf.live.EDDM | open_meteo.live.icon_d2 | unknown |  |

## Operational pending forecasts

| outcome_status | model_version | forecast_accepted | forecast_quality_status | forecasts | first_target_date_local | latest_target_date_local | acceptance_blocking_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pending_truth | climatology_mvp | unknown | unknown | 6 | 2026-07-15 | 2026-07-15 |  |
| pending_truth | quantile_mvp_issue_features_20260529 | accepted | ok | 4 | 2026-05-29 | 2026-05-29 |  |
| pending_truth | quantile_mvp_issue_features_20260529 | rejected | degraded | 1 | 2026-05-29 | 2026-05-29 | quality_status_ok, has_no_hard_reasons |
| pending_truth | quantile_mvp_retrain_20260529 | unknown | unknown | 3 | 2026-05-29 | 2026-05-29 |  |
