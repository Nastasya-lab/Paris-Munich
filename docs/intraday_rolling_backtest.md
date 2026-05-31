# Intraday rolling backtest

This report expands the fixed winter holdout into monthly forward folds. Each fold uses only information from earlier dates.

## Leakage-safe design

- rolling period: `2025-08-01` to `2025-12-30`
- evaluated rows: `1059`
- event rows Tmax >=20C / >=25C / >=30C: `266` / `133` / `35`
- every month refits the ICON-D2 residual prior and restricts METAR analogues plus Tmax timing climatology to prior dates

## Overall metrics

| model_variant | rows | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_ge_20 | brier_ge_25 | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | 1059 | 0.9380 | 1.2586 | 0.4517 | 4.1483 | 0.0073 | 0.0178 | 0.0205 | 0.0150 | 0.8196 |
| icon_d2_prior_plus_intraday | 1059 | 0.8213 | 1.1979 | -0.0523 | 3.3649 | 0.0076 | 0.0188 | 0.0163 | 0.0064 | 0.8848 |

## By fold

| model_variant | fold_start | rows | mae_expected | rmse_expected | mean_nll | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | 2025-08-01 | 216 | 1.1786 | 1.3826 | 6.7374 | 0.0067 | 0.0497 | 0.7130 |
| icon_d2_prior | 2025-09-01 | 210 | 0.7384 | 0.9734 | 3.6692 | 0.0064 | 0.0244 | 0.8905 |
| icon_d2_prior | 2025-10-01 | 213 | 0.8549 | 1.2229 | 3.5651 | 0.0071 | 0.0000 | 0.8404 |
| icon_d2_prior | 2025-11-01 | 210 | 0.7587 | 0.9542 | 3.5184 | 0.0067 | 0.0000 | 0.8619 |
| icon_d2_prior | 2025-12-01 | 210 | 1.1535 | 1.6269 | 3.1857 | 0.0098 | 0.0000 | 0.7952 |
| icon_d2_prior_plus_intraday | 2025-08-01 | 216 | 0.7446 | 0.9555 | 3.9355 | 0.0063 | 0.0235 | 0.8843 |
| icon_d2_prior_plus_intraday | 2025-09-01 | 210 | 0.7526 | 1.0234 | 3.5494 | 0.0071 | 0.0082 | 0.9000 |
| icon_d2_prior_plus_intraday | 2025-10-01 | 213 | 0.7848 | 1.2391 | 3.1491 | 0.0072 | 0.0000 | 0.8732 |
| icon_d2_prior_plus_intraday | 2025-11-01 | 210 | 0.7612 | 1.0640 | 3.2479 | 0.0073 | 0.0000 | 0.8857 |
| icon_d2_prior_plus_intraday | 2025-12-01 | 210 | 1.0661 | 1.6003 | 2.9294 | 0.0103 | 0.0000 | 0.8810 |

## Interpretation

- The rolling check is the broader historical test because it includes warm-season rows and 35 Tmax >=30C cases.
- The intraday layer improves overall MAE, RMSE, NLL, Brier >=25C, Brier >=30C, and 80% interval coverage.
- CRPS is almost neutral but slightly worse overall, mainly because early-day uncertainty widening and late-day sharpening are not yet separately calibrated.
- The strongest expected-value gains are after 12 UTC and on days where the observed METAR maximum has already dropped by several degrees.
- Brier >=20C is slightly worse, so threshold calibration should not be tightened solely from this MVP layer.

## By issue hour

| model_variant | issue_hour_utc | rows | mae_expected | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | 0 | 152 | 1.0988 | 0.0096 | 0.0174 | 0.7961 |
| icon_d2_prior | 3 | 151 | 0.9727 | 0.0090 | 0.0130 | 0.8344 |
| icon_d2_prior | 6 | 151 | 0.9742 | 0.0090 | 0.0130 | 0.8344 |
| icon_d2_prior | 9 | 150 | 0.9021 | 0.0082 | 0.0143 | 0.8600 |
| icon_d2_prior | 12 | 151 | 0.8090 | 0.0064 | 0.0142 | 0.8742 |
| icon_d2_prior | 15 | 152 | 0.9036 | 0.0046 | 0.0165 | 0.7763 |
| icon_d2_prior | 18 | 152 | 0.9045 | 0.0045 | 0.0165 | 0.7632 |
| icon_d2_prior_plus_intraday | 0 | 152 | 1.0701 | 0.0096 | 0.0128 | 0.8947 |
| icon_d2_prior_plus_intraday | 3 | 151 | 1.0467 | 0.0093 | 0.0100 | 0.9139 |
| icon_d2_prior_plus_intraday | 6 | 151 | 1.2251 | 0.0104 | 0.0094 | 0.9338 |
| icon_d2_prior_plus_intraday | 9 | 150 | 0.9347 | 0.0091 | 0.0074 | 0.9467 |
| icon_d2_prior_plus_intraday | 12 | 151 | 0.5880 | 0.0068 | 0.0046 | 0.9536 |
| icon_d2_prior_plus_intraday | 15 | 152 | 0.4697 | 0.0042 | 0.0007 | 0.8026 |
| icon_d2_prior_plus_intraday | 18 | 152 | 0.4192 | 0.0041 | 0.0002 | 0.7500 |

## Selected regimes

| model_variant | regime | rows | mae_expected | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | actual_peak_passed | 412 | 0.9150 | 0.0046 | 0.0158 | 0.8058 |
| icon_d2_prior | all | 1059 | 0.9380 | 0.0073 | 0.0150 | 0.8196 |
| icon_d2_prior | drop_ge_3c | 157 | 0.9021 | 0.0056 | 0.0243 | 0.7580 |
| icon_d2_prior | drop_ge_5c | 54 | 0.9858 | 0.0044 | 0.0141 | 0.6667 |
| icon_d2_prior | dry_recent | 815 | 0.9991 | 0.0076 | 0.0194 | 0.7914 |
| icon_d2_prior | early_day_utc_00_06 | 454 | 1.0154 | 0.0092 | 0.0145 | 0.8216 |
| icon_d2_prior | late_day_utc_12_18 | 455 | 0.8725 | 0.0052 | 0.0157 | 0.8044 |
| icon_d2_prior | precip_recent | 244 | 0.7339 | 0.0065 | 0.0003 | 0.9139 |
| icon_d2_prior_plus_intraday | actual_peak_passed | 412 | 0.5379 | 0.0038 | 0.0010 | 0.8471 |
| icon_d2_prior_plus_intraday | all | 1059 | 0.8213 | 0.0076 | 0.0064 | 0.8848 |
| icon_d2_prior_plus_intraday | drop_ge_3c | 157 | 0.6222 | 0.0047 | 0.0103 | 0.7580 |
| icon_d2_prior_plus_intraday | drop_ge_5c | 54 | 0.5317 | 0.0028 | 0.0006 | 0.7222 |
| icon_d2_prior_plus_intraday | dry_recent | 815 | 0.8595 | 0.0078 | 0.0083 | 0.8699 |
| icon_d2_prior_plus_intraday | early_day_utc_00_06 | 454 | 1.1139 | 0.0097 | 0.0107 | 0.9141 |
| icon_d2_prior_plus_intraday | late_day_utc_12_18 | 455 | 0.4921 | 0.0050 | 0.0018 | 0.8352 |
| icon_d2_prior_plus_intraday | precip_recent | 244 | 0.6938 | 0.0072 | 0.0003 | 0.9344 |

## Fold inventory

- `2025-08-01` to `2025-08-31`: `evaluated`, prior train rows `34`, test rows `216`
- `2025-09-01` to `2025-09-30`: `evaluated`, prior train rows `251`, test rows `210`
- `2025-10-01` to `2025-10-31`: `evaluated`, prior train rows `461`, test rows `213`
- `2025-11-01` to `2025-11-30`: `evaluated`, prior train rows `674`, test rows `210`
- `2025-12-01` to `2025-12-30`: `evaluated`, prior train rows `884`, test rows `210`

## Limitations

- The first August fold has only a short ICON-D2 residual history and is a warm-start stress test, not a mature production-model estimate.
- Historical feature rows are evaluated at 00/03/06/09/12/15/18 UTC. The Railway +01:40 schedule still requires forward monitoring.
- Only five expanding monthly folds are available because forecast-as-issued ICON-D2 coverage and historical METAR overlap is limited.

## Operational decision

Treat the rolling result as the broader historical check and keep the fixed winter holdout as a stricter mature-history slice. Continue forward monitoring on the Railway availability-aware schedule before tightening production acceptance gates.