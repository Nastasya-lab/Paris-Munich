# Backtest results

Current report covers the climatology baseline on available DWD targets. NWP forecast-as-issued backtests remain separated until honest issued archives exist.

Rows: 468
MAE median: 3.6534188034188033
RMSE mean: 4.40326035347395


Quantile MVP holdout, issue hour 06 UTC, test from 2025-01-01:
Rows: 507
Raw MAE/RMSE: 3.394674556213018 / 4.613416837301871
Raw NLL/CRPS: 4.885223420787219 / 0.030716105846409793
Raw coverage 50/80/90: 0.42011834319526625 / 0.6410256410256411 / 0.7218934911242604
Calibrated spread sigma bins: 2.25
Calibrated MAE/RMSE: 3.4171597633136095 / 4.613416837301871
Calibrated NLL/CRPS: 3.0222391817663334 / 0.03010888140217276
Calibrated coverage 50/80/90: 0.5187376725838264 / 0.8027613412228797 / 0.8757396449704142
Calibrated Brier ge20/ge25/ge30: 0.070672696449986 / 0.04161429351286225 / 0.014865070349069153
Isotonic CDF MAE/RMSE: 3.370216962524655 / 4.483350608156453
Isotonic CDF NLL/CRPS: 8.918887773687812 / 0.029817465935358462
Isotonic CDF coverage 50/80/90: 0.47928994082840237 / 0.6666666666666666 / 0.8382642998027613
Isotonic CDF Brier ge20/ge25/ge30: 0.071014259679178 / 0.04066120947974048 / 0.014449143331548025

Generated plots:
- `data/reports/pit_raw.png`
- `data/reports/pit_calibrated_spread.png`
- `data/reports/reliability_ge_25_calibrated.png`
- `data/reports/reliability_ge_30_calibrated.png`

## Quantile MVP rolling baseline

The recalculated quick expanding rolling report covers issue hours `06` and `18` UTC for the 2025 test window:

| forecast_variant | issue_hour_utc | rows | mae_median | rmse_mean | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 06 | 360 | 2.4542 | 3.0918 | 3.1511 | 0.0210 | 0.6972 |
| calibrated spread | 06 | 360 | 2.4297 | 3.0918 | 2.4551 | 0.0207 | 0.8111 |
| raw | 18 | 360 | 0.4292 | 0.9468 | 1.1121 | 0.0043 | 0.6306 |
| calibrated spread | 18 | 360 | 0.4319 | 0.9468 | 0.9714 | 0.0043 | 0.8944 |

Decision: keep isotonic CDF as experimental/report-only. Validation-fitted spread calibration remains the quantile MVP comparison baseline.

## NWP-aware holdout

The issued ICON-D2 holdout comparison is available in `docs/nwp_aware_holdout.md`.

| model_variant | rows | mae_expected | rmse_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| ICON-D2 residual distribution | 1029 | 0.7075 | 0.9421 | 1.4040 | 0.0068 |
| raw ICON-D2 model Tmax | 1029 | 0.7700 | 1.0182 | n/a | n/a |
| quantile MVP with NWP | 1029 | 5.1047 | 5.7635 | 5.1326 | 0.0388 |
| quantile MVP without NWP | 1029 | 6.8928 | 8.3799 | 10.8705 | 0.0688 |

Decision: keep the ICON-D2 residual distribution as the operational prior. The older quantile MVP remains a historical baseline and should not replace it.

## Intraday update holdout

The same-day METAR and sampled ICON-D2 update layer has separate leakage-safe reports:

- `docs/intraday_backtest.md`
- `docs/intraday_rolling_backtest.md`
- `data/reports/intraday_backtest_summary.parquet`
- `data/reports/intraday_rolling_backtest_summary.parquet`

Fixed mature-history comparison, November-December 2025:

| model_variant | rows | mae_expected | rmse_expected | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- |
| ICON-D2 prior | 420 | 0.9489 | 1.3307 | 3.4422 | 0.0083 | 0.8286 |
| ICON-D2 prior plus intraday update | 420 | 0.9113 | 1.3609 | 3.1859 | 0.0088 | 0.8857 |

Expanding rolling comparison, August-December 2025:

| model_variant | rows | mae_expected | rmse_expected | mean_nll | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ICON-D2 prior | 1059 | 0.9380 | 1.2586 | 4.1483 | 0.0073 | 0.0150 | 0.8196 |
| ICON-D2 prior plus intraday update | 1059 | 0.8213 | 1.1979 | 3.3649 | 0.0076 | 0.0064 | 0.8848 |

Decision: keep the intraday layer enabled as a monitored secondary correction. It improves MAE, RMSE, NLL, Brier for `Tmax >=25C` and `Tmax >=30C`, and 80% interval coverage on the rolling check. CRPS is slightly worse, so final distribution sharpness still needs calibration and forward monitoring.
