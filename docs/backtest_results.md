# Backtest results

Current report covers the climatology baseline on available DWD targets. NWP forecast-as-issued backtests remain separated until honest issued archives exist.

Rows: 438
MAE median: 3.543150684931507
RMSE mean: 4.229571994361522


Quantile MVP holdout, issue hour 06 UTC, test from 2025-01-01:
Rows: 360
Raw MAE/RMSE: 2.3825000000000003 / 3.041662065086618
Raw NLL/CRPS: 2.8937126365839805 / 0.9366949039810141
Raw coverage 50/80/90: 0.425 / 0.7055555555555556 / 0.8138888888888889
Calibrated spread sigma bins: 1.25
Calibrated MAE/RMSE: 2.3997222222222225 / 3.0416620650866184
Calibrated NLL/CRPS: 2.441389383129616 / 0.9323430384793384
Calibrated coverage 50/80/90: 0.49722222222222223 / 0.8083333333333333 / 0.9416666666666667
Calibrated Brier ge20/ge25/ge30: 0.06509948669387937 / 0.040719224710150305 / 0.015323420109800596
Isotonic CDF MAE/RMSE: 2.3908333333333336 / 3.0369422640580797
Isotonic CDF NLL/CRPS: 3.927774575315914 / 0.9386782573728654
Isotonic CDF coverage 50/80/90: 0.43333333333333335 / 0.7388888888888889 / 0.8055555555555556
Isotonic CDF Brier ge20/ge25/ge30: 0.06676726180575353 / 0.04072444599264593 / 0.014488611689317471

Generated plots:
- `data/reports/pit_raw.png`
- `data/reports/pit_calibrated_spread.png`
- `data/reports/reliability_ge_25_calibrated.png`
- `data/reports/reliability_ge_30_calibrated.png`

Quick expanding rolling backtest:

- `data/reports/rolling_quantile_backtest.parquet`
- `data/reports/rolling_quantile_summary.parquet`
- `data/reports/rolling_quantile_seasonal_summary.parquet`

Default quick run covers issue hours `06` and `18` UTC for the 2025 test window, calibrated on 2024 and trained through 2023. Full mode is available with:

```bash
python scripts/08b_run_rolling_backtest.py --full
```

Quick summary:

- 06 UTC raw coverage 50/80/90: `0.406 / 0.700 / 0.836`
- 06 UTC spread coverage 50/80/90: `0.486 / 0.817 / 0.933`
- 06 UTC isotonic CDF coverage 50/80/90: `0.436 / 0.761 / 0.839`
- 18 UTC raw coverage 50/80/90: `0.269 / 0.658 / 0.803`
- 18 UTC spread coverage 50/80/90: `0.392 / 0.906 / 0.992`
- 18 UTC isotonic CDF coverage 50/80/90: `0.283 / 0.450 / 0.456`

Decision: keep isotonic CDF as experimental/report-only for now. It underperforms validation-fitted spread calibration in both the holdout report and quick rolling report.

## Intraday update holdout

The same-day METAR and sampled ICON-D2 update layer now has a separate leakage-safe fixed holdout report:

- `docs/intraday_backtest.md`
- `data/reports/intraday_backtest_summary.parquet`
- `data/reports/intraday_backtest_by_hour.parquet`
- `data/reports/intraday_backtest_by_regime.parquet`

The holdout covers `2025-11-01` to `2025-12-30`. All ICON-D2 residual prior rows, METAR analogue rows, and Tmax timing rows used by the update layer are restricted to dates before `2025-11-01`.

Overall comparison:

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
