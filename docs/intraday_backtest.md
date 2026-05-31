# Intraday update backtest

This report evaluates whether the same-day METAR and sampled ICON-D2 update improves the ICON-D2 residual prior without temporal leakage.

## Leakage-safe design

- holdout: `2025-11-01` to `2025-12-30`
- ICON-D2 prior train period: `2025-07-27` to `2025-10-31`
- ICON-D2 prior train rows: `674`
- intraday analogue train period: `2020-01-01` to `2025-10-31`
- intraday analogue train rows: `14879`
- holdout rows: `420`
- all prior distributions, intraday analogues, and Tmax timing priors are restricted to dates before the holdout

## Overall metrics

| model_variant | rows | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | 420 | 0.9489 | 1.3307 | 0.2035 | 3.4422 | 0.0083 | 0.0000 | 0.8286 |
| icon_d2_prior_plus_intraday | 420 | 0.9113 | 1.3609 | -0.1135 | 3.1859 | 0.0088 | 0.0000 | 0.8857 |

Lower is better for MAE, RMSE, NLL, CRPS, and Brier score. Coverage 80 should be interpreted against the 0.80 target.

## Interpretation

- The intraday layer improves overall MAE, NLL, and 80% interval coverage. RMSE and CRPS are slightly worse on this mature winter slice.
- During 12/15/18 UTC issues, expected-value MAE improves materially because observations identify days where little remaining upside is plausible.
- During 00/03/06/09 UTC issues, expected-value MAE is worse but interval coverage improves: the layer mainly widens uncertainty before the daytime peak.
- For observed drops of at least 5C from the current-day METAR maximum, MAE improves from about 0.97C to 0.57C.
- The holdout contains no Tmax >=25C or >=30C events. It validates the mechanism on winter weather, not summer heat calibration.

## By issue hour

| model_variant | issue_hour_utc | rows | mae_expected | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | 0 | 60 | 1.0937 | 0.0107 | 0.0000 | 0.8167 |
| icon_d2_prior | 3 | 60 | 1.1814 | 0.0118 | 0.0000 | 0.7833 |
| icon_d2_prior | 6 | 60 | 1.1847 | 0.0117 | 0.0000 | 0.7833 |
| icon_d2_prior | 9 | 60 | 0.9368 | 0.0094 | 0.0000 | 0.8500 |
| icon_d2_prior | 12 | 60 | 0.7748 | 0.0070 | 0.0000 | 0.9000 |
| icon_d2_prior | 15 | 60 | 0.7398 | 0.0036 | 0.0000 | 0.8333 |
| icon_d2_prior | 18 | 60 | 0.7310 | 0.0035 | 0.0000 | 0.8333 |
| icon_d2_prior_plus_intraday | 0 | 60 | 1.2154 | 0.0110 | 0.0000 | 0.8833 |
| icon_d2_prior_plus_intraday | 3 | 60 | 1.2920 | 0.0119 | 0.0000 | 0.9000 |
| icon_d2_prior_plus_intraday | 6 | 60 | 1.3006 | 0.0119 | 0.0000 | 0.9000 |
| icon_d2_prior_plus_intraday | 9 | 60 | 1.1250 | 0.0105 | 0.0000 | 0.9000 |
| icon_d2_prior_plus_intraday | 12 | 60 | 0.6371 | 0.0065 | 0.0000 | 1.0000 |
| icon_d2_prior_plus_intraday | 15 | 60 | 0.4264 | 0.0050 | 0.0000 | 0.8167 |
| icon_d2_prior_plus_intraday | 18 | 60 | 0.3825 | 0.0049 | 0.0000 | 0.8000 |

## Selected regimes

| model_variant | regime | rows | mae_expected | mean_crps | brier_ge_30 | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- |
| icon_d2_prior | actual_peak_passed | 190 | 0.8181 | 0.0039 | 0.0000 | 0.8632 |
| icon_d2_prior | all | 420 | 0.9489 | 0.0083 | 0.0000 | 0.8286 |
| icon_d2_prior | drop_ge_3c | 38 | 0.9453 | 0.0056 | 0.0000 | 0.6579 |
| icon_d2_prior | drop_ge_5c | 21 | 0.9738 | 0.0037 | 0.0000 | 0.5714 |
| icon_d2_prior | dry_recent | 345 | 1.0202 | 0.0085 | 0.0000 | 0.8029 |
| icon_d2_prior | early_day_utc_00_06 | 180 | 1.1533 | 0.0114 | 0.0000 | 0.7944 |
| icon_d2_prior | late_day_utc_12_18 | 180 | 0.7485 | 0.0047 | 0.0000 | 0.8556 |
| icon_d2_prior | precip_recent | 75 | 0.6211 | 0.0069 | 0.0000 | 0.9467 |
| icon_d2_prior_plus_intraday | actual_peak_passed | 190 | 0.5915 | 0.0044 | 0.0000 | 0.8947 |
| icon_d2_prior_plus_intraday | all | 420 | 0.9113 | 0.0088 | 0.0000 | 0.8857 |
| icon_d2_prior_plus_intraday | drop_ge_3c | 38 | 0.6832 | 0.0055 | 0.0000 | 0.7105 |
| icon_d2_prior_plus_intraday | drop_ge_5c | 21 | 0.5670 | 0.0034 | 0.0000 | 0.6667 |
| icon_d2_prior_plus_intraday | dry_recent | 345 | 0.9746 | 0.0091 | 0.0000 | 0.8725 |
| icon_d2_prior_plus_intraday | early_day_utc_00_06 | 180 | 1.2693 | 0.0116 | 0.0000 | 0.8944 |
| icon_d2_prior_plus_intraday | late_day_utc_12_18 | 180 | 0.4820 | 0.0055 | 0.0000 | 0.8722 |
| icon_d2_prior_plus_intraday | precip_recent | 75 | 0.6202 | 0.0077 | 0.0000 | 0.9467 |

## Limitations

- Historical IEM METAR currently ends on 2025-12-30, so the simultaneous ICON-D2 plus METAR holdout ends there.
- Historical feature rows are evaluated at 00/03/06/09/12/15/18 UTC. The new Railway availability-aware +01:40 schedule remains a forward-test concern.
- The holdout covers November and December only; seasonal promotion requires more archived same-day evidence.
- The winter holdout has no Tmax >=25C or >=30C events, so warm-season threshold reliability remains untested.

## Operational decision

Keep the intraday layer enabled as a monitored secondary correction. Use this holdout to identify weak issue hours and regimes; do not treat the November-December slice as sufficient evidence for final seasonal calibration.