# Intraday seasonal weight analysis

This is a research-only report. Production intraday weights were not changed.

## Data Limits

- eligible period: `2025-07-27` to `2025-12-31`
- eligible rows: `1100`
- warning: Open-Meteo Single Runs ICON-D2 temperature coverage before late July 2025 is not usable in the local archive.

## Warm Scenario

Warm-season cold-start profile: fit on August 2025, test on September 2025.

- validation period: `2025-08-01` to `2025-08-31`
- test period: `2025-09-01` to `2025-09-30`
- validation inventory: `{'rows': 216, 'days': 31, 'ge20_rows': 175, 'ge25_rows': 98, 'ge30_rows': 35, 'prior_train_rows_min': 34, 'prior_train_rows_max': 34}`
- test inventory: `{'rows': 210, 'days': 30, 'ge20_rows': 84, 'ge25_rows': 35, 'ge30_rows': 0, 'prior_train_rows_min': 251, 'prior_train_rows_max': 251}`

Optimized validation profiles:

| objective | utc_00_03 | utc_06_09 | utc_12 | utc_15_18 | validation MAE | validation CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crps | 0.10 | 0.10 | 0.25 | 0.70 | 0.8460 | 0.00525 |
| mae | 0.20 | 0.20 | 0.80 | 0.95 | 0.7006 | 0.00589 |
| nll | 0.20 | 0.20 | 0.60 | 0.95 | 0.7075 | 0.00567 |

Test summary:

| variant | rows | MAE | RMSE | NLL | CRPS | Brier >=30 | coverage80 | mean weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prior | 210 | 0.7384 | 0.9734 | 3.6692 | 0.00635 | 0.02444 | 0.8905 | 0.0000 |
| current_dynamic | 210 | 0.7526 | 1.0234 | 3.5494 | 0.00712 | 0.00816 | 0.9000 | 0.5500 |
| fixed_cool_challenger_from_previous_analysis | 210 | 0.6472 | 0.8844 | 3.5783 | 0.00614 | 0.01789 | 0.8952 | 0.1929 |
| optimized_crps | 210 | 0.6473 | 0.8645 | 3.4813 | 0.00621 | 0.01456 | 0.9000 | 0.2929 |
| optimized_mae | 210 | 0.6831 | 0.9136 | 3.5062 | 0.00681 | 0.01011 | 0.8905 | 0.5000 |
| optimized_nll | 210 | 0.6785 | 0.9102 | 3.5051 | 0.00670 | 0.01058 | 0.8952 | 0.4714 |

Best late override candidate by validation CRPS:

- condition: `late_drop_ge5`
- override weight: `0.95`
- validation trigger rows: `6`
- test trigger rows: `10`

## Cool Scenario

Cool-season profile: fit on September-October 2025, test on November-December 2025.

- validation period: `2025-09-01` to `2025-10-31`
- test period: `2025-11-01` to `2025-12-30`
- validation inventory: `{'rows': 423, 'days': 61, 'ge20_rows': 91, 'ge25_rows': 35, 'ge30_rows': 0, 'prior_train_rows_min': 251, 'prior_train_rows_max': 461}`
- test inventory: `{'rows': 420, 'days': 60, 'ge20_rows': 0, 'ge25_rows': 0, 'ge30_rows': 0, 'prior_train_rows_min': 674, 'prior_train_rows_max': 884}`

Optimized validation profiles:

| objective | utc_00_03 | utc_06_09 | utc_12 | utc_15_18 | validation MAE | validation CRPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| crps | 0.05 | 0.05 | 0.25 | 0.55 | 0.7075 | 0.00645 |
| mae | 0.05 | 0.05 | 0.65 | 0.95 | 0.6597 | 0.00667 |
| nll | 0.10 | 0.10 | 0.65 | 0.85 | 0.6780 | 0.00662 |

Test summary:

| variant | rows | MAE | RMSE | NLL | CRPS | Brier >=30 | coverage80 | mean weight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prior | 420 | 0.9561 | 1.3337 | 3.3521 | 0.00826 | 0.00000 | 0.8286 | 0.0000 |
| current_dynamic | 420 | 0.9137 | 1.3589 | 3.0886 | 0.00880 | 0.00000 | 0.8833 | 0.4714 |
| fixed_cool_challenger_from_previous_analysis | 420 | 0.8722 | 1.2848 | 3.0876 | 0.00829 | 0.00000 | 0.8476 | 0.1929 |
| optimized_crps | 420 | 0.8863 | 1.2932 | 3.0864 | 0.00827 | 0.00000 | 0.8476 | 0.2214 |
| optimized_mae | 420 | 0.8480 | 1.2728 | 3.0681 | 0.00864 | 0.00000 | 0.8262 | 0.3929 |
| optimized_nll | 420 | 0.8666 | 1.2890 | 3.0662 | 0.00854 | 0.00000 | 0.8571 | 0.3929 |

Best late override candidate by validation CRPS:

- condition: `late_drop_ge5`
- override weight: `0.85`
- validation trigger rows: `21`
- test trigger rows: `20`

## Recommendation

- Do not replace production weights with a single year-round profile.
- Keep the current aggressive warm-season behavior as a useful baseline, but evaluate a CRPS-oriented warm challenger in shadow mode.
- For cool months, prefer zero morning influence and moderate afternoon weights as the next challenger.
- Implement seasonal profiles only after a shadow-mode comparison on live 2026 summer data or a larger issued archive.