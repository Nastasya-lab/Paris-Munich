# Outcome analysis

Status: `ready`
Rows: `1`

## Overall

- `forecasts`: `1`
- `mae_expected`: `0.3999999999999986`
- `bias_expected`: `-0.3999999999999986`
- `mean_nll`: `0.5108256237659907`
- `mean_crps`: `0.6799999999999999`
- `brier_ge_20`: `0.0`
- `brier_ge_25`: `0.16000000000000003`
- `brier_ge_30`: `0.0`

## By model

| model_version | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| m1 | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## By forecast quality

| forecast_quality_status | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| None | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## By forecast acceptance

| forecast_accepted | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| unknown | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## By source mismatch

| any_source_mismatch | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| False | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## By METAR source compatibility

| metar_source_compatibility_status | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| None | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## By TAF source compatibility

| taf_source_compatibility_status | forecasts | mae_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- |
| None | 1 | 0.4 | -0.4 | 0.5108256238 | 0.68 |

## Worst forecasts by CRPS

| forecast_id | model_version | target_date_local | actual_tmax_c | expected_tmax_c | error_expected_c | nll | crps | forecast_quality_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f1 | m1 | 2026-07-15 | 25.0 | 24.6 | -0.4 | 0.5108256238 | 0.68 | None |
