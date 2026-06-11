# EDDM enhanced intraday METAR feature comparison

- created: `2026-06-11T18:55:08.976710+00:00`
- target: official DWD daily Tmax; METAR is used only as as-of intraday signal
- period: `2020-01-01` to `2025-12-30`
- rows: `15299`
- days: `2186`
- recommendation: `promote_to_main_model`

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_intraday_ml | 1063 | 152 | 1.6574 | 2.3548 | 0.3470 | 4.9550 | 0.0536 | 0.0968 | 0.0991 | 0.0982 | 0.0998 | 0.6435 | 0.3690 |
| enhanced_intraday_ml | 1063 | 152 | 1.6197 | 2.2815 | 0.3804 | 4.8235 | 0.0518 | 0.0951 | 0.0977 | 0.0935 | 0.0947 | 0.6331 | 0.3688 |

## By UTC Issue Hour

| model_variant | issue_hour_utc | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_intraday_ml | 0 | 152 | 152 | 2.8248 | 3.5307 | 0.1769 | 5.8486 | 0.0925 | 0.0657 | 0.0818 | 0.1439 | 0.1709 | 0.8224 | 0.4394 |
| base_intraday_ml | 3 | 152 | 152 | 2.7053 | 3.3233 | 0.4288 | 5.5828 | 0.0869 | 0.0706 | 0.0840 | 0.1358 | 0.1612 | 0.8289 | 0.4618 |
| base_intraday_ml | 6 | 152 | 152 | 2.5180 | 3.1122 | 0.3737 | 4.8505 | 0.0826 | 0.0692 | 0.0923 | 0.1279 | 0.1619 | 0.8553 | 0.4345 |
| base_intraday_ml | 9 | 151 | 151 | 1.4141 | 1.8598 | -0.3233 | 3.2406 | 0.0543 | 0.0860 | 0.1216 | 0.1732 | 0.1725 | 0.8013 | 0.3437 |
| base_intraday_ml | 12 | 152 | 152 | 0.8306 | 1.0060 | 0.5938 | 1.9773 | 0.0280 | 0.2297 | 0.2475 | 0.1045 | 0.0305 | 0.8487 | 0.3434 |
| base_intraday_ml | 15 | 152 | 152 | 0.6550 | 0.7535 | 0.5655 | 6.3590 | 0.0162 | 0.0897 | 0.0444 | 0.0013 | 0.0012 | 0.1711 | 0.2713 |
| base_intraday_ml | 18 | 152 | 152 | 0.6527 | 0.7529 | 0.6093 | 6.8148 | 0.0143 | 0.0668 | 0.0223 | 0.0013 | 0.0012 | 0.1776 | 0.2889 |
| enhanced_intraday_ml | 0 | 152 | 152 | 2.7807 | 3.4637 | 0.3382 | 5.1893 | 0.0903 | 0.0657 | 0.0832 | 0.1313 | 0.1674 | 0.7763 | 0.4499 |
| enhanced_intraday_ml | 3 | 152 | 152 | 2.5959 | 3.1656 | 0.4646 | 5.4604 | 0.0833 | 0.0646 | 0.0797 | 0.1246 | 0.1549 | 0.8224 | 0.4530 |
| enhanced_intraday_ml | 6 | 152 | 152 | 2.4118 | 2.9860 | 0.3735 | 5.4685 | 0.0783 | 0.0667 | 0.0870 | 0.1253 | 0.1446 | 0.8355 | 0.4384 |
| enhanced_intraday_ml | 9 | 151 | 151 | 1.3704 | 1.8146 | -0.3349 | 2.4966 | 0.0529 | 0.0842 | 0.1201 | 0.1642 | 0.1633 | 0.7947 | 0.3296 |
| enhanced_intraday_ml | 12 | 152 | 152 | 0.8488 | 1.0154 | 0.6172 | 1.9682 | 0.0276 | 0.2293 | 0.2478 | 0.1068 | 0.0310 | 0.8487 | 0.3481 |
| enhanced_intraday_ml | 15 | 152 | 152 | 0.6646 | 0.7632 | 0.5781 | 6.3534 | 0.0161 | 0.0888 | 0.0443 | 0.0014 | 0.0012 | 0.1776 | 0.2723 |
| enhanced_intraday_ml | 18 | 152 | 152 | 0.6636 | 0.7635 | 0.6212 | 6.8125 | 0.0142 | 0.0664 | 0.0223 | 0.0014 | 0.0012 | 0.1776 | 0.2904 |

## Rain/CB After Current Max

| model_variant | rain_or_cb_after_max | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_intraday_ml | False | 777 | 139 | 1.7175 | 2.4467 | 0.1463 | 5.5124 | 0.0560 | 0.0877 | 0.0874 | 0.0825 | 0.0878 | 0.6396 | 0.3657 |
| base_intraday_ml | True | 286 | 69 | 1.4941 | 2.0847 | 0.8925 | 3.4406 | 0.0471 | 0.1217 | 0.1308 | 0.1408 | 0.1326 | 0.6538 | 0.3780 |
| enhanced_intraday_ml | False | 777 | 139 | 1.6653 | 2.3507 | 0.1937 | 5.4356 | 0.0536 | 0.0860 | 0.0868 | 0.0781 | 0.0810 | 0.6268 | 0.3685 |
| enhanced_intraday_ml | True | 286 | 69 | 1.4956 | 2.0817 | 0.8876 | 3.1605 | 0.0471 | 0.1198 | 0.1274 | 0.1354 | 0.1320 | 0.6503 | 0.3697 |
