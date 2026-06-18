# EDDM enhanced intraday METAR feature comparison

- created: `2026-06-11T19:41:07.455478+00:00`
- target: official DWD daily Tmax; wind/advection uses only as-of EDDM METAR
- period: `2020-01-01` to `2025-12-30`
- rows: `15299`
- days: `2186`
- recommendation: `do_not_promote_yet`

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 1063 | 152 | 1.6197 | 2.2815 | 0.3804 | 4.8235 | 0.0518 | 0.0951 | 0.0977 | 0.0935 | 0.0947 | 0.6331 | 0.3688 |
| enhanced_wind_advection_ml | 1063 | 152 | 1.6061 | 2.2639 | 0.3854 | 4.9988 | 0.0516 | 0.0963 | 0.0988 | 0.0943 | 0.0939 | 0.6209 | 0.3708 |

## By UTC Issue Hour

| model_variant | issue_hour_utc | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 0 | 152 | 152 | 2.7807 | 3.4637 | 0.3382 | 5.1893 | 0.0903 | 0.0657 | 0.0832 | 0.1313 | 0.1674 | 0.7763 | 0.4499 |
| enhanced_intraday_ml | 3 | 152 | 152 | 2.5959 | 3.1656 | 0.4646 | 5.4604 | 0.0833 | 0.0646 | 0.0797 | 0.1246 | 0.1549 | 0.8224 | 0.4530 |
| enhanced_intraday_ml | 6 | 152 | 152 | 2.4118 | 2.9860 | 0.3735 | 5.4685 | 0.0783 | 0.0667 | 0.0870 | 0.1253 | 0.1446 | 0.8355 | 0.4384 |
| enhanced_intraday_ml | 9 | 151 | 151 | 1.3704 | 1.8146 | -0.3349 | 2.4966 | 0.0529 | 0.0842 | 0.1201 | 0.1642 | 0.1633 | 0.7947 | 0.3296 |
| enhanced_intraday_ml | 12 | 152 | 152 | 0.8488 | 1.0154 | 0.6172 | 1.9682 | 0.0276 | 0.2293 | 0.2478 | 0.1068 | 0.0310 | 0.8487 | 0.3481 |
| enhanced_intraday_ml | 15 | 152 | 152 | 0.6646 | 0.7632 | 0.5781 | 6.3534 | 0.0161 | 0.0888 | 0.0443 | 0.0014 | 0.0012 | 0.1776 | 0.2723 |
| enhanced_intraday_ml | 18 | 152 | 152 | 0.6636 | 0.7635 | 0.6212 | 6.8125 | 0.0142 | 0.0664 | 0.0223 | 0.0014 | 0.0012 | 0.1776 | 0.2904 |
| enhanced_wind_advection_ml | 0 | 152 | 152 | 2.7446 | 3.4424 | 0.3331 | 5.0289 | 0.0896 | 0.0669 | 0.0822 | 0.1342 | 0.1653 | 0.8158 | 0.4508 |
| enhanced_wind_advection_ml | 3 | 152 | 152 | 2.5655 | 3.1257 | 0.4439 | 5.9106 | 0.0830 | 0.0684 | 0.0831 | 0.1274 | 0.1527 | 0.8355 | 0.4534 |
| enhanced_wind_advection_ml | 6 | 152 | 152 | 2.4235 | 2.9824 | 0.3880 | 5.8766 | 0.0786 | 0.0674 | 0.0884 | 0.1268 | 0.1433 | 0.8487 | 0.4389 |
| enhanced_wind_advection_ml | 9 | 151 | 151 | 1.3401 | 1.7768 | -0.2861 | 3.0174 | 0.0517 | 0.0831 | 0.1219 | 0.1601 | 0.1627 | 0.8278 | 0.3426 |
| enhanced_wind_advection_ml | 12 | 152 | 152 | 0.8453 | 1.0166 | 0.6227 | 1.9638 | 0.0274 | 0.2316 | 0.2499 | 0.1096 | 0.0313 | 0.8487 | 0.3526 |
| enhanced_wind_advection_ml | 15 | 152 | 152 | 0.6623 | 0.7609 | 0.5737 | 6.3674 | 0.0164 | 0.0904 | 0.0445 | 0.0014 | 0.0012 | 0.0855 | 0.2694 |
| enhanced_wind_advection_ml | 18 | 152 | 152 | 0.6598 | 0.7610 | 0.6178 | 6.8140 | 0.0143 | 0.0661 | 0.0221 | 0.0014 | 0.0012 | 0.0855 | 0.2875 |

## Rain/CB After Current Max

| model_variant | advection_regime | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | cold_advection | 17 | 16 | 1.4108 | 1.7991 | 0.0193 | 1.1117 | 0.0341 | 0.0292 | 0.0292 | 0.0081 | 0.0072 | 0.5882 | 0.2458 |
| enhanced_intraday_ml | frontal_passage | 9 | 9 | 1.3591 | 1.6130 | 0.8127 | 10.1220 | 0.0245 | 0.1333 | 0.0373 | 0.0583 | 0.0513 | 0.4444 | 0.5653 |
| enhanced_intraday_ml | neutral_or_missing | 759 | 151 | 1.5752 | 2.2639 | 0.3379 | 4.7772 | 0.0521 | 0.0964 | 0.0953 | 0.0944 | 0.0939 | 0.6509 | 0.3619 |
| enhanced_intraday_ml | north_sector | 148 | 71 | 1.7279 | 2.3226 | 0.9746 | 6.4627 | 0.0458 | 0.0924 | 0.1188 | 0.0941 | 0.0981 | 0.4865 | 0.4295 |
| enhanced_intraday_ml | south_sector | 71 | 50 | 1.7780 | 2.3773 | 0.4557 | 3.4461 | 0.0555 | 0.1080 | 0.0969 | 0.1023 | 0.0940 | 0.6338 | 0.3650 |
| enhanced_intraday_ml | warm_advection | 59 | 38 | 1.8298 | 2.4873 | -0.6161 | 3.2250 | 0.0680 | 0.0830 | 0.1065 | 0.0992 | 0.1304 | 0.8136 | 0.3160 |
| enhanced_wind_advection_ml | cold_advection | 17 | 16 | 1.4776 | 1.8765 | -0.0775 | 1.1556 | 0.0377 | 0.0287 | 0.0287 | 0.0092 | 0.0072 | 0.5294 | 0.2288 |
| enhanced_wind_advection_ml | frontal_passage | 9 | 9 | 1.3873 | 1.6504 | 0.8574 | 10.2790 | 0.0254 | 0.1339 | 0.0365 | 0.0815 | 0.0508 | 0.4444 | 0.5840 |
| enhanced_wind_advection_ml | neutral_or_missing | 759 | 151 | 1.5560 | 2.2428 | 0.3379 | 4.9321 | 0.0517 | 0.0980 | 0.0966 | 0.0946 | 0.0931 | 0.6337 | 0.3624 |
| enhanced_wind_advection_ml | north_sector | 148 | 71 | 1.7215 | 2.2958 | 0.9693 | 6.5049 | 0.0458 | 0.0932 | 0.1196 | 0.0944 | 0.0980 | 0.5068 | 0.4332 |
| enhanced_wind_advection_ml | south_sector | 71 | 50 | 1.7675 | 2.3623 | 0.4770 | 3.8953 | 0.0557 | 0.1108 | 0.0985 | 0.1019 | 0.0914 | 0.6056 | 0.3662 |
| enhanced_wind_advection_ml | warm_advection | 59 | 38 | 1.8369 | 2.5033 | -0.5179 | 3.7096 | 0.0674 | 0.0782 | 0.1064 | 0.1081 | 0.1292 | 0.8136 | 0.3356 |
