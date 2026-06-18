# EDDM weather-regime feature comparison

- created: `2026-06-14T14:52:03.680736+00:00`
- target: official DWD daily Tmax; regimes use only as-of METAR/NWP features
- period: `2020-01-01` to `2025-12-30`
- rows: `15299`
- days: `2186`
- recommendation: `do_not_promote_yet`

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 1063 | 152 | 1.6197 | 2.2815 | 0.3804 | 4.8235 | 0.0518 | 0.0951 | 0.0977 | 0.0935 | 0.0947 | 0.6331 | 0.3688 |
| enhanced_weather_regime_ml | 1063 | 152 | 1.6509 | 2.3070 | 0.2316 | 5.2965 | 0.0534 | 0.0944 | 0.0969 | 0.0932 | 0.0946 | 0.6284 | 0.3547 |

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
| enhanced_weather_regime_ml | 0 | 152 | 152 | 2.8474 | 3.4915 | -0.0293 | 5.9504 | 0.0931 | 0.0552 | 0.0735 | 0.1328 | 0.1661 | 0.7895 | 0.4229 |
| enhanced_weather_regime_ml | 3 | 152 | 152 | 2.6842 | 3.2415 | 0.0902 | 6.3240 | 0.0873 | 0.0630 | 0.0781 | 0.1247 | 0.1514 | 0.8092 | 0.4231 |
| enhanced_weather_regime_ml | 6 | 152 | 152 | 2.4564 | 2.9874 | 0.1451 | 6.5448 | 0.0802 | 0.0669 | 0.0882 | 0.1205 | 0.1432 | 0.8355 | 0.4140 |
| enhanced_weather_regime_ml | 9 | 151 | 151 | 1.3880 | 1.8471 | -0.4070 | 3.0680 | 0.0554 | 0.0821 | 0.1185 | 0.1631 | 0.1670 | 0.8146 | 0.3110 |
| enhanced_weather_regime_ml | 12 | 152 | 152 | 0.8510 | 1.0234 | 0.6214 | 1.9785 | 0.0278 | 0.2352 | 0.2505 | 0.1090 | 0.0324 | 0.8487 | 0.3507 |
| enhanced_weather_regime_ml | 15 | 152 | 152 | 0.6649 | 0.7619 | 0.5776 | 6.3611 | 0.0161 | 0.0896 | 0.0463 | 0.0014 | 0.0012 | 0.1776 | 0.2717 |
| enhanced_weather_regime_ml | 18 | 152 | 152 | 0.6629 | 0.7616 | 0.6191 | 6.8335 | 0.0142 | 0.0687 | 0.0230 | 0.0014 | 0.0012 | 0.1250 | 0.2895 |

## By Weather Regime

| model_variant | weather_regime | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | clear_heating | 167 | 62 | 1.4734 | 2.0649 | -0.3285 | 5.3206 | 0.0485 | 0.0830 | 0.0859 | 0.0404 | 0.0500 | 0.4970 | 0.3197 |
| enhanced_intraday_ml | cloud_limited | 504 | 129 | 1.6864 | 2.3097 | 0.7784 | 4.7701 | 0.0507 | 0.0976 | 0.0990 | 0.1094 | 0.1122 | 0.6706 | 0.3880 |
| enhanced_intraday_ml | cold_advection | 43 | 39 | 1.1983 | 1.7604 | 1.1751 | 11.2536 | 0.0153 | 0.0755 | 0.0761 | 0.0476 | 0.0406 | 0.0233 | 0.4944 |
| enhanced_intraday_ml | convective | 126 | 54 | 1.3506 | 1.9204 | 0.7496 | 3.8385 | 0.0446 | 0.1476 | 0.1492 | 0.1312 | 0.1244 | 0.6746 | 0.3927 |
| enhanced_intraday_ml | frontal_rain | 59 | 31 | 1.8461 | 2.9573 | -0.0248 | 4.9522 | 0.0715 | 0.0617 | 0.0756 | 0.0977 | 0.1174 | 0.6780 | 0.3177 |
| enhanced_intraday_ml | heatwave | 17 | 6 | 0.6147 | 0.7970 | 0.5707 | 3.6895 | 0.0217 | 0.1343 | 0.0824 | 0.0246 | 0.0098 | 0.4118 | 0.2980 |
| enhanced_intraday_ml | late_clearing | 147 | 68 | 1.9361 | 2.6180 | -0.5872 | 3.4842 | 0.0719 | 0.0701 | 0.0799 | 0.0868 | 0.0767 | 0.8095 | 0.3306 |
| enhanced_weather_regime_ml | clear_heating | 167 | 62 | 1.4436 | 2.0442 | -0.3454 | 5.3135 | 0.0482 | 0.0860 | 0.0858 | 0.0410 | 0.0485 | 0.5150 | 0.3150 |
| enhanced_weather_regime_ml | cloud_limited | 504 | 129 | 1.7042 | 2.2972 | 0.6194 | 5.4121 | 0.0517 | 0.0952 | 0.0969 | 0.1034 | 0.1109 | 0.6548 | 0.3695 |
| enhanced_weather_regime_ml | cold_advection | 43 | 39 | 1.1456 | 1.5834 | 1.1221 | 11.2071 | 0.0131 | 0.0704 | 0.0667 | 0.0406 | 0.0282 | 0.0465 | 0.4898 |
| enhanced_weather_regime_ml | convective | 126 | 54 | 1.3454 | 1.9098 | 0.6885 | 3.9799 | 0.0458 | 0.1483 | 0.1516 | 0.1465 | 0.1313 | 0.6825 | 0.3890 |
| enhanced_weather_regime_ml | frontal_rain | 59 | 31 | 1.9215 | 2.9776 | -0.1829 | 5.3858 | 0.0711 | 0.0675 | 0.0853 | 0.1073 | 0.1107 | 0.7797 | 0.3320 |
| enhanced_weather_regime_ml | heatwave | 17 | 6 | 0.6186 | 0.7879 | 0.5509 | 3.6533 | 0.0223 | 0.1309 | 0.0785 | 0.0254 | 0.0094 | 0.4118 | 0.2817 |
| enhanced_weather_regime_ml | late_clearing | 147 | 68 | 2.1243 | 2.8558 | -0.9653 | 4.4344 | 0.0801 | 0.0685 | 0.0780 | 0.0893 | 0.0824 | 0.7551 | 0.2979 |

## Regime Counts

| weather_regime | rows |
| --- | --- |
| clear_heating | 5400 |
| cloud_limited | 7267 |
| frontal_rain | 303 |
| convective | 1316 |
| late_clearing | 790 |
| cold_advection | 206 |
| heatwave | 17 |
