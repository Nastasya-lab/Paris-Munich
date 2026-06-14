# EDDM spatial METAR feature comparison

- created: `2026-06-14T16:16:49.822941+00:00`
- target: official DWD daily Tmax; spatial features use only as-of neighbor METAR
- period: `2020-01-01` to `2025-12-30`
- rows: `15299`
- days: `2186`
- recommendation: `promote_to_main_model`
- neighbors: `EDMO, EDMA, ETSI, ETSL`

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 3065 | 438 | 1.6214 | 2.4115 | -0.0226 | 5.0635 | 0.0788 | 0.0952 | 0.0954 | 0.1017 | 0.0993 | 0.5997 | 0.3519 |
| enhanced_spatial_metar_ml | 3065 | 438 | 1.5118 | 2.2456 | -0.0512 | 5.0870 | 0.0733 | 0.0912 | 0.0895 | 0.0964 | 0.0904 | 0.6072 | 0.3424 |

## By UTC Issue Hour

| model_variant | issue_hour_utc | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 0 | 438 | 438 | 2.9186 | 3.6354 | -0.4434 | 7.2840 | 0.1292 | 0.0580 | 0.0903 | 0.1352 | 0.1604 | 0.7489 | 0.4178 |
| enhanced_intraday_ml | 3 | 438 | 438 | 2.7796 | 3.4787 | -0.3293 | 7.1832 | 0.1210 | 0.0563 | 0.0923 | 0.1398 | 0.1583 | 0.7717 | 0.4210 |
| enhanced_intraday_ml | 6 | 438 | 438 | 2.5174 | 3.1928 | -0.0368 | 5.5297 | 0.1180 | 0.0581 | 0.0958 | 0.1396 | 0.1557 | 0.8333 | 0.4358 |
| enhanced_intraday_ml | 9 | 437 | 437 | 1.4809 | 1.9247 | -0.3147 | 2.3945 | 0.0848 | 0.0715 | 0.1225 | 0.1696 | 0.1828 | 0.8192 | 0.3742 |
| enhanced_intraday_ml | 12 | 438 | 438 | 0.7566 | 0.9351 | 0.3896 | 2.0208 | 0.0455 | 0.2190 | 0.2311 | 0.1266 | 0.0371 | 0.8607 | 0.3441 |
| enhanced_intraday_ml | 15 | 438 | 438 | 0.4408 | 0.5447 | 0.2479 | 4.9980 | 0.0284 | 0.1176 | 0.0262 | 0.0007 | 0.0005 | 0.0822 | 0.2165 |
| enhanced_intraday_ml | 18 | 438 | 438 | 0.4558 | 0.5624 | 0.3278 | 6.0284 | 0.0243 | 0.0858 | 0.0098 | 0.0007 | 0.0005 | 0.0822 | 0.2537 |
| enhanced_spatial_metar_ml | 0 | 438 | 438 | 2.6798 | 3.3824 | -0.4728 | 7.3623 | 0.1179 | 0.0559 | 0.0891 | 0.1277 | 0.1403 | 0.7694 | 0.4077 |
| enhanced_spatial_metar_ml | 3 | 438 | 438 | 2.5833 | 3.2291 | -0.3859 | 7.3311 | 0.1112 | 0.0556 | 0.0899 | 0.1302 | 0.1410 | 0.7900 | 0.4089 |
| enhanced_spatial_metar_ml | 6 | 438 | 438 | 2.3218 | 2.9655 | -0.1248 | 5.6299 | 0.1083 | 0.0606 | 0.0989 | 0.1346 | 0.1436 | 0.8493 | 0.4273 |
| enhanced_spatial_metar_ml | 9 | 437 | 437 | 1.4025 | 1.8095 | -0.3259 | 2.2119 | 0.0799 | 0.0711 | 0.1203 | 0.1670 | 0.1701 | 0.8169 | 0.3641 |
| enhanced_spatial_metar_ml | 12 | 438 | 438 | 0.6976 | 0.8626 | 0.3734 | 2.0397 | 0.0428 | 0.1913 | 0.1925 | 0.1142 | 0.0371 | 0.8607 | 0.3188 |
| enhanced_spatial_metar_ml | 15 | 438 | 438 | 0.4410 | 0.5449 | 0.2485 | 4.9986 | 0.0284 | 0.1176 | 0.0261 | 0.0007 | 0.0005 | 0.0822 | 0.2162 |
| enhanced_spatial_metar_ml | 18 | 438 | 438 | 0.4560 | 0.5627 | 0.3283 | 6.0288 | 0.0244 | 0.0858 | 0.0098 | 0.0007 | 0.0005 | 0.0822 | 0.2535 |

## By Neighbor Availability

| model_variant | spatial_available_station_count | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_peak_already_passed | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 | mean_false_upside_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enhanced_intraday_ml | 0 | 1 | 1 | 10.1883 | 10.1883 | -10.1883 | 27.6310 | 0.3798 | 0.0061 | 0.0061 | 0.0581 | 0.0767 | 0.0000 | 0.0000 |
| enhanced_intraday_ml | 3 | 227 | 37 | 1.5364 | 2.2095 | -0.0467 | 3.9993 | 0.0860 | 0.1057 | 0.1027 | 0.1011 | 0.1033 | 0.6388 | 0.3302 |
| enhanced_intraday_ml | 4 | 2837 | 409 | 1.6252 | 2.4198 | -0.0171 | 5.1407 | 0.0781 | 0.0944 | 0.0949 | 0.1018 | 0.0990 | 0.5968 | 0.3537 |
| enhanced_spatial_metar_ml | 0 | 1 | 1 | 12.2521 | 12.2521 | -12.2521 | 27.6310 | 0.5633 | 0.0355 | 0.0355 | 0.1400 | 0.3012 | 0.0000 | 0.0000 |
| enhanced_spatial_metar_ml | 3 | 227 | 37 | 1.4105 | 2.0358 | -0.0372 | 4.2776 | 0.0805 | 0.1058 | 0.1033 | 0.1033 | 0.0895 | 0.6696 | 0.3315 |
| enhanced_spatial_metar_ml | 4 | 2837 | 409 | 1.5161 | 2.2502 | -0.0480 | 5.1438 | 0.0725 | 0.0900 | 0.0884 | 0.0958 | 0.0904 | 0.6024 | 0.3433 |
