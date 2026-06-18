# LFPB ICON-D2 METAR Tmax model

Forecast-as-issued ICON-D2 candidate for daily maximum temperature reported by METAR.

- model version: `lfpb_metar_tmax_icon_d2_v1`
- target period: `2025-07-27` to `2026-05-30`
- usable rows: `2432`
- days joined: `304`
- promotion: `production_artifact_updated`

## Holdout Overall

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfpb_icon_d2_ensemble_candidate | 488 | 61 | 0.5338 | 0.7794 | 0.2601 | 1.2430 | 0.0288 | 0.0504 | 0.0462 | 0.0558 | 0.9037 |
| lfpb_icon_d2_ml_calibrated | 488 | 61 | 0.5312 | 0.7897 | 0.2515 | 3.4233 | 0.0303 | 0.0499 | 0.0464 | 0.0564 | 0.8934 |
| lfpb_metar_only_calibrated | 488 | 61 | 1.3865 | 2.1499 | 0.5167 | 4.0267 | 0.0785 | 0.0747 | 0.0805 | 0.0934 | 0.8504 |
| persistence_current_metar_max | 488 | 61 | 3.1107 | 4.7486 | -3.1107 | 17.4393 | 0.6311 | 0.6311 | 0.5266 | 0.4303 | 0.3689 |
| raw_icon_d2_residual_distribution | 488 | 61 | 0.6452 | 0.8237 | 0.4234 | 1.1067 | 0.0552 | 0.1136 | 0.0518 | 0.0481 | 0.9672 |

## By Local Issue Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfpb_icon_d2_ensemble_candidate | 6 | 61 | 61 | 0.7787 | 1.0210 | 0.1430 | 2.0221 | 0.0462 | 0.0092 | 0.0198 | 0.0506 | 0.8852 |
| lfpb_icon_d2_ensemble_candidate | 8 | 61 | 61 | 0.7474 | 0.9884 | 0.1666 | 2.0331 | 0.0442 | 0.0089 | 0.0185 | 0.0657 | 0.8361 |
| lfpb_icon_d2_ensemble_candidate | 10 | 61 | 61 | 0.7296 | 0.9180 | 0.3226 | 1.8724 | 0.0430 | 0.0086 | 0.0336 | 0.0614 | 0.8689 |
| lfpb_icon_d2_ensemble_candidate | 12 | 61 | 61 | 0.6897 | 0.8900 | 0.4344 | 1.6400 | 0.0397 | 0.0390 | 0.0854 | 0.0888 | 0.8361 |
| lfpb_icon_d2_ensemble_candidate | 14 | 61 | 61 | 0.6543 | 0.8640 | 0.5021 | 1.2772 | 0.0332 | 0.1096 | 0.1193 | 0.1616 | 0.8361 |
| lfpb_icon_d2_ensemble_candidate | 16 | 61 | 61 | 0.4418 | 0.5912 | 0.3150 | 0.8067 | 0.0183 | 0.1656 | 0.0771 | 0.0132 | 0.9672 |
| lfpb_icon_d2_ensemble_candidate | 18 | 61 | 61 | 0.1903 | 0.3254 | 0.1698 | 0.2677 | 0.0058 | 0.0614 | 0.0156 | 0.0047 | 1.0000 |
| lfpb_icon_d2_ensemble_candidate | 20 | 61 | 61 | 0.0388 | 0.0596 | 0.0274 | 0.0249 | 0.0001 | 0.0009 | 0.0000 | 0.0000 | 1.0000 |
| lfpb_icon_d2_ml_calibrated | 6 | 61 | 61 | 0.7923 | 1.0434 | 0.1393 | 7.4479 | 0.0477 | 0.0088 | 0.0200 | 0.0508 | 0.8852 |
| lfpb_icon_d2_ml_calibrated | 8 | 61 | 61 | 0.7586 | 1.0102 | 0.1569 | 7.0665 | 0.0465 | 0.0087 | 0.0185 | 0.0668 | 0.8361 |
| lfpb_icon_d2_ml_calibrated | 10 | 61 | 61 | 0.7361 | 0.9306 | 0.3197 | 6.1284 | 0.0453 | 0.0085 | 0.0338 | 0.0624 | 0.8525 |
| lfpb_icon_d2_ml_calibrated | 12 | 61 | 61 | 0.6971 | 0.9017 | 0.4419 | 2.8353 | 0.0418 | 0.0396 | 0.0866 | 0.0900 | 0.8033 |
| lfpb_icon_d2_ml_calibrated | 14 | 61 | 61 | 0.6573 | 0.8698 | 0.5007 | 1.6699 | 0.0355 | 0.1106 | 0.1197 | 0.1634 | 0.8361 |
| lfpb_icon_d2_ml_calibrated | 16 | 61 | 61 | 0.4334 | 0.5863 | 0.3023 | 1.6033 | 0.0197 | 0.1643 | 0.0772 | 0.0135 | 0.9344 |
| lfpb_icon_d2_ml_calibrated | 18 | 61 | 61 | 0.1722 | 0.3165 | 0.1483 | 0.6324 | 0.0061 | 0.0588 | 0.0157 | 0.0045 | 1.0000 |
| lfpb_icon_d2_ml_calibrated | 20 | 61 | 61 | 0.0030 | 0.0164 | 0.0030 | 0.0031 | 0.0000 | 0.0003 | 0.0000 | 0.0000 | 1.0000 |
| lfpb_metar_only_calibrated | 6 | 61 | 61 | 2.6678 | 3.2486 | 0.4950 | 8.7760 | 0.1518 | 0.0367 | 0.0567 | 0.1074 | 0.6885 |
| lfpb_metar_only_calibrated | 8 | 61 | 61 | 2.6618 | 3.2412 | 0.4656 | 7.9424 | 0.1518 | 0.0377 | 0.0583 | 0.1035 | 0.6721 |
| lfpb_metar_only_calibrated | 10 | 61 | 61 | 2.3897 | 2.9471 | 1.1005 | 7.1344 | 0.1378 | 0.0349 | 0.0730 | 0.1139 | 0.6885 |
| lfpb_metar_only_calibrated | 12 | 61 | 61 | 1.9550 | 2.3609 | 1.6893 | 4.8756 | 0.1119 | 0.0802 | 0.1393 | 0.2213 | 0.8525 |
| lfpb_metar_only_calibrated | 14 | 61 | 61 | 0.8621 | 1.1077 | 0.4592 | 2.0570 | 0.0486 | 0.1409 | 0.2462 | 0.2004 | 0.9836 |
| lfpb_metar_only_calibrated | 16 | 61 | 61 | 0.4593 | 0.5899 | -0.0781 | 0.8783 | 0.0222 | 0.2347 | 0.0532 | 0.0004 | 0.9508 |
| lfpb_metar_only_calibrated | 18 | 61 | 61 | 0.0818 | 0.2844 | -0.0120 | 0.5399 | 0.0037 | 0.0317 | 0.0168 | 0.0000 | 0.9672 |
| lfpb_metar_only_calibrated | 20 | 61 | 61 | 0.0142 | 0.0484 | 0.0142 | 0.0103 | 0.0001 | 0.0010 | 0.0001 | 0.0000 | 1.0000 |
| persistence_current_metar_max | 6 | 61 | 61 | 6.8361 | 7.6811 | -6.8361 | 26.7251 | 0.9672 | 0.9672 | 0.9344 | 0.8689 | 0.0328 |
| persistence_current_metar_max | 8 | 61 | 61 | 6.8197 | 7.6694 | -6.8197 | 26.7251 | 0.9672 | 0.9672 | 0.9344 | 0.8689 | 0.0328 |
| persistence_current_metar_max | 10 | 61 | 61 | 5.9016 | 6.6431 | -5.9016 | 26.7251 | 0.9672 | 0.9672 | 0.9180 | 0.8525 | 0.0328 |
| persistence_current_metar_max | 12 | 61 | 61 | 3.3115 | 3.7895 | -3.3115 | 25.3662 | 0.9180 | 0.9180 | 0.8361 | 0.6885 | 0.0820 |
| persistence_current_metar_max | 14 | 61 | 61 | 1.5082 | 1.8554 | -1.5082 | 21.7424 | 0.7869 | 0.7869 | 0.5246 | 0.1639 | 0.2131 |
| persistence_current_metar_max | 16 | 61 | 61 | 0.4590 | 0.7466 | -0.4590 | 11.3242 | 0.4098 | 0.4098 | 0.0492 | 0.0000 | 0.5902 |
| persistence_current_metar_max | 18 | 61 | 61 | 0.0492 | 0.2863 | -0.0492 | 0.9059 | 0.0328 | 0.0328 | 0.0164 | 0.0000 | 0.9672 |
| persistence_current_metar_max | 20 | 61 | 61 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| raw_icon_d2_residual_distribution | 6 | 61 | 61 | 0.6125 | 0.7583 | 0.2139 | 1.2149 | 0.0713 | 0.0196 | 0.0176 | 0.0493 | 0.9836 |
| raw_icon_d2_residual_distribution | 8 | 61 | 61 | 0.6599 | 0.7907 | 0.3505 | 1.2460 | 0.0637 | 0.0161 | 0.0201 | 0.0512 | 0.9836 |
| raw_icon_d2_residual_distribution | 10 | 61 | 61 | 0.6688 | 0.8003 | 0.3795 | 1.2379 | 0.0561 | 0.0159 | 0.0329 | 0.0486 | 0.9836 |
| raw_icon_d2_residual_distribution | 12 | 61 | 61 | 0.6386 | 0.7759 | 0.2904 | 1.2077 | 0.0548 | 0.0388 | 0.0711 | 0.0756 | 0.9836 |
| raw_icon_d2_residual_distribution | 14 | 61 | 61 | 0.6612 | 0.8279 | 0.5293 | 1.1693 | 0.0566 | 0.1156 | 0.1346 | 0.1375 | 0.9508 |
| raw_icon_d2_residual_distribution | 16 | 61 | 61 | 0.6231 | 0.7772 | 0.5558 | 0.9786 | 0.0518 | 0.2574 | 0.0936 | 0.0112 | 0.9508 |
| raw_icon_d2_residual_distribution | 18 | 61 | 61 | 0.5774 | 0.6694 | 0.5774 | 0.7059 | 0.0378 | 0.2304 | 0.0251 | 0.0088 | 0.9508 |
| raw_icon_d2_residual_distribution | 20 | 61 | 61 | 0.7202 | 1.1171 | 0.4907 | 1.0936 | 0.0496 | 0.2147 | 0.0193 | 0.0024 | 0.9508 |

## By Season

| model_variant | season | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfpb_icon_d2_ensemble_candidate | spring | 488 | 61 | 0.5338 | 0.7794 | 0.2601 | 1.2430 | 0.0288 | 0.0504 | 0.0462 | 0.0558 | 0.9037 |
| lfpb_icon_d2_ml_calibrated | spring | 488 | 61 | 0.5312 | 0.7897 | 0.2515 | 3.4233 | 0.0303 | 0.0499 | 0.0464 | 0.0564 | 0.8934 |
| lfpb_metar_only_calibrated | spring | 488 | 61 | 1.3865 | 2.1499 | 0.5167 | 4.0267 | 0.0785 | 0.0747 | 0.0805 | 0.0934 | 0.8504 |
| persistence_current_metar_max | spring | 488 | 61 | 3.1107 | 4.7486 | -3.1107 | 17.4393 | 0.6311 | 0.6311 | 0.5266 | 0.4303 | 0.3689 |
| raw_icon_d2_residual_distribution | spring | 488 | 61 | 0.6452 | 0.8237 | 0.4234 | 1.1067 | 0.0552 | 0.1136 | 0.0518 | 0.0481 | 0.9672 |

## Limitations

- Target is METAR Tmax, not official Meteo-France TX.
- TAF is not used because the IEM historical TAF archive returned zero LFPB rows.
- The model is trained on the currently available forecast-as-issued ICON-D2 overlap window.
- Enhanced intraday features are computed from as-of METAR only; live quality depends on AWC METAR parser coverage.
