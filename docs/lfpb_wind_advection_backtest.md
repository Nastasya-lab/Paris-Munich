# LFPB wind/advection feature backtest

- created: `2026-06-10T13:22:16.775163+00:00`
- period: `2025-07-27` to `2026-05-30`
- rows: `2432`
- days: `304`
- recommendation: `candidate_for_shadow`
- reason: Wind/advection features improved point accuracy over spatial model without material probabilistic regression.

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_intraday | 488 | 61 | 0.7896 | 1.0716 | 0.2628 | 1.1081 | 0.0329 | 0.0741 | 0.0519 | 0.0479 | 0.9836 |
| icon_d2_spatial | 488 | 61 | 0.7727 | 1.0492 | 0.2281 | 1.0898 | 0.0322 | 0.0737 | 0.0511 | 0.0482 | 0.9898 |
| icon_d2_spatial_wind_advection | 488 | 61 | 0.7440 | 1.0139 | 0.2471 | 1.1091 | 0.0317 | 0.0729 | 0.0509 | 0.0484 | 0.9877 |
| persistence_current_metar_max | 488 | 61 | 3.1107 | 4.7486 | -3.1107 | 17.4393 | 0.6311 | 0.6311 | 0.5266 | 0.4303 | 0.3689 |

## By Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_intraday | 6 | 61 | 61 | 1.2117 | 1.5037 | 0.0528 | 1.5995 | 0.0534 | 0.0257 | 0.0311 | 0.0500 | 0.9508 |
| icon_d2_intraday | 8 | 61 | 61 | 1.2457 | 1.5151 | 0.1134 | 1.5747 | 0.0531 | 0.0247 | 0.0351 | 0.0528 | 0.9672 |
| icon_d2_intraday | 10 | 61 | 61 | 1.1826 | 1.4047 | 0.3684 | 1.6060 | 0.0532 | 0.0235 | 0.0501 | 0.0567 | 0.9508 |
| icon_d2_intraday | 12 | 61 | 61 | 0.8915 | 1.1263 | 0.4627 | 1.4865 | 0.0432 | 0.0522 | 0.0859 | 0.1008 | 1.0000 |
| icon_d2_intraday | 14 | 61 | 61 | 0.6175 | 0.7664 | 0.2605 | 1.2115 | 0.0294 | 0.1246 | 0.1477 | 0.1163 | 1.0000 |
| icon_d2_intraday | 16 | 61 | 61 | 0.4729 | 0.5689 | 0.2873 | 0.7625 | 0.0175 | 0.1927 | 0.0495 | 0.0037 | 1.0000 |
| icon_d2_intraday | 18 | 61 | 61 | 0.3380 | 0.4051 | 0.3153 | 0.3608 | 0.0074 | 0.0917 | 0.0116 | 0.0023 | 1.0000 |
| icon_d2_intraday | 20 | 61 | 61 | 0.3570 | 0.5595 | 0.2422 | 0.2633 | 0.0057 | 0.0575 | 0.0045 | 0.0004 | 1.0000 |
| icon_d2_spatial | 6 | 61 | 61 | 1.2048 | 1.4784 | -0.0225 | 1.5594 | 0.0519 | 0.0244 | 0.0295 | 0.0500 | 0.9672 |
| icon_d2_spatial | 8 | 61 | 61 | 1.2190 | 1.4791 | 0.0205 | 1.5710 | 0.0521 | 0.0238 | 0.0319 | 0.0530 | 0.9836 |
| icon_d2_spatial | 10 | 61 | 61 | 1.0914 | 1.3342 | 0.2809 | 1.5443 | 0.0502 | 0.0240 | 0.0490 | 0.0546 | 0.9672 |
| icon_d2_spatial | 12 | 61 | 61 | 0.8849 | 1.1278 | 0.4474 | 1.4474 | 0.0429 | 0.0512 | 0.0840 | 0.1024 | 1.0000 |
| icon_d2_spatial | 14 | 61 | 61 | 0.6199 | 0.7674 | 0.2587 | 1.2074 | 0.0296 | 0.1269 | 0.1471 | 0.1192 | 1.0000 |
| icon_d2_spatial | 16 | 61 | 61 | 0.4737 | 0.5671 | 0.2916 | 0.7684 | 0.0176 | 0.1915 | 0.0516 | 0.0039 | 1.0000 |
| icon_d2_spatial | 18 | 61 | 61 | 0.3295 | 0.3956 | 0.3050 | 0.3567 | 0.0073 | 0.0902 | 0.0114 | 0.0023 | 1.0000 |
| icon_d2_spatial | 20 | 61 | 61 | 0.3583 | 0.5611 | 0.2436 | 0.2640 | 0.0058 | 0.0579 | 0.0046 | 0.0004 | 1.0000 |
| icon_d2_spatial_wind_advection | 6 | 61 | 61 | 1.1557 | 1.4189 | 0.0221 | 1.6242 | 0.0506 | 0.0254 | 0.0295 | 0.0493 | 0.9508 |
| icon_d2_spatial_wind_advection | 8 | 61 | 61 | 1.1717 | 1.4424 | 0.1044 | 1.5917 | 0.0516 | 0.0230 | 0.0341 | 0.0547 | 0.9836 |
| icon_d2_spatial_wind_advection | 10 | 61 | 61 | 1.0257 | 1.2715 | 0.3083 | 1.6137 | 0.0498 | 0.0253 | 0.0490 | 0.0533 | 0.9672 |
| icon_d2_spatial_wind_advection | 12 | 61 | 61 | 0.8259 | 1.0736 | 0.4606 | 1.4438 | 0.0415 | 0.0517 | 0.0813 | 0.0989 | 1.0000 |
| icon_d2_spatial_wind_advection | 14 | 61 | 61 | 0.6241 | 0.7696 | 0.2628 | 1.2534 | 0.0304 | 0.1302 | 0.1494 | 0.1242 | 1.0000 |
| icon_d2_spatial_wind_advection | 16 | 61 | 61 | 0.4663 | 0.5574 | 0.2767 | 0.7403 | 0.0171 | 0.1882 | 0.0479 | 0.0038 | 1.0000 |
| icon_d2_spatial_wind_advection | 18 | 61 | 61 | 0.3199 | 0.3742 | 0.2941 | 0.3350 | 0.0065 | 0.0787 | 0.0113 | 0.0023 | 1.0000 |
| icon_d2_spatial_wind_advection | 20 | 61 | 61 | 0.3628 | 0.5639 | 0.2480 | 0.2705 | 0.0059 | 0.0603 | 0.0046 | 0.0004 | 1.0000 |
| persistence_current_metar_max | 6 | 61 | 61 | 6.8361 | 7.6811 | -6.8361 | 26.7251 | 0.9672 | 0.9672 | 0.9344 | 0.8689 | 0.0328 |
| persistence_current_metar_max | 8 | 61 | 61 | 6.8197 | 7.6694 | -6.8197 | 26.7251 | 0.9672 | 0.9672 | 0.9344 | 0.8689 | 0.0328 |
| persistence_current_metar_max | 10 | 61 | 61 | 5.9016 | 6.6431 | -5.9016 | 26.7251 | 0.9672 | 0.9672 | 0.9180 | 0.8525 | 0.0328 |
| persistence_current_metar_max | 12 | 61 | 61 | 3.3115 | 3.7895 | -3.3115 | 25.3662 | 0.9180 | 0.9180 | 0.8361 | 0.6885 | 0.0820 |
| persistence_current_metar_max | 14 | 61 | 61 | 1.5082 | 1.8554 | -1.5082 | 21.7424 | 0.7869 | 0.7869 | 0.5246 | 0.1639 | 0.2131 |
| persistence_current_metar_max | 16 | 61 | 61 | 0.4590 | 0.7466 | -0.4590 | 11.3242 | 0.4098 | 0.4098 | 0.0492 | 0.0000 | 0.5902 |
| persistence_current_metar_max | 18 | 61 | 61 | 0.0492 | 0.2863 | -0.0492 | 0.9059 | 0.0328 | 0.0328 | 0.0164 | 0.0000 | 0.9672 |
| persistence_current_metar_max | 20 | 61 | 61 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

## By Advection Regime

| model_variant | advection_regime | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_intraday | cold_advection | 4 | 4 | 0.3835 | 0.5878 | 0.3441 | 1.2663 | 0.0344 | 0.0118 | 0.0092 | 0.0142 | 1.0000 |
| icon_d2_intraday | frontal_passage | 14 | 8 | 0.3747 | 0.4645 | 0.2833 | 0.5499 | 0.0116 | 0.0450 | 0.0270 | 0.0083 | 1.0000 |
| icon_d2_intraday | neutral_or_missing | 179 | 48 | 0.7318 | 1.0143 | 0.3917 | 1.0633 | 0.0312 | 0.0797 | 0.0638 | 0.0571 | 0.9832 |
| icon_d2_intraday | north_sector | 184 | 47 | 1.0048 | 1.2886 | 0.1752 | 1.2311 | 0.0393 | 0.0842 | 0.0460 | 0.0456 | 0.9728 |
| icon_d2_intraday | south_sector | 35 | 20 | 0.6769 | 0.9228 | 0.1574 | 0.9322 | 0.0282 | 0.0250 | 0.0246 | 0.0243 | 1.0000 |
| icon_d2_intraday | warm_advection | 72 | 24 | 0.5412 | 0.7117 | 0.2092 | 1.0903 | 0.0270 | 0.0674 | 0.0582 | 0.0518 | 1.0000 |
| icon_d2_spatial | cold_advection | 4 | 4 | 0.4017 | 0.6468 | 0.3870 | 1.0164 | 0.0273 | 0.0097 | 0.0048 | 0.0100 | 1.0000 |
| icon_d2_spatial | frontal_passage | 14 | 8 | 0.3854 | 0.4871 | 0.2393 | 0.5666 | 0.0126 | 0.0436 | 0.0232 | 0.0065 | 1.0000 |
| icon_d2_spatial | neutral_or_missing | 179 | 48 | 0.7185 | 0.9894 | 0.3628 | 1.0372 | 0.0303 | 0.0800 | 0.0624 | 0.0563 | 0.9832 |
| icon_d2_spatial | north_sector | 184 | 47 | 0.9901 | 1.2698 | 0.1525 | 1.2256 | 0.0391 | 0.0828 | 0.0456 | 0.0451 | 0.9891 |
| icon_d2_spatial | south_sector | 35 | 20 | 0.6277 | 0.8515 | 0.0952 | 0.9271 | 0.0263 | 0.0267 | 0.0222 | 0.0263 | 1.0000 |
| icon_d2_spatial | warm_advection | 72 | 24 | 0.5184 | 0.6970 | 0.1404 | 1.0586 | 0.0261 | 0.0672 | 0.0595 | 0.0569 | 1.0000 |
| icon_d2_spatial_wind_advection | cold_advection | 4 | 4 | 0.4007 | 0.6262 | 0.3662 | 1.1663 | 0.0282 | 0.0081 | 0.0062 | 0.0064 | 1.0000 |
| icon_d2_spatial_wind_advection | frontal_passage | 14 | 8 | 0.3499 | 0.4346 | 0.1892 | 0.5491 | 0.0113 | 0.0447 | 0.0208 | 0.0045 | 1.0000 |
| icon_d2_spatial_wind_advection | neutral_or_missing | 179 | 48 | 0.6798 | 0.9447 | 0.3691 | 1.0395 | 0.0294 | 0.0794 | 0.0615 | 0.0553 | 0.9888 |
| icon_d2_spatial_wind_advection | north_sector | 184 | 47 | 0.9585 | 1.2366 | 0.1734 | 1.2318 | 0.0382 | 0.0818 | 0.0454 | 0.0456 | 0.9783 |
| icon_d2_spatial_wind_advection | south_sector | 35 | 20 | 0.5813 | 0.8016 | 0.1301 | 1.0209 | 0.0263 | 0.0256 | 0.0230 | 0.0276 | 1.0000 |
| icon_d2_spatial_wind_advection | warm_advection | 72 | 24 | 0.5304 | 0.6854 | 0.1939 | 1.1170 | 0.0272 | 0.0657 | 0.0605 | 0.0590 | 1.0000 |
| persistence_current_metar_max | cold_advection | 4 | 4 | 2.7500 | 4.0311 | -2.7500 | 13.8155 | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| persistence_current_metar_max | frontal_passage | 14 | 8 | 1.6429 | 3.1736 | -1.6429 | 9.8682 | 0.3571 | 0.3571 | 0.2857 | 0.2143 | 0.6429 |
| persistence_current_metar_max | neutral_or_missing | 179 | 48 | 2.4358 | 3.8418 | -2.4358 | 16.6712 | 0.6034 | 0.6034 | 0.4693 | 0.3687 | 0.3966 |
| persistence_current_metar_max | north_sector | 184 | 47 | 4.0489 | 5.8806 | -4.0489 | 17.8701 | 0.6467 | 0.6467 | 0.5761 | 0.4946 | 0.3533 |
| persistence_current_metar_max | south_sector | 35 | 20 | 3.2857 | 4.7359 | -3.2857 | 16.5786 | 0.6000 | 0.6000 | 0.5714 | 0.5429 | 0.4000 |
| persistence_current_metar_max | warm_advection | 72 | 24 | 2.6111 | 3.7417 | -2.6111 | 20.3395 | 0.7361 | 0.7361 | 0.5694 | 0.4028 | 0.2639 |

## By Station Availability

| model_variant | adv_available_station_count | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| icon_d2_intraday | 3 | 488 | 61 | 0.7896 | 1.0716 | 0.2628 | 1.1081 | 0.0329 | 0.0741 | 0.0519 | 0.0479 | 0.9836 |
| icon_d2_spatial | 3 | 488 | 61 | 0.7727 | 1.0492 | 0.2281 | 1.0898 | 0.0322 | 0.0737 | 0.0511 | 0.0482 | 0.9898 |
| icon_d2_spatial_wind_advection | 3 | 488 | 61 | 0.7440 | 1.0139 | 0.2471 | 1.1091 | 0.0317 | 0.0729 | 0.0509 | 0.0484 | 0.9877 |
| persistence_current_metar_max | 3 | 488 | 61 | 3.1107 | 4.7486 | -3.1107 | 17.4393 | 0.6311 | 0.6311 | 0.5266 | 0.4303 | 0.3689 |
