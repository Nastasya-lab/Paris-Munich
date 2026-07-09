# EHAM ICON-D2 METAR Tmax model

Forecast-as-issued ICON-D2 candidate for daily maximum temperature reported by METAR.

- model version: `eham_metar_tmax_icon_d2_v1`
- target period: `2026-06-01` to `2026-07-07`
- usable rows: `296`
- days joined: `37`
- promotion: `production_artifact_updated`

## Holdout Overall

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eham_icon_d2_ensemble_candidate | 64 | 8 | 0.7035 | 0.8806 | 0.0579 | 1.1396 | 0.0286 | 0.0833 | 0.1021 | 0.0588 | 0.9062 |
| eham_icon_d2_ml_calibrated | 64 | 8 | 0.6987 | 0.8845 | 0.0372 | 2.3082 | 0.0324 | 0.0798 | 0.1026 | 0.0628 | 0.8906 |
| eham_metar_only_calibrated | 64 | 8 | 1.0710 | 1.3612 | 0.6423 | 3.3599 | 0.0509 | 0.0883 | 0.1382 | 0.1304 | 0.9688 |
| persistence_current_metar_max | 64 | 8 | 1.6406 | 2.6250 | -1.6406 | 14.6790 | 0.5312 | 0.5312 | 0.4062 | 0.2812 | 0.4688 |
| raw_icon_d2_residual_distribution | 64 | 8 | 0.7529 | 0.9419 | 0.1201 | 1.1740 | 0.0757 | 0.1380 | 0.1100 | 0.0529 | 0.9062 |

## By Local Issue Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eham_icon_d2_ensemble_candidate | 6 | 8 | 8 | 0.7248 | 0.9875 | -0.3499 | 1.7294 | 0.0436 | 0.0020 | 0.0266 | 0.0167 | 0.8750 |
| eham_icon_d2_ensemble_candidate | 8 | 8 | 8 | 0.9415 | 1.1473 | -0.0836 | 1.7083 | 0.0468 | 0.0021 | 0.0352 | 0.0981 | 0.8750 |
| eham_icon_d2_ensemble_candidate | 10 | 8 | 8 | 0.7891 | 0.9668 | -0.0210 | 1.6154 | 0.0369 | 0.0359 | 0.1119 | 0.0882 | 0.8750 |
| eham_icon_d2_ensemble_candidate | 12 | 8 | 8 | 0.9132 | 1.0293 | -0.0014 | 1.6414 | 0.0414 | 0.1279 | 0.3281 | 0.1475 | 0.7500 |
| eham_icon_d2_ensemble_candidate | 14 | 8 | 8 | 0.9652 | 1.0649 | 0.1685 | 1.2324 | 0.0356 | 0.2406 | 0.1901 | 0.1127 | 0.8750 |
| eham_icon_d2_ensemble_candidate | 16 | 8 | 8 | 0.6099 | 0.7265 | 0.0664 | 0.8322 | 0.0195 | 0.1965 | 0.1124 | 0.0037 | 1.0000 |
| eham_icon_d2_ensemble_candidate | 18 | 8 | 8 | 0.3436 | 0.3589 | 0.3436 | 0.1814 | 0.0027 | 0.0311 | 0.0061 | 0.0017 | 1.0000 |
| eham_icon_d2_ensemble_candidate | 20 | 8 | 8 | 0.3405 | 0.3563 | 0.3405 | 0.1760 | 0.0027 | 0.0300 | 0.0061 | 0.0018 | 1.0000 |
| eham_icon_d2_ml_calibrated | 6 | 8 | 8 | 0.7655 | 1.0222 | -0.3931 | 4.7763 | 0.0472 | 0.0026 | 0.0242 | 0.0161 | 0.8750 |
| eham_icon_d2_ml_calibrated | 8 | 8 | 8 | 0.9657 | 1.1792 | -0.0552 | 4.6533 | 0.0520 | 0.0026 | 0.0314 | 0.1086 | 0.8750 |
| eham_icon_d2_ml_calibrated | 10 | 8 | 8 | 0.7872 | 0.9586 | -0.0035 | 1.8115 | 0.0409 | 0.0398 | 0.1095 | 0.0939 | 0.8750 |
| eham_icon_d2_ml_calibrated | 12 | 8 | 8 | 0.9332 | 1.0376 | 0.0076 | 1.7356 | 0.0488 | 0.1389 | 0.3341 | 0.1551 | 0.7500 |
| eham_icon_d2_ml_calibrated | 14 | 8 | 8 | 0.9728 | 1.0734 | 0.1456 | 1.3552 | 0.0428 | 0.2359 | 0.1996 | 0.1192 | 0.8750 |
| eham_icon_d2_ml_calibrated | 16 | 8 | 8 | 0.5389 | 0.6891 | -0.0304 | 3.9314 | 0.0242 | 0.1988 | 0.1111 | 0.0037 | 0.8750 |
| eham_icon_d2_ml_calibrated | 18 | 8 | 8 | 0.3152 | 0.3191 | 0.3152 | 0.1045 | 0.0016 | 0.0105 | 0.0054 | 0.0028 | 1.0000 |
| eham_icon_d2_ml_calibrated | 20 | 8 | 8 | 0.3111 | 0.3149 | 0.3111 | 0.0974 | 0.0015 | 0.0092 | 0.0053 | 0.0031 | 1.0000 |
| eham_metar_only_calibrated | 6 | 8 | 8 | 1.5295 | 1.8317 | 0.3983 | 2.3839 | 0.0868 | 0.0056 | 0.1082 | 0.1091 | 1.0000 |
| eham_metar_only_calibrated | 8 | 8 | 8 | 1.6744 | 1.9402 | 1.2635 | 5.9767 | 0.0938 | 0.0103 | 0.1130 | 0.1797 | 1.0000 |
| eham_metar_only_calibrated | 10 | 8 | 8 | 1.4904 | 1.7232 | 1.2025 | 11.2793 | 0.0803 | 0.0167 | 0.2346 | 0.2794 | 1.0000 |
| eham_metar_only_calibrated | 12 | 8 | 8 | 1.2553 | 1.5136 | 0.9201 | 2.0638 | 0.0699 | 0.1927 | 0.2785 | 0.3452 | 1.0000 |
| eham_metar_only_calibrated | 14 | 8 | 8 | 1.2236 | 1.2709 | 0.5058 | 0.9975 | 0.0467 | 0.2539 | 0.2331 | 0.1178 | 0.8750 |
| eham_metar_only_calibrated | 16 | 8 | 8 | 0.5762 | 0.6985 | 0.0303 | 3.9374 | 0.0248 | 0.2008 | 0.1155 | 0.0037 | 0.8750 |
| eham_metar_only_calibrated | 18 | 8 | 8 | 0.4041 | 0.4064 | 0.4041 | 0.1208 | 0.0024 | 0.0131 | 0.0114 | 0.0039 | 1.0000 |
| eham_metar_only_calibrated | 20 | 8 | 8 | 0.4141 | 0.4169 | 0.4141 | 0.1199 | 0.0024 | 0.0130 | 0.0113 | 0.0044 | 1.0000 |
| persistence_current_metar_max | 6 | 8 | 8 | 4.8750 | 5.2559 | -4.8750 | 27.6310 | 1.0000 | 1.0000 | 0.8750 | 0.8750 | 0.0000 |
| persistence_current_metar_max | 8 | 8 | 8 | 3.6250 | 3.9528 | -3.6250 | 27.6310 | 1.0000 | 1.0000 | 0.8750 | 0.7500 | 0.0000 |
| persistence_current_metar_max | 10 | 8 | 8 | 2.2500 | 2.5981 | -2.2500 | 27.6310 | 1.0000 | 1.0000 | 0.6250 | 0.3750 | 0.0000 |
| persistence_current_metar_max | 12 | 8 | 8 | 1.3750 | 1.6956 | -1.3750 | 20.7233 | 0.7500 | 0.7500 | 0.5000 | 0.1250 | 0.2500 |
| persistence_current_metar_max | 14 | 8 | 8 | 0.6250 | 1.2748 | -0.6250 | 6.9078 | 0.2500 | 0.2500 | 0.2500 | 0.1250 | 0.7500 |
| persistence_current_metar_max | 16 | 8 | 8 | 0.3750 | 0.7906 | -0.3750 | 6.9078 | 0.2500 | 0.2500 | 0.1250 | 0.0000 | 0.7500 |
| persistence_current_metar_max | 18 | 8 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| persistence_current_metar_max | 20 | 8 | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| raw_icon_d2_residual_distribution | 6 | 8 | 8 | 0.8832 | 1.0926 | -0.2203 | 1.5087 | 0.0872 | 0.0016 | 0.0366 | 0.0243 | 0.8750 |
| raw_icon_d2_residual_distribution | 8 | 8 | 8 | 0.8690 | 1.0733 | -0.1689 | 1.4955 | 0.0859 | 0.0017 | 0.0497 | 0.0748 | 0.8750 |
| raw_icon_d2_residual_distribution | 10 | 8 | 8 | 0.7951 | 1.0085 | -0.0735 | 1.4537 | 0.0818 | 0.0286 | 0.1312 | 0.0747 | 0.8750 |
| raw_icon_d2_residual_distribution | 12 | 8 | 8 | 0.8531 | 1.0300 | -0.0283 | 1.4774 | 0.0915 | 0.1039 | 0.3153 | 0.1372 | 0.8750 |
| raw_icon_d2_residual_distribution | 14 | 8 | 8 | 0.9424 | 1.0699 | 0.2373 | 1.2504 | 0.0873 | 0.2663 | 0.1753 | 0.1019 | 0.8750 |
| raw_icon_d2_residual_distribution | 16 | 8 | 8 | 0.8228 | 0.9546 | 0.3569 | 1.1530 | 0.0762 | 0.3241 | 0.1336 | 0.0104 | 0.8750 |
| raw_icon_d2_residual_distribution | 18 | 8 | 8 | 0.4287 | 0.5557 | 0.4287 | 0.5266 | 0.0478 | 0.1889 | 0.0191 | 0.0002 | 1.0000 |
| raw_icon_d2_residual_distribution | 20 | 8 | 8 | 0.4287 | 0.5557 | 0.4287 | 0.5266 | 0.0478 | 0.1889 | 0.0191 | 0.0002 | 1.0000 |

## By Season

| model_variant | season | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eham_icon_d2_ensemble_candidate | summer | 64 | 8 | 0.7035 | 0.8806 | 0.0579 | 1.1396 | 0.0286 | 0.0833 | 0.1021 | 0.0588 | 0.9062 |
| eham_icon_d2_ml_calibrated | summer | 64 | 8 | 0.6987 | 0.8845 | 0.0372 | 2.3082 | 0.0324 | 0.0798 | 0.1026 | 0.0628 | 0.8906 |
| eham_metar_only_calibrated | summer | 64 | 8 | 1.0710 | 1.3612 | 0.6423 | 3.3599 | 0.0509 | 0.0883 | 0.1382 | 0.1304 | 0.9688 |
| persistence_current_metar_max | summer | 64 | 8 | 1.6406 | 2.6250 | -1.6406 | 14.6790 | 0.5312 | 0.5312 | 0.4062 | 0.2812 | 0.4688 |
| raw_icon_d2_residual_distribution | summer | 64 | 8 | 0.7529 | 0.9419 | 0.1201 | 1.1740 | 0.0757 | 0.1380 | 0.1100 | 0.0529 | 0.9062 |

## Limitations

- Target is METAR Tmax, not official Meteo-France TX.
- TAF is not used because the IEM historical TAF archive returned zero LFPB rows.
- The model is trained on the currently available forecast-as-issued ICON-D2 overlap window.
- Enhanced intraday features are computed from as-of METAR only; live quality depends on AWC METAR parser coverage.
