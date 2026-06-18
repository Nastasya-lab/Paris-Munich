# LFPB ICON-D2 NWP stress test

- rows joined: `2432`
- days joined: `304`
- target period: `2025-07-27` to `2026-05-30`
- split: `{'train_start': '2025-07-27', 'train_end': '2026-02-12', 'test_start': '2026-02-13', 'test_end': '2026-05-30', 'train_rows': 1576, 'test_rows': 856, 'train_days': 197, 'test_days': 107}`

## Overall

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nwp_aware_ml_uncalibrated | 856 | 107 | 0.7108 | 1.0354 | -0.1530 | 3.1992 | 0.0315 | 0.9708 |
| persistence_current_metar_max | 856 | 107 | 2.9217 | 4.5201 | -2.9217 | 16.9143 | 0.6121 | 0.3879 |
| raw_icon_d2_point | 856 | 107 | 0.6133 | 0.9466 | -0.2418 | 14.2029 | 0.3353 | 0.4860 |
| raw_icon_d2_residual_distribution | 856 | 107 | 0.6295 | 0.9064 | 0.2636 | 1.1911 | 0.0619 | 0.9661 |

## By local issue hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nwp_aware_ml_uncalibrated | 6 | 107 | 107 | 1.2072 | 1.6179 | -0.9435 | 7.8054 | 0.0595 | 0.9346 |
| nwp_aware_ml_uncalibrated | 8 | 107 | 107 | 1.2129 | 1.6082 | -0.9118 | 8.0052 | 0.0589 | 0.9439 |
| nwp_aware_ml_uncalibrated | 10 | 107 | 107 | 0.9543 | 1.2705 | -0.5966 | 4.8077 | 0.0459 | 0.9720 |
| nwp_aware_ml_uncalibrated | 12 | 107 | 107 | 0.6631 | 0.8218 | 0.0005 | 2.6556 | 0.0368 | 0.9533 |
| nwp_aware_ml_uncalibrated | 14 | 107 | 107 | 0.5824 | 0.7432 | 0.3306 | 1.1487 | 0.0290 | 0.9626 |
| nwp_aware_ml_uncalibrated | 16 | 107 | 107 | 0.4617 | 0.5538 | 0.3361 | 0.8440 | 0.0150 | 1.0000 |
| nwp_aware_ml_uncalibrated | 18 | 107 | 107 | 0.3623 | 0.4054 | 0.3234 | 0.2341 | 0.0052 | 1.0000 |
| nwp_aware_ml_uncalibrated | 20 | 107 | 107 | 0.2425 | 0.2438 | 0.2376 | 0.0932 | 0.0013 | 1.0000 |
| persistence_current_metar_max | 6 | 107 | 107 | 6.3925 | 7.2795 | -6.3925 | 26.5981 | 0.9626 | 0.0374 |
| persistence_current_metar_max | 8 | 107 | 107 | 6.3645 | 7.2467 | -6.3645 | 26.5981 | 0.9626 | 0.0374 |
| persistence_current_metar_max | 10 | 107 | 107 | 5.5701 | 6.3334 | -5.5701 | 26.3399 | 0.9533 | 0.0467 |
| persistence_current_metar_max | 12 | 107 | 107 | 3.1121 | 3.6004 | -3.1121 | 25.5652 | 0.9252 | 0.0748 |
| persistence_current_metar_max | 14 | 107 | 107 | 1.3645 | 1.8444 | -1.3645 | 20.6587 | 0.7477 | 0.2523 |
| persistence_current_metar_max | 16 | 107 | 107 | 0.4019 | 0.9323 | -0.4019 | 8.0052 | 0.2897 | 0.7103 |
| persistence_current_metar_max | 18 | 107 | 107 | 0.1121 | 0.6557 | -0.1121 | 1.0329 | 0.0374 | 0.9626 |
| persistence_current_metar_max | 20 | 107 | 107 | 0.0561 | 0.4102 | -0.0561 | 0.5165 | 0.0187 | 0.9813 |
| raw_icon_d2_point | 6 | 107 | 107 | 0.6168 | 0.8538 | -0.2243 | 15.4940 | 0.3738 | 0.4393 |
| raw_icon_d2_point | 8 | 107 | 107 | 0.6822 | 0.9521 | -0.2710 | 16.0105 | 0.3832 | 0.4206 |
| raw_icon_d2_point | 10 | 107 | 107 | 0.6916 | 1.0047 | -0.2617 | 15.7523 | 0.3645 | 0.4299 |
| raw_icon_d2_point | 12 | 107 | 107 | 0.6916 | 1.0047 | -0.3738 | 15.7523 | 0.4112 | 0.4299 |
| raw_icon_d2_point | 14 | 107 | 107 | 0.5794 | 0.9171 | -0.1121 | 13.4282 | 0.2710 | 0.5140 |
| raw_icon_d2_point | 16 | 107 | 107 | 0.5047 | 0.8428 | -0.1121 | 12.1370 | 0.2523 | 0.5607 |
| raw_icon_d2_point | 18 | 107 | 107 | 0.4953 | 0.8260 | -0.1589 | 12.1370 | 0.2710 | 0.5607 |
| raw_icon_d2_point | 20 | 107 | 107 | 0.6449 | 1.1315 | -0.4206 | 12.9117 | 0.3551 | 0.5327 |
| raw_icon_d2_residual_distribution | 6 | 107 | 107 | 0.5952 | 0.7399 | 0.0617 | 1.1876 | 0.0682 | 0.9813 |
| raw_icon_d2_residual_distribution | 8 | 107 | 107 | 0.6712 | 0.9533 | 0.0739 | 1.4477 | 0.0775 | 0.9720 |
| raw_icon_d2_residual_distribution | 10 | 107 | 107 | 0.7501 | 1.1867 | 0.0459 | 1.7287 | 0.0861 | 0.9626 |
| raw_icon_d2_residual_distribution | 12 | 107 | 107 | 0.7309 | 1.1716 | 0.0402 | 1.7031 | 0.0844 | 0.9626 |
| raw_icon_d2_residual_distribution | 14 | 107 | 107 | 0.6221 | 0.7942 | 0.4112 | 1.1183 | 0.0575 | 0.9439 |
| raw_icon_d2_residual_distribution | 16 | 107 | 107 | 0.5509 | 0.6920 | 0.4909 | 0.8220 | 0.0453 | 0.9720 |
| raw_icon_d2_residual_distribution | 18 | 107 | 107 | 0.4992 | 0.6024 | 0.4992 | 0.6202 | 0.0336 | 0.9720 |
| raw_icon_d2_residual_distribution | 20 | 107 | 107 | 0.6163 | 0.9286 | 0.4854 | 0.9014 | 0.0425 | 0.9626 |

## Limitations

- This is a preliminary stress-test on the currently backfilled LFPB ICON-D2 window only.
- The full 2025-2026 NWP backfill is not complete yet.
- NWP-aware ML is uncalibrated here because the overlap period is still short.
- All NWP rows are selected as-of: knowledge_time_utc <= issue_time_utc.
