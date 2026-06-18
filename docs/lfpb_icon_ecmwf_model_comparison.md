# LFPB ICON/ECMWF model comparison

- created: `2026-06-10T06:29:29.545189+00:00`
- period: `2025-07-27` to `2026-05-30`
- rows: `2432`
- days: `304`
- split: `{'method': 'chronological_60_20_20_by_target_day_on_common_icon_ecmwf_overlap', 'train_start': '2025-07-27', 'train_end': '2026-01-28', 'calibration_start': '2026-01-29', 'calibration_end': '2026-03-30', 'test_start': '2026-03-31', 'test_end': '2026-05-30', 'train_rows': 1456, 'calibration_rows': 488, 'test_rows': 488, 'train_days': 182, 'calibration_days': 61, 'test_days': 61}`
- recommendation: `keep_as_shadow_only`
- reason: The best calibrated ICON+ECMWF candidate improved MAE slightly, but the gain is too small for direct production promotion.

## Overall

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecmwf_ensemble_candidate | 488 | 61 | 0.6725 | 0.9124 | 0.2051 | 1.2267 | 0.0329 | 0.0695 | 0.0581 | 0.0464 | 0.9078 |
| ecmwf_ml_calibrated | 488 | 61 | 0.6609 | 0.9557 | 0.1501 | 3.0447 | 0.0372 | 0.0690 | 0.0645 | 0.0478 | 0.8238 |
| icon_ecmwf_feature_ml_calibrated | 488 | 61 | 0.5338 | 0.7627 | 0.2598 | 3.7021 | 0.0303 | 0.0552 | 0.0575 | 0.0625 | 0.8402 |
| icon_ecmwf_posthoc_icon_safety_blend | 488 | 61 | 0.5332 | 0.7587 | 0.2516 | 1.2842 | 0.0287 | 0.0537 | 0.0553 | 0.0603 | 0.8852 |
| icon_ecmwf_posthoc_smooth | 488 | 61 | 0.5338 | 0.7627 | 0.2598 | 3.7021 | 0.0303 | 0.0552 | 0.0575 | 0.0625 | 0.8402 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 488 | 61 | 0.5332 | 0.7587 | 0.2516 | 1.2842 | 0.0287 | 0.0537 | 0.0553 | 0.0603 | 0.8852 |
| icon_ensemble_candidate | 488 | 61 | 0.6408 | 0.9112 | 0.1784 | 1.1016 | 0.0285 | 0.0530 | 0.0469 | 0.0497 | 0.9713 |
| icon_ml_calibrated | 488 | 61 | 0.6854 | 1.0488 | 0.0982 | 2.4694 | 0.0334 | 0.0492 | 0.0494 | 0.0518 | 0.9611 |
| raw_ecmwf_residual_distribution | 488 | 61 | 0.7952 | 0.9665 | 0.3702 | 1.2790 | 0.0764 | 0.1457 | 0.0703 | 0.0524 | 0.9508 |
| raw_icon_residual_distribution | 488 | 61 | 0.6414 | 0.8234 | 0.4189 | 1.1307 | 0.0617 | 0.1163 | 0.0526 | 0.0493 | 0.9631 |

## By Phase

| model_variant | phase | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecmwf_ensemble_candidate | afternoon | 61 | 61 | 0.4383 | 0.5389 | 0.0252 | 0.7858 | 0.0171 | 0.2026 | 0.0467 | 0.0039 | 0.9836 |
| ecmwf_ensemble_candidate | before_work | 122 | 61 | 1.0061 | 1.2193 | 0.1840 | 1.8647 | 0.0524 | 0.0254 | 0.0240 | 0.0501 | 0.8525 |
| ecmwf_ensemble_candidate | evening | 122 | 61 | 0.2079 | 0.2689 | 0.1932 | 0.2003 | 0.0032 | 0.0353 | 0.0084 | 0.0017 | 1.0000 |
| ecmwf_ensemble_candidate | midday | 122 | 61 | 0.7854 | 0.9840 | 0.3018 | 1.5531 | 0.0424 | 0.1033 | 0.1559 | 0.1036 | 0.8607 |
| ecmwf_ensemble_candidate | morning | 61 | 61 | 0.9429 | 1.1464 | 0.2574 | 1.7915 | 0.0503 | 0.0258 | 0.0413 | 0.0570 | 0.8525 |
| ecmwf_ml_calibrated | afternoon | 61 | 61 | 0.4152 | 0.5616 | -0.1665 | 1.5632 | 0.0206 | 0.2226 | 0.0445 | 0.0011 | 0.9508 |
| ecmwf_ml_calibrated | before_work | 122 | 61 | 1.0819 | 1.3083 | 0.1970 | 6.0613 | 0.0602 | 0.0267 | 0.0296 | 0.0534 | 0.6557 |
| ecmwf_ml_calibrated | evening | 122 | 61 | 0.0528 | 0.1716 | 0.0237 | 0.2720 | 0.0016 | 0.0143 | 0.0066 | 0.0005 | 1.0000 |
| ecmwf_ml_calibrated | midday | 122 | 61 | 0.8045 | 1.0139 | 0.3166 | 2.7652 | 0.0484 | 0.1095 | 0.1763 | 0.1070 | 0.7951 |
| ecmwf_ml_calibrated | morning | 61 | 61 | 0.9936 | 1.2054 | 0.2924 | 4.5974 | 0.0564 | 0.0285 | 0.0463 | 0.0595 | 0.7377 |
| icon_ecmwf_feature_ml_calibrated | afternoon | 61 | 61 | 0.4775 | 0.6796 | 0.3153 | 1.0834 | 0.0266 | 0.1995 | 0.1373 | 0.0086 | 0.8525 |
| icon_ecmwf_feature_ml_calibrated | before_work | 122 | 61 | 0.8091 | 0.9857 | 0.1881 | 6.3676 | 0.0474 | 0.0157 | 0.0159 | 0.0716 | 0.7869 |
| icon_ecmwf_feature_ml_calibrated | evening | 122 | 61 | 0.0830 | 0.2372 | 0.0830 | 0.3116 | 0.0031 | 0.0259 | 0.0097 | 0.0040 | 1.0000 |
| icon_ecmwf_feature_ml_calibrated | midday | 122 | 61 | 0.6653 | 0.8559 | 0.4712 | 4.4048 | 0.0379 | 0.0750 | 0.1190 | 0.1417 | 0.7377 |
| icon_ecmwf_feature_ml_calibrated | morning | 61 | 61 | 0.6783 | 0.8190 | 0.2783 | 6.3653 | 0.0393 | 0.0087 | 0.0337 | 0.0570 | 0.8197 |
| icon_ecmwf_posthoc_icon_safety_blend | afternoon | 61 | 61 | 0.4692 | 0.6602 | 0.3037 | 0.9787 | 0.0237 | 0.1940 | 0.1284 | 0.0081 | 0.9672 |
| icon_ecmwf_posthoc_icon_safety_blend | before_work | 122 | 61 | 0.8139 | 0.9955 | 0.1833 | 2.0425 | 0.0462 | 0.0151 | 0.0162 | 0.0698 | 0.7951 |
| icon_ecmwf_posthoc_icon_safety_blend | evening | 122 | 61 | 0.0928 | 0.2313 | 0.0900 | 0.1322 | 0.0027 | 0.0246 | 0.0092 | 0.0037 | 1.0000 |
| icon_ecmwf_posthoc_icon_safety_blend | midday | 122 | 61 | 0.6437 | 0.8313 | 0.4423 | 1.5467 | 0.0349 | 0.0734 | 0.1146 | 0.1356 | 0.8525 |
| icon_ecmwf_posthoc_icon_safety_blend | morning | 61 | 61 | 0.6957 | 0.8355 | 0.2781 | 1.8519 | 0.0385 | 0.0089 | 0.0336 | 0.0564 | 0.8197 |
| icon_ecmwf_posthoc_smooth | afternoon | 61 | 61 | 0.4775 | 0.6796 | 0.3153 | 1.0834 | 0.0266 | 0.1995 | 0.1373 | 0.0086 | 0.8525 |
| icon_ecmwf_posthoc_smooth | before_work | 122 | 61 | 0.8091 | 0.9857 | 0.1881 | 6.3676 | 0.0474 | 0.0157 | 0.0159 | 0.0716 | 0.7869 |
| icon_ecmwf_posthoc_smooth | evening | 122 | 61 | 0.0830 | 0.2372 | 0.0830 | 0.3116 | 0.0031 | 0.0259 | 0.0097 | 0.0040 | 1.0000 |
| icon_ecmwf_posthoc_smooth | midday | 122 | 61 | 0.6653 | 0.8559 | 0.4712 | 4.4048 | 0.0379 | 0.0750 | 0.1190 | 0.1417 | 0.7377 |
| icon_ecmwf_posthoc_smooth | morning | 61 | 61 | 0.6783 | 0.8190 | 0.2783 | 6.3653 | 0.0393 | 0.0087 | 0.0337 | 0.0570 | 0.8197 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | afternoon | 61 | 61 | 0.4692 | 0.6602 | 0.3037 | 0.9787 | 0.0237 | 0.1940 | 0.1284 | 0.0081 | 0.9672 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | before_work | 122 | 61 | 0.8139 | 0.9955 | 0.1833 | 2.0425 | 0.0462 | 0.0151 | 0.0162 | 0.0698 | 0.7951 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | evening | 122 | 61 | 0.0928 | 0.2313 | 0.0900 | 0.1322 | 0.0027 | 0.0246 | 0.0092 | 0.0037 | 1.0000 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | midday | 122 | 61 | 0.6437 | 0.8313 | 0.4423 | 1.5467 | 0.0349 | 0.0734 | 0.1146 | 0.1356 | 0.8525 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | morning | 61 | 61 | 0.6957 | 0.8355 | 0.2781 | 1.8519 | 0.0385 | 0.0089 | 0.0336 | 0.0564 | 0.8197 |
| icon_ensemble_candidate | afternoon | 61 | 61 | 0.4197 | 0.5364 | 0.1986 | 0.8338 | 0.0185 | 0.1798 | 0.0717 | 0.0058 | 1.0000 |
| icon_ensemble_candidate | before_work | 122 | 61 | 1.0698 | 1.3449 | 0.1404 | 1.6966 | 0.0484 | 0.0137 | 0.0241 | 0.0608 | 0.9262 |
| icon_ensemble_candidate | evening | 122 | 61 | 0.1953 | 0.2870 | 0.1530 | 0.1781 | 0.0029 | 0.0299 | 0.0071 | 0.0014 | 1.0000 |
| icon_ensemble_candidate | midday | 122 | 61 | 0.6098 | 0.7654 | 0.1825 | 1.3039 | 0.0310 | 0.0723 | 0.1012 | 0.1036 | 0.9836 |
| icon_ensemble_candidate | morning | 61 | 61 | 0.9570 | 1.1833 | 0.2765 | 1.6221 | 0.0454 | 0.0123 | 0.0385 | 0.0603 | 0.9508 |
| icon_ml_calibrated | afternoon | 61 | 61 | 0.3777 | 0.5045 | 0.0774 | 1.3102 | 0.0200 | 0.1857 | 0.0692 | 0.0045 | 0.9836 |
| icon_ml_calibrated | before_work | 122 | 61 | 1.2921 | 1.6239 | 0.0951 | 5.4431 | 0.0602 | 0.0123 | 0.0270 | 0.0661 | 0.9180 |
| icon_ml_calibrated | evening | 122 | 61 | 0.0579 | 0.1678 | 0.0326 | 0.2709 | 0.0016 | 0.0132 | 0.0069 | 0.0008 | 1.0000 |
| icon_ml_calibrated | midday | 122 | 61 | 0.6419 | 0.8083 | 0.1030 | 2.0523 | 0.0343 | 0.0728 | 0.1085 | 0.1052 | 0.9754 |
| icon_ml_calibrated | morning | 61 | 61 | 1.1220 | 1.3817 | 0.2469 | 2.9124 | 0.0545 | 0.0113 | 0.0413 | 0.0660 | 0.9180 |
| raw_ecmwf_residual_distribution | afternoon | 61 | 61 | 0.7583 | 0.9168 | 0.6001 | 1.1559 | 0.0730 | 0.2971 | 0.1390 | 0.0258 | 0.9836 |
| raw_ecmwf_residual_distribution | before_work | 122 | 61 | 0.8487 | 1.0446 | 0.1451 | 1.4980 | 0.0877 | 0.0246 | 0.0124 | 0.0435 | 0.9180 |
| raw_ecmwf_residual_distribution | evening | 122 | 61 | 0.7016 | 0.8274 | 0.7016 | 0.8605 | 0.0583 | 0.3004 | 0.0507 | 0.0105 | 0.9836 |
| raw_ecmwf_residual_distribution | midday | 122 | 61 | 0.8289 | 1.0010 | 0.2576 | 1.4302 | 0.0824 | 0.0990 | 0.1334 | 0.1169 | 0.9426 |
| raw_ecmwf_residual_distribution | morning | 61 | 61 | 0.8451 | 1.0375 | 0.1525 | 1.4986 | 0.0817 | 0.0208 | 0.0306 | 0.0516 | 0.9344 |
| raw_icon_residual_distribution | afternoon | 61 | 61 | 0.6260 | 0.7834 | 0.5623 | 0.9936 | 0.0558 | 0.2617 | 0.0985 | 0.0120 | 0.9508 |
| raw_icon_residual_distribution | before_work | 122 | 61 | 0.6355 | 0.7733 | 0.2764 | 1.2605 | 0.0731 | 0.0185 | 0.0189 | 0.0508 | 0.9836 |
| raw_icon_residual_distribution | evening | 122 | 61 | 0.6292 | 0.9140 | 0.5145 | 0.9254 | 0.0452 | 0.2292 | 0.0214 | 0.0051 | 0.9426 |
| raw_icon_residual_distribution | midday | 122 | 61 | 0.6558 | 0.8101 | 0.4207 | 1.2116 | 0.0654 | 0.0786 | 0.1043 | 0.1107 | 0.9590 |
| raw_icon_residual_distribution | morning | 61 | 61 | 0.6643 | 0.7942 | 0.3655 | 1.2568 | 0.0704 | 0.0162 | 0.0329 | 0.0490 | 0.9836 |

## By Local Issue Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecmwf_ensemble_candidate | 6 | 61 | 61 | 1.0118 | 1.2283 | 0.1990 | 1.8912 | 0.0543 | 0.0248 | 0.0224 | 0.0558 | 0.8689 |
| ecmwf_ensemble_candidate | 8 | 61 | 61 | 1.0003 | 1.2102 | 0.1691 | 1.8382 | 0.0506 | 0.0259 | 0.0255 | 0.0444 | 0.8361 |
| ecmwf_ensemble_candidate | 10 | 61 | 61 | 0.9429 | 1.1464 | 0.2574 | 1.7915 | 0.0503 | 0.0258 | 0.0413 | 0.0570 | 0.8525 |
| ecmwf_ensemble_candidate | 12 | 61 | 61 | 0.8563 | 1.0652 | 0.3140 | 1.7190 | 0.0484 | 0.0739 | 0.0896 | 0.1066 | 0.8525 |
| ecmwf_ensemble_candidate | 14 | 61 | 61 | 0.7144 | 0.8955 | 0.2896 | 1.3872 | 0.0363 | 0.1327 | 0.2223 | 0.1006 | 0.8689 |
| ecmwf_ensemble_candidate | 16 | 61 | 61 | 0.4383 | 0.5389 | 0.0252 | 0.7858 | 0.0171 | 0.2026 | 0.0467 | 0.0039 | 0.9836 |
| ecmwf_ensemble_candidate | 18 | 61 | 61 | 0.2455 | 0.3183 | 0.2160 | 0.2511 | 0.0044 | 0.0438 | 0.0153 | 0.0033 | 1.0000 |
| ecmwf_ensemble_candidate | 20 | 61 | 61 | 0.1703 | 0.2080 | 0.1703 | 0.1496 | 0.0020 | 0.0268 | 0.0016 | 0.0001 | 1.0000 |
| ecmwf_ml_calibrated | 6 | 61 | 61 | 1.0944 | 1.3199 | 0.2195 | 6.3302 | 0.0626 | 0.0250 | 0.0286 | 0.0607 | 0.6557 |
| ecmwf_ml_calibrated | 8 | 61 | 61 | 1.0694 | 1.2965 | 0.1746 | 5.7925 | 0.0578 | 0.0285 | 0.0306 | 0.0460 | 0.6557 |
| ecmwf_ml_calibrated | 10 | 61 | 61 | 0.9936 | 1.2054 | 0.2924 | 4.5974 | 0.0564 | 0.0285 | 0.0463 | 0.0595 | 0.7377 |
| ecmwf_ml_calibrated | 12 | 61 | 61 | 0.8845 | 1.0965 | 0.3618 | 3.1760 | 0.0547 | 0.0809 | 0.1008 | 0.1113 | 0.7541 |
| ecmwf_ml_calibrated | 14 | 61 | 61 | 0.7244 | 0.9238 | 0.2713 | 2.3544 | 0.0421 | 0.1381 | 0.2517 | 0.1027 | 0.8361 |
| ecmwf_ml_calibrated | 16 | 61 | 61 | 0.4152 | 0.5616 | -0.1665 | 1.5632 | 0.0206 | 0.2226 | 0.0445 | 0.0011 | 0.9508 |
| ecmwf_ml_calibrated | 18 | 61 | 61 | 0.0842 | 0.2286 | 0.0260 | 0.5182 | 0.0028 | 0.0219 | 0.0132 | 0.0010 | 1.0000 |
| ecmwf_ml_calibrated | 20 | 61 | 61 | 0.0213 | 0.0814 | 0.0213 | 0.0259 | 0.0005 | 0.0066 | 0.0000 | 0.0000 | 1.0000 |
| icon_ecmwf_feature_ml_calibrated | 6 | 61 | 61 | 0.8259 | 0.9952 | 0.2124 | 5.4497 | 0.0504 | 0.0227 | 0.0147 | 0.0731 | 0.8033 |
| icon_ecmwf_feature_ml_calibrated | 8 | 61 | 61 | 0.7923 | 0.9760 | 0.1637 | 7.2856 | 0.0444 | 0.0087 | 0.0172 | 0.0702 | 0.7705 |
| icon_ecmwf_feature_ml_calibrated | 10 | 61 | 61 | 0.6783 | 0.8190 | 0.2783 | 6.3653 | 0.0393 | 0.0087 | 0.0337 | 0.0570 | 0.8197 |
| icon_ecmwf_feature_ml_calibrated | 12 | 61 | 61 | 0.6619 | 0.8346 | 0.3795 | 3.7835 | 0.0368 | 0.0494 | 0.0877 | 0.0879 | 0.8033 |
| icon_ecmwf_feature_ml_calibrated | 14 | 61 | 61 | 0.6688 | 0.8767 | 0.5630 | 5.0260 | 0.0390 | 0.1005 | 0.1502 | 0.1956 | 0.6721 |
| icon_ecmwf_feature_ml_calibrated | 16 | 61 | 61 | 0.4775 | 0.6796 | 0.3153 | 1.0834 | 0.0266 | 0.1995 | 0.1373 | 0.0086 | 0.8525 |
| icon_ecmwf_feature_ml_calibrated | 18 | 61 | 61 | 0.1523 | 0.3288 | 0.1523 | 0.6066 | 0.0058 | 0.0473 | 0.0194 | 0.0081 | 1.0000 |
| icon_ecmwf_feature_ml_calibrated | 20 | 61 | 61 | 0.0136 | 0.0665 | 0.0136 | 0.0166 | 0.0003 | 0.0044 | 0.0000 | 0.0000 | 1.0000 |
| icon_ecmwf_posthoc_icon_safety_blend | 6 | 61 | 61 | 0.8227 | 0.9994 | 0.2031 | 2.0616 | 0.0488 | 0.0213 | 0.0151 | 0.0709 | 0.8197 |
| icon_ecmwf_posthoc_icon_safety_blend | 8 | 61 | 61 | 0.8052 | 0.9916 | 0.1634 | 2.0234 | 0.0437 | 0.0090 | 0.0174 | 0.0687 | 0.7705 |
| icon_ecmwf_posthoc_icon_safety_blend | 10 | 61 | 61 | 0.6957 | 0.8355 | 0.2781 | 1.8519 | 0.0385 | 0.0089 | 0.0336 | 0.0564 | 0.8197 |
| icon_ecmwf_posthoc_icon_safety_blend | 12 | 61 | 61 | 0.6459 | 0.8171 | 0.3561 | 1.5001 | 0.0345 | 0.0476 | 0.0861 | 0.0857 | 0.8361 |
| icon_ecmwf_posthoc_icon_safety_blend | 14 | 61 | 61 | 0.6416 | 0.8452 | 0.5286 | 1.5933 | 0.0353 | 0.0993 | 0.1432 | 0.1856 | 0.8689 |
| icon_ecmwf_posthoc_icon_safety_blend | 16 | 61 | 61 | 0.4692 | 0.6602 | 0.3037 | 0.9787 | 0.0237 | 0.1940 | 0.1284 | 0.0081 | 0.9672 |
| icon_ecmwf_posthoc_icon_safety_blend | 18 | 61 | 61 | 0.1559 | 0.3196 | 0.1559 | 0.2384 | 0.0051 | 0.0451 | 0.0184 | 0.0074 | 1.0000 |
| icon_ecmwf_posthoc_icon_safety_blend | 20 | 61 | 61 | 0.0298 | 0.0696 | 0.0240 | 0.0260 | 0.0003 | 0.0041 | 0.0000 | 0.0000 | 1.0000 |
| icon_ecmwf_posthoc_smooth | 6 | 61 | 61 | 0.8259 | 0.9952 | 0.2124 | 5.4497 | 0.0504 | 0.0227 | 0.0147 | 0.0731 | 0.8033 |
| icon_ecmwf_posthoc_smooth | 8 | 61 | 61 | 0.7923 | 0.9760 | 0.1637 | 7.2856 | 0.0444 | 0.0087 | 0.0172 | 0.0702 | 0.7705 |
| icon_ecmwf_posthoc_smooth | 10 | 61 | 61 | 0.6783 | 0.8190 | 0.2783 | 6.3653 | 0.0393 | 0.0087 | 0.0337 | 0.0570 | 0.8197 |
| icon_ecmwf_posthoc_smooth | 12 | 61 | 61 | 0.6619 | 0.8346 | 0.3795 | 3.7835 | 0.0368 | 0.0494 | 0.0877 | 0.0879 | 0.8033 |
| icon_ecmwf_posthoc_smooth | 14 | 61 | 61 | 0.6688 | 0.8767 | 0.5630 | 5.0260 | 0.0390 | 0.1005 | 0.1502 | 0.1956 | 0.6721 |
| icon_ecmwf_posthoc_smooth | 16 | 61 | 61 | 0.4775 | 0.6796 | 0.3153 | 1.0834 | 0.0266 | 0.1995 | 0.1373 | 0.0086 | 0.8525 |
| icon_ecmwf_posthoc_smooth | 18 | 61 | 61 | 0.1523 | 0.3288 | 0.1523 | 0.6066 | 0.0058 | 0.0473 | 0.0194 | 0.0081 | 1.0000 |
| icon_ecmwf_posthoc_smooth | 20 | 61 | 61 | 0.0136 | 0.0665 | 0.0136 | 0.0166 | 0.0003 | 0.0044 | 0.0000 | 0.0000 | 1.0000 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 6 | 61 | 61 | 0.8227 | 0.9994 | 0.2031 | 2.0616 | 0.0488 | 0.0213 | 0.0151 | 0.0709 | 0.8197 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 8 | 61 | 61 | 0.8052 | 0.9916 | 0.1634 | 2.0234 | 0.0437 | 0.0090 | 0.0174 | 0.0687 | 0.7705 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 10 | 61 | 61 | 0.6957 | 0.8355 | 0.2781 | 1.8519 | 0.0385 | 0.0089 | 0.0336 | 0.0564 | 0.8197 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 12 | 61 | 61 | 0.6459 | 0.8171 | 0.3561 | 1.5001 | 0.0345 | 0.0476 | 0.0861 | 0.0857 | 0.8361 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 14 | 61 | 61 | 0.6416 | 0.8452 | 0.5286 | 1.5933 | 0.0353 | 0.0993 | 0.1432 | 0.1856 | 0.8689 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 16 | 61 | 61 | 0.4692 | 0.6602 | 0.3037 | 0.9787 | 0.0237 | 0.1940 | 0.1284 | 0.0081 | 0.9672 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 18 | 61 | 61 | 0.1559 | 0.3196 | 0.1559 | 0.2384 | 0.0051 | 0.0451 | 0.0184 | 0.0074 | 1.0000 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | 20 | 61 | 61 | 0.0298 | 0.0696 | 0.0240 | 0.0260 | 0.0003 | 0.0041 | 0.0000 | 0.0000 | 1.0000 |
| icon_ensemble_candidate | 6 | 61 | 61 | 1.0530 | 1.3278 | 0.1194 | 1.7088 | 0.0482 | 0.0148 | 0.0235 | 0.0598 | 0.9180 |
| icon_ensemble_candidate | 8 | 61 | 61 | 1.0866 | 1.3619 | 0.1615 | 1.6844 | 0.0485 | 0.0125 | 0.0248 | 0.0618 | 0.9344 |
| icon_ensemble_candidate | 10 | 61 | 61 | 0.9570 | 1.1833 | 0.2765 | 1.6221 | 0.0454 | 0.0123 | 0.0385 | 0.0603 | 0.9508 |
| icon_ensemble_candidate | 12 | 61 | 61 | 0.6989 | 0.8552 | 0.1459 | 1.4007 | 0.0351 | 0.0393 | 0.0802 | 0.0839 | 0.9672 |
| icon_ensemble_candidate | 14 | 61 | 61 | 0.5207 | 0.6636 | 0.2190 | 1.2071 | 0.0268 | 0.1053 | 0.1221 | 0.1234 | 1.0000 |
| icon_ensemble_candidate | 16 | 61 | 61 | 0.4197 | 0.5364 | 0.1986 | 0.8338 | 0.0185 | 0.1798 | 0.0717 | 0.0058 | 1.0000 |
| icon_ensemble_candidate | 18 | 61 | 61 | 0.2159 | 0.2959 | 0.1886 | 0.2372 | 0.0044 | 0.0456 | 0.0131 | 0.0028 | 1.0000 |
| icon_ensemble_candidate | 20 | 61 | 61 | 0.1748 | 0.2778 | 0.1174 | 0.1190 | 0.0014 | 0.0142 | 0.0011 | 0.0001 | 1.0000 |
| icon_ml_calibrated | 6 | 61 | 61 | 1.2767 | 1.6036 | 0.0886 | 5.2706 | 0.0601 | 0.0133 | 0.0266 | 0.0654 | 0.9180 |
| icon_ml_calibrated | 8 | 61 | 61 | 1.3074 | 1.6439 | 0.1015 | 5.6155 | 0.0604 | 0.0113 | 0.0273 | 0.0669 | 0.9180 |
| icon_ml_calibrated | 10 | 61 | 61 | 1.1220 | 1.3817 | 0.2469 | 2.9124 | 0.0545 | 0.0113 | 0.0413 | 0.0660 | 0.9180 |
| icon_ml_calibrated | 12 | 61 | 61 | 0.7638 | 0.9380 | 0.0969 | 2.7537 | 0.0399 | 0.0402 | 0.0880 | 0.0899 | 0.9508 |
| icon_ml_calibrated | 14 | 61 | 61 | 0.5201 | 0.6532 | 0.1092 | 1.3510 | 0.0287 | 0.1055 | 0.1290 | 0.1204 | 1.0000 |
| icon_ml_calibrated | 16 | 61 | 61 | 0.3777 | 0.5045 | 0.0774 | 1.3102 | 0.0200 | 0.1857 | 0.0692 | 0.0045 | 0.9836 |
| icon_ml_calibrated | 18 | 61 | 61 | 0.1147 | 0.2371 | 0.0641 | 0.5406 | 0.0032 | 0.0264 | 0.0138 | 0.0016 | 1.0000 |
| icon_ml_calibrated | 20 | 61 | 61 | 0.0011 | 0.0083 | 0.0011 | 0.0011 | 0.0000 | 0.0001 | 0.0000 | 0.0000 | 1.0000 |
| raw_ecmwf_residual_distribution | 6 | 61 | 61 | 0.8522 | 1.0517 | 0.1376 | 1.4974 | 0.0938 | 0.0285 | 0.0102 | 0.0453 | 0.9016 |
| raw_ecmwf_residual_distribution | 8 | 61 | 61 | 0.8451 | 1.0375 | 0.1525 | 1.4986 | 0.0817 | 0.0208 | 0.0145 | 0.0416 | 0.9344 |
| raw_ecmwf_residual_distribution | 10 | 61 | 61 | 0.8451 | 1.0375 | 0.1525 | 1.4986 | 0.0817 | 0.0208 | 0.0306 | 0.0516 | 0.9344 |
| raw_ecmwf_residual_distribution | 12 | 61 | 61 | 0.8488 | 1.0421 | 0.1707 | 1.4915 | 0.0821 | 0.0580 | 0.0756 | 0.1103 | 0.9344 |
| raw_ecmwf_residual_distribution | 14 | 61 | 61 | 0.8090 | 0.9581 | 0.3445 | 1.3690 | 0.0828 | 0.1400 | 0.1912 | 0.1235 | 0.9508 |
| raw_ecmwf_residual_distribution | 16 | 61 | 61 | 0.7583 | 0.9168 | 0.6001 | 1.1559 | 0.0730 | 0.2971 | 0.1390 | 0.0258 | 0.9836 |
| raw_ecmwf_residual_distribution | 18 | 61 | 61 | 0.7860 | 0.9292 | 0.7860 | 0.9570 | 0.0666 | 0.3238 | 0.0765 | 0.0201 | 0.9836 |
| raw_ecmwf_residual_distribution | 20 | 61 | 61 | 0.6173 | 0.7112 | 0.6173 | 0.7640 | 0.0500 | 0.2770 | 0.0249 | 0.0009 | 0.9836 |
| raw_icon_residual_distribution | 6 | 61 | 61 | 0.6140 | 0.7582 | 0.2116 | 1.2496 | 0.0724 | 0.0203 | 0.0177 | 0.0499 | 0.9836 |
| raw_icon_residual_distribution | 8 | 61 | 61 | 0.6570 | 0.7882 | 0.3413 | 1.2714 | 0.0737 | 0.0168 | 0.0201 | 0.0518 | 0.9836 |
| raw_icon_residual_distribution | 10 | 61 | 61 | 0.6643 | 0.7942 | 0.3655 | 1.2568 | 0.0704 | 0.0162 | 0.0329 | 0.0490 | 0.9836 |
| raw_icon_residual_distribution | 12 | 61 | 61 | 0.6406 | 0.7770 | 0.2929 | 1.2244 | 0.0683 | 0.0389 | 0.0717 | 0.0765 | 0.9836 |
| raw_icon_residual_distribution | 14 | 61 | 61 | 0.6710 | 0.8419 | 0.5486 | 1.1987 | 0.0624 | 0.1184 | 0.1369 | 0.1449 | 0.9344 |
| raw_icon_residual_distribution | 16 | 61 | 61 | 0.6260 | 0.7834 | 0.5623 | 0.9936 | 0.0558 | 0.2617 | 0.0985 | 0.0120 | 0.9508 |
| raw_icon_residual_distribution | 18 | 61 | 61 | 0.5624 | 0.6631 | 0.5624 | 0.7223 | 0.0395 | 0.2344 | 0.0253 | 0.0087 | 0.9508 |
| raw_icon_residual_distribution | 20 | 61 | 61 | 0.6961 | 1.1096 | 0.4666 | 1.1285 | 0.0510 | 0.2240 | 0.0175 | 0.0014 | 0.9344 |

## By Season

| model_variant | season | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ecmwf_ensemble_candidate | spring | 488 | 61 | 0.6725 | 0.9124 | 0.2051 | 1.2267 | 0.0329 | 0.0695 | 0.0581 | 0.0464 | 0.9078 |
| ecmwf_ml_calibrated | spring | 488 | 61 | 0.6609 | 0.9557 | 0.1501 | 3.0447 | 0.0372 | 0.0690 | 0.0645 | 0.0478 | 0.8238 |
| icon_ecmwf_feature_ml_calibrated | spring | 488 | 61 | 0.5338 | 0.7627 | 0.2598 | 3.7021 | 0.0303 | 0.0552 | 0.0575 | 0.0625 | 0.8402 |
| icon_ecmwf_posthoc_icon_safety_blend | spring | 488 | 61 | 0.5332 | 0.7587 | 0.2516 | 1.2842 | 0.0287 | 0.0537 | 0.0553 | 0.0603 | 0.8852 |
| icon_ecmwf_posthoc_smooth | spring | 488 | 61 | 0.5338 | 0.7627 | 0.2598 | 3.7021 | 0.0303 | 0.0552 | 0.0575 | 0.0625 | 0.8402 |
| icon_ecmwf_posthoc_smooth_icon_safety_blend | spring | 488 | 61 | 0.5332 | 0.7587 | 0.2516 | 1.2842 | 0.0287 | 0.0537 | 0.0553 | 0.0603 | 0.8852 |
| icon_ensemble_candidate | spring | 488 | 61 | 0.6408 | 0.9112 | 0.1784 | 1.1016 | 0.0285 | 0.0530 | 0.0469 | 0.0497 | 0.9713 |
| icon_ml_calibrated | spring | 488 | 61 | 0.6854 | 1.0488 | 0.0982 | 2.4694 | 0.0334 | 0.0492 | 0.0494 | 0.0518 | 0.9611 |
| raw_ecmwf_residual_distribution | spring | 488 | 61 | 0.7952 | 0.9665 | 0.3702 | 1.2790 | 0.0764 | 0.1457 | 0.0703 | 0.0524 | 0.9508 |
| raw_icon_residual_distribution | spring | 488 | 61 | 0.6414 | 0.8234 | 0.4189 | 1.1307 | 0.0617 | 0.1163 | 0.0526 | 0.0493 | 0.9631 |

## Limitations

- All variants are compared on the common ICON-D2/ECMWF overlap only.
- The target is METAR Tmax, not official climate Tmax.
- The combined feature model is not deployed; this script is an offline diagnostic.
- A positive production decision should require improvement in MAE without worse NLL/CRPS or obvious phase instability.
