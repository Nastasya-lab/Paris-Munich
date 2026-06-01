# Intraday seasonal survival prior analysis

This is a shadow-only historical experiment. Production forecasting logic is unchanged.

## Design

Shadow-only expanding monthly backtest. For each August-December 2025 fold, the ICON-D2 residual prior, intraday analogues, and seasonal METAR Tmax timing survival table use only dates before the fold.

The seasonal survival prior is `P(first rounded METAR Tmax attainment is still ahead | season, local hour)`.

## Seasonal hourly survival prior

| season | local_hour | training_days | peak_ahead_days | survival_prior |
| --- | --- | --- | --- | --- |
| winter_DJF | 0 | 541 | 468 | 0.8651 |
| winter_DJF | 1 | 541 | 464 | 0.8577 |
| winter_DJF | 2 | 541 | 458 | 0.8466 |
| winter_DJF | 3 | 541 | 454 | 0.8392 |
| winter_DJF | 4 | 541 | 452 | 0.8355 |
| winter_DJF | 5 | 541 | 447 | 0.8262 |
| winter_DJF | 6 | 541 | 442 | 0.8170 |
| winter_DJF | 7 | 541 | 438 | 0.8096 |
| winter_DJF | 8 | 541 | 438 | 0.8096 |
| winter_DJF | 9 | 541 | 432 | 0.7985 |
| winter_DJF | 10 | 541 | 416 | 0.7689 |
| winter_DJF | 11 | 541 | 359 | 0.6636 |
| winter_DJF | 12 | 541 | 277 | 0.5120 |
| winter_DJF | 13 | 541 | 164 | 0.3031 |
| winter_DJF | 14 | 541 | 83 | 0.1534 |
| winter_DJF | 15 | 541 | 40 | 0.0739 |
| winter_DJF | 16 | 541 | 35 | 0.0647 |
| winter_DJF | 17 | 541 | 31 | 0.0573 |
| winter_DJF | 18 | 541 | 23 | 0.0425 |
| winter_DJF | 19 | 541 | 19 | 0.0351 |
| winter_DJF | 20 | 541 | 15 | 0.0277 |
| winter_DJF | 21 | 541 | 10 | 0.0185 |
| winter_DJF | 22 | 541 | 6 | 0.0111 |
| winter_DJF | 23 | 541 | 0 | 0.0000 |
| spring_MAM | 0 | 552 | 532 | 0.9638 |
| spring_MAM | 1 | 552 | 531 | 0.9620 |
| spring_MAM | 2 | 552 | 530 | 0.9601 |
| spring_MAM | 3 | 552 | 529 | 0.9583 |
| spring_MAM | 4 | 552 | 529 | 0.9583 |
| spring_MAM | 5 | 552 | 529 | 0.9583 |
| spring_MAM | 6 | 552 | 529 | 0.9583 |
| spring_MAM | 7 | 552 | 529 | 0.9583 |
| spring_MAM | 8 | 552 | 526 | 0.9529 |
| spring_MAM | 9 | 552 | 521 | 0.9438 |
| spring_MAM | 10 | 552 | 510 | 0.9239 |
| spring_MAM | 11 | 552 | 487 | 0.8822 |
| spring_MAM | 12 | 552 | 443 | 0.8025 |
| spring_MAM | 13 | 552 | 368 | 0.6667 |
| spring_MAM | 14 | 552 | 255 | 0.4620 |
| spring_MAM | 15 | 552 | 139 | 0.2518 |
| spring_MAM | 16 | 552 | 47 | 0.0851 |
| spring_MAM | 17 | 552 | 11 | 0.0199 |
| spring_MAM | 18 | 552 | 3 | 0.0054 |
| spring_MAM | 19 | 552 | 1 | 0.0018 |
| spring_MAM | 20 | 552 | 1 | 0.0018 |
| spring_MAM | 21 | 552 | 1 | 0.0018 |
| spring_MAM | 22 | 552 | 0 | 0.0000 |
| spring_MAM | 23 | 552 | 0 | 0.0000 |
| summer_JJA | 0 | 552 | 544 | 0.9855 |
| summer_JJA | 1 | 552 | 544 | 0.9855 |
| summer_JJA | 2 | 552 | 541 | 0.9801 |
| summer_JJA | 3 | 552 | 541 | 0.9801 |
| summer_JJA | 4 | 552 | 541 | 0.9801 |
| summer_JJA | 5 | 552 | 541 | 0.9801 |
| summer_JJA | 6 | 552 | 541 | 0.9801 |
| summer_JJA | 7 | 552 | 541 | 0.9801 |
| summer_JJA | 8 | 552 | 539 | 0.9764 |
| summer_JJA | 9 | 552 | 534 | 0.9674 |
| summer_JJA | 10 | 552 | 530 | 0.9601 |
| summer_JJA | 11 | 552 | 513 | 0.9293 |
| summer_JJA | 12 | 552 | 482 | 0.8732 |
| summer_JJA | 13 | 552 | 429 | 0.7772 |
| summer_JJA | 14 | 552 | 325 | 0.5888 |
| summer_JJA | 15 | 552 | 181 | 0.3279 |
| summer_JJA | 16 | 552 | 65 | 0.1178 |
| summer_JJA | 17 | 552 | 16 | 0.0290 |
| summer_JJA | 18 | 552 | 2 | 0.0036 |
| summer_JJA | 19 | 552 | 0 | 0.0000 |
| summer_JJA | 20 | 552 | 0 | 0.0000 |
| summer_JJA | 21 | 552 | 0 | 0.0000 |
| summer_JJA | 22 | 552 | 0 | 0.0000 |
| summer_JJA | 23 | 552 | 0 | 0.0000 |
| autumn_SON | 0 | 546 | 519 | 0.9505 |
| autumn_SON | 1 | 546 | 515 | 0.9432 |
| autumn_SON | 2 | 546 | 514 | 0.9414 |
| autumn_SON | 3 | 546 | 513 | 0.9396 |
| autumn_SON | 4 | 546 | 513 | 0.9396 |
| autumn_SON | 5 | 546 | 510 | 0.9341 |
| autumn_SON | 6 | 546 | 509 | 0.9322 |
| autumn_SON | 7 | 546 | 508 | 0.9304 |
| autumn_SON | 8 | 546 | 505 | 0.9249 |
| autumn_SON | 9 | 546 | 500 | 0.9158 |
| autumn_SON | 10 | 546 | 482 | 0.8828 |
| autumn_SON | 11 | 546 | 445 | 0.8150 |
| autumn_SON | 12 | 546 | 377 | 0.6905 |
| autumn_SON | 13 | 546 | 263 | 0.4817 |
| autumn_SON | 14 | 546 | 140 | 0.2564 |
| autumn_SON | 15 | 546 | 53 | 0.0971 |
| autumn_SON | 16 | 546 | 16 | 0.0293 |
| autumn_SON | 17 | 546 | 11 | 0.0201 |
| autumn_SON | 18 | 546 | 10 | 0.0183 |
| autumn_SON | 19 | 546 | 8 | 0.0147 |
| autumn_SON | 20 | 546 | 6 | 0.0110 |
| autumn_SON | 21 | 546 | 2 | 0.0037 |
| autumn_SON | 22 | 546 | 1 | 0.0018 |
| autumn_SON | 23 | 546 | 0 | 0.0000 |

## Overall candidate comparison

| model_variant | rows | mae | rmse | nll | crps | brier_upside | coverage80 | mean_predicted_upside_probability | actual_upside_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all__cap_blend_025 | 1059 | 0.8293 | 1.2162 | 3.3744 | 0.0078 | 0.0789 | 0.8848 | 0.5999 | 0.5647 |
| all__cap_blend_050 | 1059 | 0.8398 | 1.2383 | 3.3862 | 0.0079 | 0.0789 | 0.8810 | 0.5850 | 0.5647 |
| all__cap_blend_075 | 1059 | 0.8526 | 1.2641 | 3.4013 | 0.0081 | 0.0800 | 0.8763 | 0.5700 | 0.5647 |
| all__cap_blend_100 | 1059 | 0.8679 | 1.2934 | 3.4431 | 0.0084 | 0.0821 | 0.8754 | 0.5550 | 0.5647 |
| all__multiply_025 | 1059 | 0.8275 | 1.2235 | 3.4143 | 0.0080 | 0.0769 | 0.8763 | 0.5725 | 0.5647 |
| all__multiply_050 | 1059 | 0.8449 | 1.2588 | 3.4521 | 0.0084 | 0.0786 | 0.8735 | 0.5438 | 0.5647 |
| all__multiply_075 | 1059 | 0.8673 | 1.2986 | 3.4946 | 0.0087 | 0.0819 | 0.8678 | 0.5213 | 0.5647 |
| all__multiply_100 | 1059 | 0.8916 | 1.3412 | 3.5394 | 0.0091 | 0.0858 | 0.8555 | 0.5024 | 0.5647 |
| current_dynamic | 1059 | 0.8213 | 1.1979 | 3.3649 | 0.0076 | 0.0799 | 0.8848 | 0.6149 | 0.5647 |
| local_ge17__cap_blend_025 | 1059 | 0.8187 | 1.1968 | 3.3649 | 0.0076 | 0.0792 | 0.8820 | 0.6112 | 0.5647 |
| local_ge17__cap_blend_050 | 1059 | 0.8165 | 1.1957 | 3.3659 | 0.0077 | 0.0787 | 0.8782 | 0.6075 | 0.5647 |
| local_ge17__cap_blend_075 | 1059 | 0.8148 | 1.1949 | 3.3685 | 0.0077 | 0.0785 | 0.8754 | 0.6038 | 0.5647 |
| local_ge17__cap_blend_100 | 1059 | 0.8137 | 1.1942 | 3.3958 | 0.0077 | 0.0784 | 0.8754 | 0.6001 | 0.5647 |
| local_ge17__multiply_025 | 1059 | 0.8143 | 1.1946 | 3.3929 | 0.0077 | 0.0785 | 0.8754 | 0.6026 | 0.5647 |
| local_ge17__multiply_050 | 1059 | 0.8129 | 1.1939 | 3.4045 | 0.0077 | 0.0785 | 0.8754 | 0.5985 | 0.5647 |
| local_ge17__multiply_075 | 1059 | 0.8125 | 1.1937 | 3.4179 | 0.0078 | 0.0785 | 0.8754 | 0.5970 | 0.5647 |
| local_ge17__multiply_100 | 1059 | 0.8124 | 1.1936 | 3.4321 | 0.0078 | 0.0786 | 0.8754 | 0.5964 | 0.5647 |
| survival_le_005__cap_blend_025 | 1059 | 0.8184 | 1.1966 | 3.3649 | 0.0076 | 0.0791 | 0.8810 | 0.6108 | 0.5647 |
| survival_le_005__cap_blend_050 | 1059 | 0.8160 | 1.1955 | 3.3659 | 0.0077 | 0.0786 | 0.8772 | 0.6066 | 0.5647 |
| survival_le_005__cap_blend_075 | 1059 | 0.8141 | 1.1945 | 3.3687 | 0.0077 | 0.0783 | 0.8744 | 0.6024 | 0.5647 |
| survival_le_005__cap_blend_100 | 1059 | 0.8127 | 1.1936 | 3.3963 | 0.0077 | 0.0783 | 0.8744 | 0.5983 | 0.5647 |
| survival_le_005__multiply_025 | 1059 | 0.8134 | 1.1941 | 3.3933 | 0.0077 | 0.0783 | 0.8744 | 0.6009 | 0.5647 |
| survival_le_005__multiply_050 | 1059 | 0.8117 | 1.1932 | 3.4061 | 0.0078 | 0.0783 | 0.8744 | 0.5962 | 0.5647 |
| survival_le_005__multiply_075 | 1059 | 0.8111 | 1.1929 | 3.4209 | 0.0078 | 0.0784 | 0.8744 | 0.5943 | 0.5647 |
| survival_le_005__multiply_100 | 1059 | 0.8109 | 1.1928 | 3.4366 | 0.0078 | 0.0784 | 0.8744 | 0.5936 | 0.5647 |
| survival_le_010__cap_blend_025 | 1059 | 0.8185 | 1.1967 | 3.3652 | 0.0077 | 0.0791 | 0.8801 | 0.6105 | 0.5647 |
| survival_le_010__cap_blend_050 | 1059 | 0.8162 | 1.1956 | 3.3666 | 0.0077 | 0.0787 | 0.8763 | 0.6060 | 0.5647 |
| survival_le_010__cap_blend_075 | 1059 | 0.8143 | 1.1947 | 3.3700 | 0.0077 | 0.0784 | 0.8716 | 0.6016 | 0.5647 |
| survival_le_010__cap_blend_100 | 1059 | 0.8131 | 1.1939 | 3.3984 | 0.0078 | 0.0785 | 0.8716 | 0.5971 | 0.5647 |
| survival_le_010__multiply_025 | 1059 | 0.8135 | 1.1942 | 3.3948 | 0.0077 | 0.0785 | 0.8716 | 0.5994 | 0.5647 |
| survival_le_010__multiply_050 | 1059 | 0.8119 | 1.1934 | 3.4096 | 0.0078 | 0.0786 | 0.8716 | 0.5940 | 0.5647 |
| survival_le_010__multiply_075 | 1059 | 0.8115 | 1.1932 | 3.4268 | 0.0078 | 0.0787 | 0.8716 | 0.5918 | 0.5647 |
| survival_le_010__multiply_100 | 1059 | 0.8113 | 1.1931 | 3.4450 | 0.0078 | 0.0788 | 0.8716 | 0.5908 | 0.5647 |
| survival_le_020__cap_blend_025 | 1059 | 0.8185 | 1.1967 | 3.3652 | 0.0077 | 0.0791 | 0.8801 | 0.6105 | 0.5647 |
| survival_le_020__cap_blend_050 | 1059 | 0.8162 | 1.1956 | 3.3666 | 0.0077 | 0.0787 | 0.8763 | 0.6060 | 0.5647 |
| survival_le_020__cap_blend_075 | 1059 | 0.8143 | 1.1947 | 3.3700 | 0.0077 | 0.0784 | 0.8716 | 0.6016 | 0.5647 |
| survival_le_020__cap_blend_100 | 1059 | 0.8131 | 1.1939 | 3.3984 | 0.0078 | 0.0785 | 0.8716 | 0.5971 | 0.5647 |
| survival_le_020__multiply_025 | 1059 | 0.8135 | 1.1942 | 3.3948 | 0.0077 | 0.0785 | 0.8716 | 0.5994 | 0.5647 |
| survival_le_020__multiply_050 | 1059 | 0.8119 | 1.1934 | 3.4096 | 0.0078 | 0.0786 | 0.8716 | 0.5940 | 0.5647 |
| survival_le_020__multiply_075 | 1059 | 0.8115 | 1.1932 | 3.4268 | 0.0078 | 0.0787 | 0.8716 | 0.5918 | 0.5647 |
| survival_le_020__multiply_100 | 1059 | 0.8113 | 1.1931 | 3.4450 | 0.0078 | 0.0788 | 0.8716 | 0.5908 | 0.5647 |
| survival_le_035__cap_blend_025 | 1059 | 0.8185 | 1.1966 | 3.3674 | 0.0077 | 0.0789 | 0.8801 | 0.6071 | 0.5647 |
| survival_le_035__cap_blend_050 | 1059 | 0.8166 | 1.1962 | 3.3719 | 0.0078 | 0.0786 | 0.8763 | 0.5992 | 0.5647 |
| survival_le_035__cap_blend_075 | 1059 | 0.8157 | 1.1966 | 3.3794 | 0.0078 | 0.0791 | 0.8716 | 0.5914 | 0.5647 |
| survival_le_035__cap_blend_100 | 1059 | 0.8162 | 1.1980 | 3.4134 | 0.0079 | 0.0802 | 0.8716 | 0.5835 | 0.5647 |
| survival_le_035__multiply_025 | 1059 | 0.8141 | 1.1951 | 3.4020 | 0.0078 | 0.0780 | 0.8716 | 0.5897 | 0.5647 |
| survival_le_035__multiply_050 | 1059 | 0.8144 | 1.1970 | 3.4263 | 0.0080 | 0.0795 | 0.8697 | 0.5772 | 0.5647 |
| survival_le_035__multiply_075 | 1059 | 0.8166 | 1.1997 | 3.4544 | 0.0081 | 0.0814 | 0.8650 | 0.5699 | 0.5647 |
| survival_le_035__multiply_100 | 1059 | 0.8191 | 1.2024 | 3.4843 | 0.0082 | 0.0832 | 0.8565 | 0.5652 | 0.5647 |

## Late-day comparison

| regime | model_variant | rows | mae | nll | crps | brier_upside | mean_predicted_upside_probability | actual_upside_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| late_local_ge17 | all__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | all__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | all__cap_blend_025 | 244 | 0.4479 | 7.1617 | 0.0038 | 0.0278 | 0.0700 | 0.0246 |
| late_local_ge17 | all__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | all__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | all__cap_blend_050 | 244 | 0.4370 | 7.1640 | 0.0039 | 0.0255 | 0.0528 | 0.0246 |
| late_local_ge17 | all__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | all__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | all__cap_blend_075 | 244 | 0.4280 | 7.1734 | 0.0040 | 0.0242 | 0.0356 | 0.0246 |
| late_local_ge17 | all__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | all__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | all__cap_blend_100 | 244 | 0.4216 | 7.2889 | 0.0042 | 0.0239 | 0.0184 | 0.0246 |
| late_local_ge17 | all__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | all__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | all__multiply_025 | 244 | 0.4257 | 7.2800 | 0.0041 | 0.0243 | 0.0307 | 0.0246 |
| late_local_ge17 | all__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | all__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | all__multiply_050 | 244 | 0.4175 | 7.3263 | 0.0042 | 0.0241 | 0.0120 | 0.0246 |
| late_local_ge17 | all__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | all__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | all__multiply_075 | 244 | 0.4148 | 7.3812 | 0.0043 | 0.0243 | 0.0048 | 0.0246 |
| late_local_ge17 | all__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | all__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | all__multiply_100 | 244 | 0.4138 | 7.4393 | 0.0043 | 0.0245 | 0.0019 | 0.0246 |
| late_local_ge17 | current_dynamic | 238 | 0.4487 | 6.7725 | 0.0039 | 0.0308 | 0.0844 | 0.0252 |
| summer_late_local_ge17 | current_dynamic | 62 | 0.5455 | 8.6865 | 0.0034 | 0.0360 | 0.1398 | 0.0161 |
| warm_half_late_local_ge16 | current_dynamic | 244 | 0.4607 | 7.1633 | 0.0037 | 0.0311 | 0.0872 | 0.0246 |
| late_local_ge17 | local_ge17__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | local_ge17__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__cap_blend_025 | 244 | 0.4490 | 7.1618 | 0.0038 | 0.0280 | 0.0720 | 0.0246 |
| late_local_ge17 | local_ge17__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | local_ge17__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__cap_blend_050 | 244 | 0.4391 | 7.1639 | 0.0039 | 0.0260 | 0.0568 | 0.0246 |
| late_local_ge17 | local_ge17__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | local_ge17__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__cap_blend_075 | 244 | 0.4311 | 7.1727 | 0.0040 | 0.0249 | 0.0416 | 0.0246 |
| late_local_ge17 | local_ge17__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | local_ge17__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__cap_blend_100 | 244 | 0.4256 | 7.2867 | 0.0041 | 0.0247 | 0.0264 | 0.0246 |
| late_local_ge17 | local_ge17__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | local_ge17__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__multiply_025 | 244 | 0.4295 | 7.2781 | 0.0040 | 0.0250 | 0.0380 | 0.0246 |
| late_local_ge17 | local_ge17__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | local_ge17__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__multiply_050 | 244 | 0.4228 | 7.3195 | 0.0041 | 0.0248 | 0.0224 | 0.0246 |
| late_local_ge17 | local_ge17__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | local_ge17__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__multiply_075 | 244 | 0.4208 | 7.3682 | 0.0042 | 0.0250 | 0.0164 | 0.0246 |
| late_local_ge17 | local_ge17__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | local_ge17__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | local_ge17__multiply_100 | 244 | 0.4200 | 7.4195 | 0.0042 | 0.0252 | 0.0141 | 0.0246 |
| late_local_ge17 | survival_le_005__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__cap_blend_025 | 244 | 0.4479 | 7.1617 | 0.0038 | 0.0278 | 0.0700 | 0.0246 |
| late_local_ge17 | survival_le_005__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__cap_blend_050 | 244 | 0.4370 | 7.1640 | 0.0039 | 0.0255 | 0.0528 | 0.0246 |
| late_local_ge17 | survival_le_005__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__cap_blend_075 | 244 | 0.4280 | 7.1734 | 0.0040 | 0.0242 | 0.0356 | 0.0246 |
| late_local_ge17 | survival_le_005__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__cap_blend_100 | 244 | 0.4216 | 7.2889 | 0.0042 | 0.0239 | 0.0184 | 0.0246 |
| late_local_ge17 | survival_le_005__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__multiply_025 | 244 | 0.4257 | 7.2800 | 0.0041 | 0.0243 | 0.0307 | 0.0246 |
| late_local_ge17 | survival_le_005__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__multiply_050 | 244 | 0.4175 | 7.3263 | 0.0042 | 0.0241 | 0.0120 | 0.0246 |
| late_local_ge17 | survival_le_005__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__multiply_075 | 244 | 0.4148 | 7.3812 | 0.0043 | 0.0243 | 0.0048 | 0.0246 |
| late_local_ge17 | survival_le_005__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | survival_le_005__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_005__multiply_100 | 244 | 0.4138 | 7.4393 | 0.0043 | 0.0245 | 0.0019 | 0.0246 |
| late_local_ge17 | survival_le_010__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__cap_blend_025 | 244 | 0.4479 | 7.1617 | 0.0038 | 0.0278 | 0.0700 | 0.0246 |
| late_local_ge17 | survival_le_010__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__cap_blend_050 | 244 | 0.4370 | 7.1640 | 0.0039 | 0.0255 | 0.0528 | 0.0246 |
| late_local_ge17 | survival_le_010__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__cap_blend_075 | 244 | 0.4280 | 7.1734 | 0.0040 | 0.0242 | 0.0356 | 0.0246 |
| late_local_ge17 | survival_le_010__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__cap_blend_100 | 244 | 0.4216 | 7.2889 | 0.0042 | 0.0239 | 0.0184 | 0.0246 |
| late_local_ge17 | survival_le_010__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__multiply_025 | 244 | 0.4257 | 7.2800 | 0.0041 | 0.0243 | 0.0307 | 0.0246 |
| late_local_ge17 | survival_le_010__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__multiply_050 | 244 | 0.4175 | 7.3263 | 0.0042 | 0.0241 | 0.0120 | 0.0246 |
| late_local_ge17 | survival_le_010__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__multiply_075 | 244 | 0.4148 | 7.3812 | 0.0043 | 0.0243 | 0.0048 | 0.0246 |
| late_local_ge17 | survival_le_010__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | survival_le_010__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_010__multiply_100 | 244 | 0.4138 | 7.4393 | 0.0043 | 0.0245 | 0.0019 | 0.0246 |
| late_local_ge17 | survival_le_020__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__cap_blend_025 | 244 | 0.4479 | 7.1617 | 0.0038 | 0.0278 | 0.0700 | 0.0246 |
| late_local_ge17 | survival_le_020__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__cap_blend_050 | 244 | 0.4370 | 7.1640 | 0.0039 | 0.0255 | 0.0528 | 0.0246 |
| late_local_ge17 | survival_le_020__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__cap_blend_075 | 244 | 0.4280 | 7.1734 | 0.0040 | 0.0242 | 0.0356 | 0.0246 |
| late_local_ge17 | survival_le_020__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__cap_blend_100 | 244 | 0.4216 | 7.2889 | 0.0042 | 0.0239 | 0.0184 | 0.0246 |
| late_local_ge17 | survival_le_020__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__multiply_025 | 244 | 0.4257 | 7.2800 | 0.0041 | 0.0243 | 0.0307 | 0.0246 |
| late_local_ge17 | survival_le_020__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__multiply_050 | 244 | 0.4175 | 7.3263 | 0.0042 | 0.0241 | 0.0120 | 0.0246 |
| late_local_ge17 | survival_le_020__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__multiply_075 | 244 | 0.4148 | 7.3812 | 0.0043 | 0.0243 | 0.0048 | 0.0246 |
| late_local_ge17 | survival_le_020__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | survival_le_020__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_020__multiply_100 | 244 | 0.4138 | 7.4393 | 0.0043 | 0.0245 | 0.0019 | 0.0246 |
| late_local_ge17 | survival_le_035__cap_blend_025 | 238 | 0.4369 | 6.7725 | 0.0040 | 0.0277 | 0.0679 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__cap_blend_025 | 62 | 0.5143 | 8.6716 | 0.0036 | 0.0269 | 0.1082 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__cap_blend_025 | 244 | 0.4479 | 7.1617 | 0.0038 | 0.0278 | 0.0700 | 0.0246 |
| late_local_ge17 | survival_le_035__cap_blend_050 | 238 | 0.4271 | 6.7768 | 0.0041 | 0.0256 | 0.0514 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__cap_blend_050 | 62 | 0.4897 | 8.6627 | 0.0038 | 0.0204 | 0.0765 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__cap_blend_050 | 244 | 0.4370 | 7.1640 | 0.0039 | 0.0255 | 0.0528 | 0.0246 |
| late_local_ge17 | survival_le_035__cap_blend_075 | 238 | 0.4195 | 6.7887 | 0.0043 | 0.0246 | 0.0349 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__cap_blend_075 | 62 | 0.4709 | 8.6646 | 0.0040 | 0.0167 | 0.0448 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__cap_blend_075 | 244 | 0.4280 | 7.1734 | 0.0040 | 0.0242 | 0.0356 | 0.0246 |
| late_local_ge17 | survival_le_035__cap_blend_100 | 238 | 0.4146 | 6.9100 | 0.0044 | 0.0245 | 0.0185 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__cap_blend_100 | 62 | 0.4616 | 9.0442 | 0.0043 | 0.0156 | 0.0132 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__cap_blend_100 | 244 | 0.4216 | 7.2889 | 0.0042 | 0.0239 | 0.0184 | 0.0246 |
| late_local_ge17 | survival_le_035__multiply_025 | 238 | 0.4174 | 6.8972 | 0.0043 | 0.0246 | 0.0295 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__multiply_025 | 62 | 0.4718 | 9.0328 | 0.0040 | 0.0172 | 0.0428 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__multiply_025 | 244 | 0.4257 | 7.2800 | 0.0041 | 0.0243 | 0.0307 | 0.0246 |
| late_local_ge17 | survival_le_035__multiply_050 | 238 | 0.4112 | 6.9487 | 0.0045 | 0.0246 | 0.0116 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__multiply_050 | 62 | 0.4616 | 9.0424 | 0.0042 | 0.0156 | 0.0173 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__multiply_050 | 244 | 0.4175 | 7.3263 | 0.0042 | 0.0241 | 0.0120 | 0.0246 |
| late_local_ge17 | survival_le_035__multiply_075 | 238 | 0.4095 | 7.0085 | 0.0045 | 0.0249 | 0.0046 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__multiply_075 | 62 | 0.4595 | 9.0640 | 0.0043 | 0.0158 | 0.0070 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__multiply_075 | 244 | 0.4148 | 7.3812 | 0.0043 | 0.0243 | 0.0048 | 0.0246 |
| late_local_ge17 | survival_le_035__multiply_100 | 238 | 0.4088 | 7.0714 | 0.0045 | 0.0251 | 0.0018 | 0.0252 |
| summer_late_local_ge17 | survival_le_035__multiply_100 | 62 | 0.4586 | 9.0901 | 0.0044 | 0.0160 | 0.0028 | 0.0161 |
| warm_half_late_local_ge16 | survival_le_035__multiply_100 | 244 | 0.4138 | 7.4393 | 0.0043 | 0.0245 | 0.0019 | 0.0246 |

## By local hour

| model_variant | local_hour_floor | rows | mae | nll | crps | brier_upside | mean_predicted_upside_probability | actual_upside_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_dynamic | 1 | 65 | 1.2473 | 2.4781 | 0.0112 | 0.1168 | 0.9013 | 0.8154 |
| current_dynamic | 2 | 87 | 0.9378 | 2.2121 | 0.0084 | 0.0166 | 0.9724 | 0.9540 |
| current_dynamic | 4 | 66 | 1.2627 | 2.1633 | 0.0116 | 0.1027 | 0.8904 | 0.8182 |
| current_dynamic | 5 | 85 | 0.8789 | 1.9389 | 0.0075 | 0.0160 | 0.9625 | 0.9529 |
| current_dynamic | 7 | 66 | 1.2663 | 2.1495 | 0.0116 | 0.1290 | 0.8750 | 0.7727 |
| current_dynamic | 8 | 85 | 1.1931 | 2.0277 | 0.0094 | 0.0190 | 0.9596 | 0.9529 |
| current_dynamic | 10 | 66 | 1.0930 | 1.9839 | 0.0102 | 0.1521 | 0.8122 | 0.7121 |
| current_dynamic | 11 | 84 | 0.8103 | 1.9714 | 0.0083 | 0.0576 | 0.8981 | 0.9048 |
| current_dynamic | 13 | 66 | 0.6260 | 1.0597 | 0.0065 | 0.2388 | 0.5051 | 0.3485 |
| current_dynamic | 14 | 85 | 0.5586 | 2.7142 | 0.0070 | 0.1834 | 0.4649 | 0.4588 |
| current_dynamic | 16 | 66 | 0.4291 | 5.7433 | 0.0049 | 0.0561 | 0.0931 | 0.0606 |
| current_dynamic | 17 | 86 | 0.5008 | 6.7253 | 0.0036 | 0.0483 | 0.1400 | 0.0349 |
| current_dynamic | 19 | 66 | 0.3876 | 5.7042 | 0.0048 | 0.0294 | 0.0537 | 0.0303 |
| current_dynamic | 20 | 86 | 0.4434 | 7.6398 | 0.0035 | 0.0144 | 0.0522 | 0.0116 |
| local_ge17__cap_blend_025 | 1 | 65 | 1.2473 | 2.4781 | 0.0112 | 0.1168 | 0.9013 | 0.8154 |
| local_ge17__cap_blend_025 | 2 | 87 | 0.9378 | 2.2121 | 0.0084 | 0.0166 | 0.9724 | 0.9540 |
| local_ge17__cap_blend_025 | 4 | 66 | 1.2627 | 2.1633 | 0.0116 | 0.1027 | 0.8904 | 0.8182 |
| local_ge17__cap_blend_025 | 5 | 85 | 0.8789 | 1.9389 | 0.0075 | 0.0160 | 0.9625 | 0.9529 |
| local_ge17__cap_blend_025 | 7 | 66 | 1.2663 | 2.1495 | 0.0116 | 0.1290 | 0.8750 | 0.7727 |
| local_ge17__cap_blend_025 | 8 | 85 | 1.1931 | 2.0277 | 0.0094 | 0.0190 | 0.9596 | 0.9529 |
| local_ge17__cap_blend_025 | 10 | 66 | 1.0930 | 1.9839 | 0.0102 | 0.1521 | 0.8122 | 0.7121 |
| local_ge17__cap_blend_025 | 11 | 84 | 0.8103 | 1.9714 | 0.0083 | 0.0576 | 0.8981 | 0.9048 |
| local_ge17__cap_blend_025 | 13 | 66 | 0.6260 | 1.0597 | 0.0065 | 0.2388 | 0.5051 | 0.3485 |
| local_ge17__cap_blend_025 | 14 | 85 | 0.5586 | 2.7142 | 0.0070 | 0.1834 | 0.4649 | 0.4588 |
| local_ge17__cap_blend_025 | 16 | 66 | 0.4291 | 5.7433 | 0.0049 | 0.0561 | 0.0931 | 0.0606 |
| local_ge17__cap_blend_025 | 17 | 86 | 0.4779 | 6.7168 | 0.0038 | 0.0412 | 0.1112 | 0.0349 |
| local_ge17__cap_blend_025 | 19 | 66 | 0.3865 | 5.7121 | 0.0049 | 0.0292 | 0.0464 | 0.0303 |
| local_ge17__cap_blend_025 | 20 | 86 | 0.4346 | 7.6421 | 0.0036 | 0.0131 | 0.0411 | 0.0116 |

## Recommendation

Least harmful historical shadow candidate by CRPS then NLL: `local_ge17__cap_blend_025`.

Eligible for production promotion from this experiment alone: `False`.

Keep production unchanged. No tested candidate improves overall CRPS; run the least harmful late-day correction in forward shadow mode before considering promotion.

## Limitations

- Historical evaluation is limited to August-December 2025 because honest forecast-as-issued ICON-D2 overlap starts in late July 2025.
- Historical feature rows run at 00/03/06/09/12/15/18 UTC; Railway METAR events arrive around :20/:50 and require continued forward shadow monitoring.
- METAR temperatures are integer-rounded. The survival prior measures first attainment of the rounded daily METAR maximum, not the exact DWD 10-minute Tmax time.
- This report does not change production forecasts.