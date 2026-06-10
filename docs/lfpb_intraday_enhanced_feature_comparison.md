# LFPB enhanced intraday METAR feature comparison

- created: `2026-06-10T09:04:33.734188+00:00`
- period: `2020-01-01` to `2026-06-05`
- rows: `18479`
- days: `2310`
- recommendation: `candidate_for_shadow`
- reason: Enhanced intraday features improved point accuracy without a major probabilistic penalty.

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_metar_intraday | 3696 | 462 | 1.1153 | 1.7815 | -0.3082 | 1.8670 | 0.0585 | 0.0748 | 0.0866 | 0.0937 | 0.9242 |
| enhanced_metar_intraday | 3696 | 462 | 1.0399 | 1.6453 | -0.2529 | 1.8328 | 0.0542 | 0.0736 | 0.0836 | 0.0896 | 0.9307 |
| persistence_current_metar_max | 3696 | 462 | 2.6529 | 4.1423 | -2.6529 | 16.7461 | 0.6061 | 0.6061 | 0.4984 | 0.3934 | 0.3939 |

## By Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_metar_intraday | 6 | 462 | 462 | 2.2332 | 2.8392 | -0.7505 | 3.6276 | 0.1196 | 0.0544 | 0.0935 | 0.1387 | 0.8550 |
| base_metar_intraday | 8 | 462 | 462 | 2.1908 | 2.7849 | -0.7249 | 3.4826 | 0.1169 | 0.0526 | 0.0941 | 0.1349 | 0.8723 |
| base_metar_intraday | 10 | 462 | 462 | 1.7746 | 2.2716 | -0.5328 | 3.0451 | 0.0953 | 0.0510 | 0.0882 | 0.1351 | 0.9069 |
| base_metar_intraday | 12 | 462 | 462 | 1.1937 | 1.5667 | -0.2647 | 2.1685 | 0.0650 | 0.0623 | 0.1324 | 0.1770 | 0.9004 |
| base_metar_intraday | 14 | 462 | 462 | 0.8396 | 1.1201 | -0.1758 | 1.4302 | 0.0435 | 0.1499 | 0.1996 | 0.1318 | 0.9221 |
| base_metar_intraday | 16 | 462 | 462 | 0.4948 | 0.7030 | -0.0251 | 0.8471 | 0.0208 | 0.1703 | 0.0683 | 0.0191 | 0.9675 |
| base_metar_intraday | 18 | 462 | 462 | 0.1469 | 0.4046 | 0.0076 | 0.2655 | 0.0054 | 0.0431 | 0.0124 | 0.0085 | 0.9762 |
| base_metar_intraday | 20 | 462 | 462 | 0.0485 | 0.2145 | 0.0005 | 0.0692 | 0.0018 | 0.0149 | 0.0041 | 0.0043 | 0.9935 |
| enhanced_metar_intraday | 6 | 462 | 462 | 2.0167 | 2.5534 | -0.5599 | 3.5839 | 0.1069 | 0.0507 | 0.0887 | 0.1317 | 0.8658 |
| enhanced_metar_intraday | 8 | 462 | 462 | 1.9693 | 2.5147 | -0.6071 | 3.5329 | 0.1044 | 0.0503 | 0.0858 | 0.1261 | 0.8853 |
| enhanced_metar_intraday | 10 | 462 | 462 | 1.6568 | 2.1294 | -0.4301 | 2.8470 | 0.0881 | 0.0480 | 0.0854 | 0.1247 | 0.9177 |
| enhanced_metar_intraday | 12 | 462 | 462 | 1.1588 | 1.5279 | -0.2091 | 2.1334 | 0.0629 | 0.0625 | 0.1269 | 0.1704 | 0.9134 |
| enhanced_metar_intraday | 14 | 462 | 462 | 0.8450 | 1.1236 | -0.1804 | 1.4269 | 0.0436 | 0.1524 | 0.1986 | 0.1319 | 0.9264 |
| enhanced_metar_intraday | 16 | 462 | 462 | 0.4834 | 0.6963 | -0.0425 | 0.8089 | 0.0207 | 0.1696 | 0.0673 | 0.0191 | 0.9654 |
| enhanced_metar_intraday | 18 | 462 | 462 | 0.1412 | 0.3928 | 0.0041 | 0.2617 | 0.0052 | 0.0412 | 0.0118 | 0.0085 | 0.9762 |
| enhanced_metar_intraday | 20 | 462 | 462 | 0.0477 | 0.2084 | 0.0015 | 0.0675 | 0.0017 | 0.0140 | 0.0040 | 0.0043 | 0.9957 |
| persistence_current_metar_max | 6 | 462 | 462 | 5.6017 | 6.5614 | -5.6017 | 25.9564 | 0.9394 | 0.9394 | 0.8918 | 0.7879 | 0.0606 |
| persistence_current_metar_max | 8 | 462 | 462 | 5.5584 | 6.5300 | -5.5584 | 25.8966 | 0.9372 | 0.9372 | 0.8896 | 0.7814 | 0.0628 |
| persistence_current_metar_max | 10 | 462 | 462 | 5.0130 | 5.8339 | -5.0130 | 25.7770 | 0.9329 | 0.9329 | 0.8831 | 0.7641 | 0.0671 |
| persistence_current_metar_max | 12 | 462 | 462 | 3.0779 | 3.5971 | -3.0779 | 25.1789 | 0.9113 | 0.9113 | 0.7944 | 0.6061 | 0.0887 |
| persistence_current_metar_max | 14 | 462 | 462 | 1.4242 | 1.8990 | -1.4242 | 20.0355 | 0.7251 | 0.7251 | 0.4351 | 0.1753 | 0.2749 |
| persistence_current_metar_max | 16 | 462 | 462 | 0.4416 | 0.8729 | -0.4416 | 9.2103 | 0.3333 | 0.3333 | 0.0758 | 0.0195 | 0.6667 |
| persistence_current_metar_max | 18 | 462 | 462 | 0.0779 | 0.4264 | -0.0779 | 1.3756 | 0.0498 | 0.0498 | 0.0130 | 0.0087 | 0.9502 |
| persistence_current_metar_max | 20 | 462 | 462 | 0.0281 | 0.2326 | -0.0281 | 0.5383 | 0.0195 | 0.0195 | 0.0043 | 0.0043 | 0.9805 |

## Rain/CB After Max Regime

| model_variant | rain_or_cb_after_max | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_metar_intraday | False | 2561 | 409 | 1.2011 | 1.9033 | -0.5522 | 1.9751 | 0.0635 | 0.0621 | 0.0745 | 0.0858 | 0.9145 |
| base_metar_intraday | True | 1135 | 219 | 0.9216 | 1.4702 | 0.2424 | 1.6230 | 0.0474 | 0.1034 | 0.1139 | 0.1114 | 0.9463 |
| enhanced_metar_intraday | False | 2561 | 409 | 1.1133 | 1.7435 | -0.4306 | 1.9687 | 0.0581 | 0.0616 | 0.0712 | 0.0817 | 0.9192 |
| enhanced_metar_intraday | True | 1135 | 219 | 0.8741 | 1.3988 | 0.1478 | 1.5261 | 0.0453 | 0.1005 | 0.1114 | 0.1075 | 0.9568 |
| persistence_current_metar_max | False | 2561 | 409 | 3.1257 | 4.6585 | -3.1257 | 17.9316 | 0.6490 | 0.6490 | 0.5400 | 0.4455 | 0.3510 |
| persistence_current_metar_max | True | 1135 | 219 | 1.5859 | 2.6282 | -1.5859 | 14.0711 | 0.5093 | 0.5093 | 0.4044 | 0.2758 | 0.4907 |
