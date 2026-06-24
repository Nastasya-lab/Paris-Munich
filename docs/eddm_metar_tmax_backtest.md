# EDDM METAR Tmax target backtest

- created: `2026-06-24T05:27:05.715117+00:00`
- production changed: `True`
- target: daily maximum temperature reported by EDDM METAR
- period: `2025-07-27` to `2026-05-30`
- rows: `2464`
- days: `308`
- spatial enabled: `True`
- recommendation: `keep_as_research_candidate`

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_munich_production_core_on_metar | 496 | 62 | 0.5921 | 0.8258 | -0.0245 | 1.1571 | 0.0047 | 0.0789 | 0.0486 | 0.0421 |
| eddm_metar_icon_d2 | 496 | 62 | 0.7375 | 1.1624 | 0.0680 | 1.1925 | 0.0293 | 0.0527 | 0.0481 | 0.0350 |
| eddm_metar_icon_d2_spatial | 496 | 62 | 0.7816 | 1.2666 | 0.0042 | 1.1522 | 0.0290 | 0.0517 | 0.0430 | 0.0359 |

## 10-17 Local Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_munich_production_core_on_metar | 248 | 62 | 0.6286 | 0.8525 | -0.0515 | 1.2337 | 0.0052 | 0.1066 | 0.0902 | 0.0708 |
| eddm_metar_icon_d2 | 248 | 62 | 0.6045 | 0.8029 | 0.2558 | 1.2536 | 0.0298 | 0.0890 | 0.0884 | 0.0575 |
| eddm_metar_icon_d2_spatial | 248 | 62 | 0.5890 | 0.7971 | 0.1700 | 1.1933 | 0.0283 | 0.0882 | 0.0769 | 0.0579 |

## By Local Issue Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | brier_upside_ge_1c | brier_upside_ge_2c | brier_upside_ge_3c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_munich_production_core_on_metar | 6 | 62 | 62 | 0.7980 | 1.0620 | -0.2793 | 1.8229 | 0.0070 | 0.0029 | 0.0099 | 0.0186 |
| current_munich_production_core_on_metar | 8 | 62 | 62 | 0.8121 | 1.0698 | -0.3049 | 1.8096 | 0.0070 | 0.0049 | 0.0100 | 0.0347 |
| current_munich_production_core_on_metar | 10 | 62 | 62 | 0.7984 | 1.0624 | -0.2782 | 1.8226 | 0.0070 | 0.0049 | 0.0241 | 0.0836 |
| current_munich_production_core_on_metar | 12 | 62 | 62 | 0.6785 | 0.8830 | -0.2094 | 1.3088 | 0.0059 | 0.0507 | 0.1278 | 0.1441 |
| current_munich_production_core_on_metar | 14 | 62 | 62 | 0.5545 | 0.7533 | -0.0017 | 1.0610 | 0.0047 | 0.1510 | 0.1690 | 0.0474 |
| current_munich_production_core_on_metar | 16 | 62 | 62 | 0.4830 | 0.6565 | 0.2835 | 0.7423 | 0.0033 | 0.2197 | 0.0399 | 0.0083 |
| current_munich_production_core_on_metar | 18 | 62 | 62 | 0.3517 | 0.3983 | 0.3334 | 0.3922 | 0.0014 | 0.1113 | 0.0046 | 0.0001 |
| current_munich_production_core_on_metar | 20 | 62 | 62 | 0.2603 | 0.3432 | 0.2603 | 0.2974 | 0.0011 | 0.0859 | 0.0032 | 0.0001 |
| eddm_metar_icon_d2 | 6 | 62 | 62 | 1.8751 | 2.3889 | -0.7508 | 2.2186 | 0.0638 | 0.0040 | 0.0160 | 0.0161 |
| eddm_metar_icon_d2 | 8 | 62 | 62 | 1.2439 | 1.5611 | -0.0690 | 1.9736 | 0.0470 | 0.0040 | 0.0128 | 0.0339 |
| eddm_metar_icon_d2 | 10 | 62 | 62 | 0.7852 | 1.0676 | 0.2974 | 1.7289 | 0.0447 | 0.0043 | 0.0142 | 0.0733 |
| eddm_metar_icon_d2 | 12 | 62 | 62 | 0.6327 | 0.8216 | 0.3307 | 1.3794 | 0.0333 | 0.0346 | 0.1121 | 0.1087 |
| eddm_metar_icon_d2 | 14 | 62 | 62 | 0.5426 | 0.6904 | 0.1432 | 1.2928 | 0.0273 | 0.1285 | 0.2121 | 0.0435 |
| eddm_metar_icon_d2 | 16 | 62 | 62 | 0.4574 | 0.5360 | 0.2519 | 0.6134 | 0.0140 | 0.1888 | 0.0151 | 0.0044 |
| eddm_metar_icon_d2 | 18 | 62 | 62 | 0.2057 | 0.2385 | 0.1834 | 0.2045 | 0.0030 | 0.0410 | 0.0015 | 0.0001 |
| eddm_metar_icon_d2 | 20 | 62 | 62 | 0.1576 | 0.1731 | 0.1576 | 0.1288 | 0.0013 | 0.0167 | 0.0013 | 0.0001 |
| eddm_metar_icon_d2_spatial | 6 | 62 | 62 | 2.1057 | 2.6448 | -0.8241 | 2.1187 | 0.0652 | 0.0039 | 0.0171 | 0.0187 |
| eddm_metar_icon_d2_spatial | 8 | 62 | 62 | 1.4373 | 1.7939 | -0.1517 | 2.0044 | 0.0499 | 0.0039 | 0.0164 | 0.0365 |
| eddm_metar_icon_d2_spatial | 10 | 62 | 62 | 0.8284 | 1.1088 | 0.1753 | 1.7089 | 0.0448 | 0.0078 | 0.0150 | 0.0750 |
| eddm_metar_icon_d2_spatial | 12 | 62 | 62 | 0.6304 | 0.8143 | 0.2184 | 1.3992 | 0.0327 | 0.0374 | 0.1162 | 0.1179 |
| eddm_metar_icon_d2_spatial | 14 | 62 | 62 | 0.4959 | 0.6360 | 0.1127 | 1.0915 | 0.0233 | 0.1365 | 0.1643 | 0.0358 |
| eddm_metar_icon_d2_spatial | 16 | 62 | 62 | 0.4014 | 0.4942 | 0.1736 | 0.5736 | 0.0125 | 0.1711 | 0.0119 | 0.0031 |
| eddm_metar_icon_d2_spatial | 18 | 62 | 62 | 0.1973 | 0.2256 | 0.1731 | 0.1931 | 0.0027 | 0.0362 | 0.0014 | 0.0001 |
| eddm_metar_icon_d2_spatial | 20 | 62 | 62 | 0.1567 | 0.1729 | 0.1567 | 0.1279 | 0.0013 | 0.0166 | 0.0013 | 0.0001 |
