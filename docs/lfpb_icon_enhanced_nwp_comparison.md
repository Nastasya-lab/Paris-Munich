# LFPB enhanced ICON-D2 NWP feature comparison

- created: `2026-06-10T08:32:42.684846+00:00`
- period: `2025-07-27` to `2026-05-30`
- rows joined: `2324`
- days joined: `304`
- recommendation: `do_not_promote_yet`
- reason: Enhanced weather features did not pass the promotion gate.

## Summary

| model_variant | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_icon_ensemble | 458 | 61 | 0.5455 | 0.8284 | 0.1248 | 2.3900 | 0.0271 | 0.9629 |
| base_icon_ml | 458 | 61 | 0.5455 | 0.8284 | 0.1248 | 2.3900 | 0.0279 | 0.9629 |
| enhanced_icon_ensemble | 458 | 61 | 0.5517 | 0.8265 | 0.1261 | 2.3114 | 0.0258 | 0.9607 |
| enhanced_icon_ml | 458 | 61 | 0.5517 | 0.8265 | 0.1261 | 2.3114 | 0.0266 | 0.9607 |
| raw_icon_residual_distribution | 458 | 61 | 0.6449 | 0.8319 | 0.4247 | 1.1511 | 0.0625 | 0.9716 |

## By Hour

| model_variant | local_issue_hour | rows | distinct_days | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps | coverage_80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_icon_ensemble | 6 | 43 | 43 | 0.8935 | 1.2324 | 0.1221 | 5.2313 | 0.0487 | 0.9535 |
| base_icon_ensemble | 8 | 55 | 55 | 0.9986 | 1.2997 | 0.1113 | 5.4605 | 0.0494 | 0.9273 |
| base_icon_ensemble | 10 | 55 | 55 | 0.8597 | 1.0599 | 0.2725 | 3.8987 | 0.0437 | 0.9455 |
| base_icon_ensemble | 12 | 61 | 61 | 0.7091 | 0.8687 | 0.2224 | 2.0028 | 0.0377 | 0.8852 |
| base_icon_ensemble | 14 | 61 | 61 | 0.5052 | 0.6621 | 0.1465 | 1.2692 | 0.0267 | 1.0000 |
| base_icon_ensemble | 16 | 61 | 61 | 0.3816 | 0.5035 | 0.0056 | 1.9232 | 0.0160 | 0.9836 |
| base_icon_ensemble | 18 | 61 | 61 | 0.1563 | 0.2956 | 0.0923 | 0.5887 | 0.0042 | 1.0000 |
| base_icon_ensemble | 20 | 61 | 61 | 0.0383 | 0.1088 | 0.0383 | 0.0341 | 0.0005 | 1.0000 |
| base_icon_ml | 6 | 43 | 43 | 0.8935 | 1.2324 | 0.1221 | 5.2313 | 0.0487 | 0.9535 |
| base_icon_ml | 8 | 55 | 55 | 0.9986 | 1.2997 | 0.1113 | 5.4605 | 0.0500 | 0.9273 |
| base_icon_ml | 10 | 55 | 55 | 0.8597 | 1.0599 | 0.2725 | 3.8987 | 0.0445 | 0.9455 |
| base_icon_ml | 12 | 61 | 61 | 0.7091 | 0.8687 | 0.2224 | 2.0028 | 0.0388 | 0.8852 |
| base_icon_ml | 14 | 61 | 61 | 0.5052 | 0.6621 | 0.1465 | 1.2692 | 0.0286 | 1.0000 |
| base_icon_ml | 16 | 61 | 61 | 0.3816 | 0.5035 | 0.0056 | 1.9232 | 0.0173 | 0.9836 |
| base_icon_ml | 18 | 61 | 61 | 0.1563 | 0.2956 | 0.0923 | 0.5887 | 0.0044 | 1.0000 |
| base_icon_ml | 20 | 61 | 61 | 0.0383 | 0.1088 | 0.0383 | 0.0341 | 0.0006 | 1.0000 |
| enhanced_icon_ensemble | 6 | 43 | 43 | 0.9204 | 1.2486 | 0.1554 | 5.7287 | 0.0476 | 0.9535 |
| enhanced_icon_ensemble | 8 | 55 | 55 | 1.0012 | 1.2698 | 0.1143 | 3.8880 | 0.0448 | 0.9273 |
| enhanced_icon_ensemble | 10 | 55 | 55 | 0.8884 | 1.0666 | 0.2484 | 3.8269 | 0.0399 | 0.9273 |
| enhanced_icon_ensemble | 12 | 61 | 61 | 0.7018 | 0.8690 | 0.2150 | 2.6470 | 0.0365 | 0.9016 |
| enhanced_icon_ensemble | 14 | 61 | 61 | 0.5117 | 0.6553 | 0.1279 | 1.5676 | 0.0263 | 0.9836 |
| enhanced_icon_ensemble | 16 | 61 | 61 | 0.3855 | 0.5134 | 0.0310 | 1.5422 | 0.0166 | 0.9836 |
| enhanced_icon_ensemble | 18 | 61 | 61 | 0.1508 | 0.2952 | 0.0963 | 0.5718 | 0.0039 | 1.0000 |
| enhanced_icon_ensemble | 20 | 61 | 61 | 0.0401 | 0.1230 | 0.0401 | 0.0315 | 0.0006 | 1.0000 |
| enhanced_icon_ml | 6 | 43 | 43 | 0.9204 | 1.2486 | 0.1554 | 5.7287 | 0.0475 | 0.9535 |
| enhanced_icon_ml | 8 | 55 | 55 | 1.0012 | 1.2698 | 0.1143 | 3.8880 | 0.0452 | 0.9273 |
| enhanced_icon_ml | 10 | 55 | 55 | 0.8884 | 1.0666 | 0.2484 | 3.8269 | 0.0406 | 0.9273 |
| enhanced_icon_ml | 12 | 61 | 61 | 0.7018 | 0.8690 | 0.2150 | 2.6470 | 0.0376 | 0.9016 |
| enhanced_icon_ml | 14 | 61 | 61 | 0.5117 | 0.6553 | 0.1279 | 1.5676 | 0.0283 | 0.9836 |
| enhanced_icon_ml | 16 | 61 | 61 | 0.3855 | 0.5134 | 0.0310 | 1.5422 | 0.0180 | 0.9836 |
| enhanced_icon_ml | 18 | 61 | 61 | 0.1508 | 0.2952 | 0.0963 | 0.5718 | 0.0041 | 1.0000 |
| enhanced_icon_ml | 20 | 61 | 61 | 0.0401 | 0.1230 | 0.0401 | 0.0315 | 0.0006 | 1.0000 |
| raw_icon_residual_distribution | 6 | 43 | 43 | 0.5753 | 0.7127 | 0.1519 | 1.2298 | 0.0761 | 1.0000 |
| raw_icon_residual_distribution | 8 | 55 | 55 | 0.6312 | 0.7517 | 0.3010 | 1.2597 | 0.0708 | 1.0000 |
| raw_icon_residual_distribution | 10 | 55 | 55 | 0.6403 | 0.7604 | 0.3311 | 1.2553 | 0.0681 | 1.0000 |
| raw_icon_residual_distribution | 12 | 61 | 61 | 0.6439 | 0.7964 | 0.3537 | 1.2579 | 0.0705 | 0.9836 |
| raw_icon_residual_distribution | 14 | 61 | 61 | 0.6699 | 0.8410 | 0.4939 | 1.2225 | 0.0633 | 0.9508 |
| raw_icon_residual_distribution | 16 | 61 | 61 | 0.6457 | 0.8048 | 0.5636 | 1.0299 | 0.0573 | 0.9672 |
| raw_icon_residual_distribution | 18 | 61 | 61 | 0.6187 | 0.7387 | 0.6187 | 0.8057 | 0.0452 | 0.9344 |
| raw_icon_residual_distribution | 20 | 61 | 61 | 0.7116 | 1.1279 | 0.4821 | 1.1919 | 0.0538 | 0.9508 |

## Limitations

- This is a partial-window stress test because the full enhanced ICON-D2 archive is not downloaded yet.
- Do not promote based on this report alone; rerun after the enhanced archive covers the full 2025-07-27..2026-05-30 overlap.
- Enhanced future aggregates are relative to model availability time, matching the existing NWP archive convention.
