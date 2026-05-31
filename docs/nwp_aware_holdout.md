# NWP-aware holdout comparison

Experimental comparison using only rows where Open-Meteo Single Runs ICON-D2 is available.

- train period: `2025-05-31` to `2025-12-31`
- test period: `2026-01-01` to `2026-05-27`
- train rows: `1470`
- test rows: `1029`

| model_variant | rows | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- | --- |
| bias_corrected_nwp_residual_distribution | 1029.0 | 0.7074662225082066 | 0.9420635785396765 | 0.12781171635645297 | 1.4040034420503755 | 0.9776920183356095 |
| raw_nwp_model_tmax | 1029.0 | 0.7699708454810498 | 1.018244645178359 | -0.4092322643343052 | nan | nan |
| quantile_with_nwp | 1029.0 | 5.1046796084363315 | 5.763541231267836 | 4.903677137109374 | 5.132602241652077 | 0.9057083062455281 |
| quantile_without_nwp | 1029.0 | 6.892797230784741 | 8.379929631450477 | 6.059274944574528 | 10.870519912624214 | 0.8944790919640067 |

This is a short seasonal slice, not enough for final promotion by itself.
It is useful for deciding whether to continue toward a production NWP-aware model.