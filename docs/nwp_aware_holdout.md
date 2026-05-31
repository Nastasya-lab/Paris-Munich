# NWP-aware holdout comparison

Experimental comparison using only rows where Open-Meteo Single Runs ICON-D2 is available.

- train period: `2025-09-01` to `2025-11-30`
- test period: `2025-12-01` to `2025-12-31`
- train rows: `637`
- test rows: `217`

| model_variant | rows | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps |
| --- | --- | --- | --- | --- | --- | --- |
| raw_nwp_model_tmax | 217.0 | 1.2092165898617513 | 1.7711200861169762 | -0.6294930875576038 | nan | nan |
| quantile_with_nwp | 217.0 | 3.24104324833565 | 3.7747964951698543 | 2.7637546979423604 | 6.8197108786487854 | 0.9411986226328214 |
| quantile_without_nwp | 217.0 | 3.837916187976368 | 4.465029511856287 | 2.6523780255049463 | 9.135466754885035 | 0.9343237705032075 |

This is a short seasonal slice, not enough for final promotion by itself.
It is useful for deciding whether to continue toward a production NWP-aware model.