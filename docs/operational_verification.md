# Operational verification

This project keeps the production forecast and experimental shadow forecasts separate.
The production distribution remains `production_champion`; shadow distributions are
logged and scored only after the DWD truth layer publishes the daily Tmax.

## Tables

`data/reports/forecast_monitoring.parquet` remains backward-compatible and contains
one row per logged forecast for the production champion.

`data/reports/forecast_variant_monitoring.parquet` contains one row per forecast
variant and is the primary table for champion-vs-shadow evaluation.

Current variants:

| forecast_variant | Meaning |
| --- | --- |
| `production_champion` | The operational distribution returned to users. |
| `base_prior` | Full-day prior before same-day intraday update. |
| `shadow_seasonal_intraday` | Experimental seasonal intraday challenger. |

## Metrics

Each variant is scored against the same DWD truth value:

- expected Tmax error;
- median Tmax error;
- negative log likelihood of the actual integer bin;
- discrete CRPS;
- Brier scores for `Tmax >= 20/25/30C`;
- probability assigned to the actual integer bin.

`docs/outcome_analysis.md` includes aggregate variant scores plus a paired
`production_champion` versus `shadow_seasonal_intraday` comparison on matching
`forecast_id` rows.

## Promotion Policy

Shadow variants must not replace the champion from a single day. Promotion requires
a useful live sample, stable paired improvements, and no obvious degradation by
issue hour, season, source availability, or weather regime.
