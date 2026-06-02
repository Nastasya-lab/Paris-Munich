# Source identity policy

Every physical source has its own `source_id`. Same meteorological variable from different providers is not interchangeable.

Examples:

- `iem.metar.archive.EDDM`
- `awc.metar.live.EDDM`
- `dwd.10min.air_temperature.01262`
- `open_meteo.previous_runs`
- `dwd.mosmix.forecast_as_issued`

Training, backtest, and live predictions must preserve source IDs in lineage and forecast logs.

## Runtime compatibility statuses

Live forecasts must audit each feature role against the source profile used during training.

| Status | Meaning | Forecast effect |
| --- | --- | --- |
| `exact_match` | Runtime source is the same `source_id` as training. | No warning. |
| `known_compatible` | Runtime source differs, but is explicitly allowed as a compatible operational substitute. | Forecast remains usable; add caution and monitor outcomes separately. |
| `unknown_mismatch` | Runtime source is allowed for the role, but compatibility has not been validated. | Forecast is `degraded`. |
| `forbidden_mismatch` | Runtime source is not allowed for the trained feature role. | Forecast is `invalid` and should be rejected by acceptance checks. |
| `missing` | Source is absent in the as-of feature view. | Handled by freshness/missingness checks, not as source mismatch by itself. |

Current known-compatible substitutions:

- METAR: `awc.metar.live.EDDM` may substitute `iem.metar.archive.EDDM` for live operations.
- TAF: `awc.taf.live.EDDM` may substitute `iem.taf.archive.EDDM` for live operations.
- NWP: `open_meteo.live.icon_d2` may substitute `open_meteo.single_run.icon_d2` for live operations, but outcomes must be monitored separately.

Known-compatible does not mean identical. Outcome monitoring must keep compatibility status so we can compare exact, compatible, unknown, and missing-source performance.
