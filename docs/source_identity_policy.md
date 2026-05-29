# Source identity policy

Every physical source has its own `source_id`. Same meteorological variable from different providers is not interchangeable.

Examples:

- `iem.metar.archive.EDDM`
- `awc.metar.live.EDDM`
- `dwd.10min.air_temperature.01262`
- `open_meteo.previous_runs`
- `dwd.mosmix.forecast_as_issued`

Training, backtest, and live predictions must preserve source IDs in lineage and forecast logs.
