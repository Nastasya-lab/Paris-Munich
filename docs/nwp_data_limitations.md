# NWP data limitations

The MVP does not pretend to have historical forecast-as-issued NWP unless files are supplied or accumulated.

Accepted NWP data must preserve:

- model name;
- model run time;
- model availability time;
- forecast reference time;
- source ID;
- raw reference or extract hash.

Reanalysis and seamless historical weather endpoints are not allowed as substitutes in the main backtest.

Until issued archives are available, honest backtests are limited to DWD truth plus climatology and available METAR/TAF feature periods. NWP metrics from the operational archiver are forward-test metrics.

Implemented forward archiver:

```bash
python scripts/11_archive_operational_data.py --airport EDDM --target-date YYYY-MM-DD --provider open-meteo
```

This writes `data/forecasts/open_meteo_archive.parquet`. It is not a historical backtest source before the rows are actually accumulated as live forecasts.

Open-Meteo Single Runs may be used for historical backfill only when each row preserves the concrete model `run` time and an explicit `model_availability_time_utc`. The project keeps this source separate as `open_meteo.single_run.icon_d2` in `data/forecasts/open_meteo_single_runs_icon_d2.parquet`; unlike seamless historical weather, it represents a specific issued model run.

For ICON-D2 backfill the MVP applies a conservative three-hour availability latency. This deliberately avoids exposing a newly initialized run to an earlier training row before it would reasonably have been available operationally.

Review a small backfill before requesting a multi-month archive:

```bash
python scripts/34_backfill_open_meteo_single_runs.py --start-date 2026-05-01 --end-date 2026-05-31 --max-runs 4 --dry-run
```

Remove `--dry-run` only after reviewing the summary. A full multi-month backfill may require hundreds of API requests.

Current archive state:

- First Open-Meteo live ICON-D2 extract for EDDM was archived for target date `2026-05-29`.
- The row is available for live/as-of prediction after its `knowledge_time_utc`.
- It is explicitly excluded from historical training until a DWD truth outcome exists.
