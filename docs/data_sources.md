# Data sources

The source registry lives in `config/data_sources.yaml`.

Truth target source for EDDM is DWD 10-minute air temperature station `01262` whenever available. DWD hourly and daily products are target fallbacks only.

Feature sources include IEM/AWC METAR, IEM/AWC TAF, honest forecast-as-issued NWP archives, and MOSMIX issued forecasts if archived.

Historical weather endpoints that do not preserve issued forecast runs are not allowed as training features.

Current workspace ingestion:

- DWD station `01262` 10-minute temperature: 2020-01-01 to 2025-12-31 UTC archive parsed.
- IEM METAR EDDM: 2020-2025 archive parsed.
- IEM TAF EDDM: endpoint available, but the tested 2020-2025 request returned no rows; pipeline keeps `taf_missing`.
- Open-Meteo live ICON-D2: operational forward archiver implemented, but not yet used in historical backtest.
- AWC live METAR/TAF: operational forward archiver implemented. AWC Data API is live/recent only for this project and is not a multi-year historical TAF source.
