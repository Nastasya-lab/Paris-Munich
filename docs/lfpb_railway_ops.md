# LFPB Railway Operations

Paris/LFPB runs independently from Munich/EDDM and predicts daily `METAR_Tmax`: the maximum temperature reported by LFPB METAR during the local day.

## Services

Use existing Railway service slots if the project service limit is tight.

Recommended service names:

- `lfpb-forecast-cron`
- `lfpb-metar-cron`

The entrypoint detects these names automatically:

- `lfpb-forecast-cron` runs `scripts/53_lfpb_forecast_job.py`
- `lfpb-metar-cron` runs `scripts/54_lfpb_metar_event_job.py`

Alternatively set `WEATHER_TMAX_JOB` explicitly:

- `WEATHER_TMAX_JOB=lfpb-forecast`
- `WEATHER_TMAX_JOB=lfpb-metar-event`

## Telegram

Paris can use a separate Telegram bot/chat without affecting Munich:

```text
TELEGRAM_BOT_TOKEN_LFPB=<Paris bot token>
TELEGRAM_CHAT_ID_LFPB=<Paris chat id>
WEATHER_TMAX_ENABLE_TELEGRAM=1
```

The LFPB job copies these values into `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` only inside the LFPB process.

## Schedules

Main ICON-aware forecasts:

```text
01:30, 04:30, 07:30, 10:30, 13:30, 16:30, 19:30, 22:30 UTC
```

METAR event polling:

```text
Start around :00 and :30 METAR publication windows, with polling every 30 seconds for up to 10 minutes.
```

Useful environment:

```text
METAR_POLL_TIMEOUT_SECONDS=600
METAR_POLL_INTERVAL_SECONDS=30
```

## Manual Commands

One forecast:

```powershell
python scripts/53_lfpb_forecast_job.py
```

One METAR polling cycle:

```powershell
python scripts/54_lfpb_metar_event_job.py --poll-timeout-seconds 600 --poll-interval-seconds 30
```

Direct predictor:

```powershell
python scripts/48_predict_lfpb_metar_tmax.py --airport LFPB --target-date 2026-06-09 --issue-time now --auto-refresh --refresh-nwp --notify --model-path data/models/lfpb_metar_tmax_icon_d2_v1.joblib --metadata-path data/models/lfpb_metar_tmax_icon_d2_v1.metadata.json
```

## Model

Active Paris candidate:

```text
data/models/lfpb_metar_tmax_icon_d2_v1.joblib
```

This is an ensemble of:

- calibrated METAR remaining-upside ML model;
- ICON-D2 residual distribution.

It requires live or archived Open-Meteo ICON-D2 features.
