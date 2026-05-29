from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import requests

from weather_tmax_bot.utils.hashing import stable_hash
from weather_tmax_bot.utils.time import local_day_bounds_utc


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo_live_extract(
    airport_icao: str,
    latitude: float,
    longitude: float,
    target_date_local: date,
    timezone_name: str,
    model_name: str = "icon_d2",
) -> pd.DataFrame:
    ingest_time = datetime.now(timezone.utc)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "dew_point_2m",
            "relative_humidity_2m",
            "cloud_cover",
            "precipitation",
            "shortwave_radiation",
            "wind_speed_10m",
            "wind_gusts_10m",
            "surface_pressure",
        ],
        "models": model_name,
        "timezone": "UTC",
        "forecast_days": 7,
    }
    response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["valid_time_utc"] = pd.to_datetime(df["time"], utc=True)
    start_utc, end_utc = local_day_bounds_utc(target_date_local, timezone_name)
    day = df[(df["valid_time_utc"] >= pd.Timestamp(start_utc)) & (df["valid_time_utc"] <= pd.Timestamp(end_utc))]
    if day.empty:
        return pd.DataFrame()
    row = {
        "airport_icao": airport_icao,
        "target_date_local": target_date_local.isoformat(),
        "model_name": f"open_meteo.{model_name}",
        "model_run_time_utc": None,
        "model_availability_time_utc": ingest_time,
        "knowledge_time_utc": ingest_time,
        "forecast_reference_time": ingest_time,
        "forecast_horizon_hours": None,
        "model_tmax_c": pd.to_numeric(day.get("temperature_2m"), errors="coerce").max(),
        "model_temp_at_08_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 8),
        "model_temp_at_11_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 11),
        "model_temp_at_14_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 14),
        "model_temp_at_17_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 17),
        "model_cloud_cover_mean": pd.to_numeric(day.get("cloud_cover"), errors="coerce").mean(),
        "model_precip_sum": pd.to_numeric(day.get("precipitation"), errors="coerce").sum(),
        "model_shortwave_radiation_sum": pd.to_numeric(day.get("shortwave_radiation"), errors="coerce").sum(),
        "model_wind_speed_max": pd.to_numeric(day.get("wind_speed_10m"), errors="coerce").max(),
        "model_gust_max": pd.to_numeric(day.get("wind_gusts_10m"), errors="coerce").max(),
        "model_pressure_mean": pd.to_numeric(day.get("surface_pressure"), errors="coerce").mean(),
        "model_dewpoint_mean": pd.to_numeric(day.get("dew_point_2m"), errors="coerce").mean(),
        "model_relative_humidity_mean": pd.to_numeric(day.get("relative_humidity_2m"), errors="coerce").mean(),
        "nearest_gridpoint_value": pd.to_numeric(day.get("temperature_2m"), errors="coerce").max(),
        "source_id": f"open_meteo.live.{model_name}",
        "source_version": "open_meteo_forecast_api_live",
        "ingest_time_utc": ingest_time,
        "raw_file_reference": OPEN_METEO_FORECAST_URL,
        "raw_record_hash": stable_hash(payload),
        "quality_flag": "forward_archive_not_historical_backtest",
    }
    return pd.DataFrame([row])


def _value_at_local_hour(day: pd.DataFrame, column: str, target_date: date, timezone_name: str, hour: int):
    local = day["valid_time_utc"].dt.tz_convert(timezone_name)
    mask = (local.dt.date == target_date) & (local.dt.hour == hour)
    values = pd.to_numeric(day.loc[mask, column], errors="coerce")
    return None if values.empty else values.iloc[0]
