from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from weather_tmax_bot.utils.hashing import stable_hash
from weather_tmax_bot.utils.time import local_day_bounds_utc


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_SINGLE_RUN_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "precipitation",
    "precipitation_probability",
    "rain",
    "showers",
    "weather_code",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "sunshine_duration",
    "wind_speed_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "cape",
    "lifted_index",
]


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
        "hourly": HOURLY_VARIABLES,
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
    row = _aggregate_open_meteo_day(
        payload=payload,
        day=day,
        airport_icao=airport_icao,
        target_date_local=target_date_local,
        timezone_name=timezone_name,
        model_name=f"open_meteo.{model_name}",
        model_run_time_utc=None,
        model_availability_time_utc=ingest_time,
        knowledge_time_utc=ingest_time,
        source_id=f"open_meteo.live.{model_name}",
        source_version="open_meteo_forecast_api_live",
        raw_file_reference=OPEN_METEO_FORECAST_URL,
        quality_flag="forward_archive_not_historical_backtest",
        ingest_time_utc=ingest_time,
    )
    return pd.DataFrame([row])


def fetch_open_meteo_single_run_extract(
    airport_icao: str,
    latitude: float,
    longitude: float,
    run_time_utc: datetime,
    target_dates_local: list[date],
    timezone_name: str,
    model_name: str = "icon_d2",
    forecast_days: int = 3,
    availability_latency_hours: float = 3.0,
) -> pd.DataFrame:
    """Fetch one concrete Open-Meteo run and aggregate it by local target day."""
    run = run_time_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    ingest_time = datetime.now(timezone.utc)
    availability = run + timedelta(hours=availability_latency_hours)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": HOURLY_VARIABLES,
        "models": model_name,
        "timezone": "UTC",
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": forecast_days,
    }
    response = requests.get(OPEN_METEO_SINGLE_RUN_URL, params=params, timeout=60)
    if response.status_code in {400, 404}:
        return pd.DataFrame()
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly", {})
    if not hourly or "time" not in hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["valid_time_utc"] = pd.to_datetime(df["time"], utc=True)
    rows = []
    for target_date_local in target_dates_local:
        start_utc, end_utc = local_day_bounds_utc(target_date_local, timezone_name)
        day = df[(df["valid_time_utc"] >= pd.Timestamp(start_utc)) & (df["valid_time_utc"] <= pd.Timestamp(end_utc))]
        if day.empty:
            continue
        rows.append(
            _aggregate_open_meteo_day(
                payload=payload,
                day=day,
                airport_icao=airport_icao,
                target_date_local=target_date_local,
                timezone_name=timezone_name,
                model_name=f"open_meteo.{model_name}",
                model_run_time_utc=run,
                model_availability_time_utc=availability,
                knowledge_time_utc=availability,
                source_id=f"open_meteo.single_run.{model_name}",
                source_version="open_meteo_single_runs_api",
                raw_file_reference=f"{OPEN_METEO_SINGLE_RUN_URL}?run={run.strftime('%Y-%m-%dT%H:%M')}",
                quality_flag="historical_forecast_as_issued_single_run",
                ingest_time_utc=ingest_time,
            )
        )
    return pd.DataFrame(rows)


def _aggregate_open_meteo_day(
    *,
    payload: dict,
    day: pd.DataFrame,
    airport_icao: str,
    target_date_local: date,
    timezone_name: str,
    model_name: str,
    model_run_time_utc: datetime | None,
    model_availability_time_utc: datetime,
    knowledge_time_utc: datetime,
    source_id: str,
    source_version: str,
    raw_file_reference: str,
    quality_flag: str,
    ingest_time_utc: datetime,
) -> dict:
    horizon = None
    if model_run_time_utc is not None:
        horizon = (day["valid_time_utc"].min().to_pydatetime() - model_run_time_utc).total_seconds() / 3600
    remaining_day = day[day["valid_time_utc"] >= pd.Timestamp(model_availability_time_utc)]
    return {
        "airport_icao": airport_icao,
        "target_date_local": target_date_local.isoformat(),
        "model_name": model_name,
        "model_run_time_utc": model_run_time_utc,
        "model_availability_time_utc": model_availability_time_utc,
        "knowledge_time_utc": knowledge_time_utc,
        "forecast_reference_time": model_run_time_utc or ingest_time_utc,
        "forecast_horizon_hours": horizon,
        "model_tmax_c": pd.to_numeric(day.get("temperature_2m"), errors="coerce").max(),
        "model_temp_at_08_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 8),
        "model_temp_at_11_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 11),
        "model_temp_at_14_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 14),
        "model_temp_at_17_local": _value_at_local_hour(day, "temperature_2m", target_date_local, timezone_name, 17),
        "model_cloud_cover_mean": pd.to_numeric(day.get("cloud_cover"), errors="coerce").mean(),
        "model_cloud_cover_max": pd.to_numeric(day.get("cloud_cover"), errors="coerce").max(),
        "model_low_cloud_cover_mean": pd.to_numeric(day.get("cloud_cover_low"), errors="coerce").mean(),
        "model_low_cloud_cover_max": pd.to_numeric(day.get("cloud_cover_low"), errors="coerce").max(),
        "model_mid_cloud_cover_mean": pd.to_numeric(day.get("cloud_cover_mid"), errors="coerce").mean(),
        "model_mid_cloud_cover_max": pd.to_numeric(day.get("cloud_cover_mid"), errors="coerce").max(),
        "model_high_cloud_cover_mean": pd.to_numeric(day.get("cloud_cover_high"), errors="coerce").mean(),
        "model_high_cloud_cover_max": pd.to_numeric(day.get("cloud_cover_high"), errors="coerce").max(),
        "model_precip_sum": pd.to_numeric(day.get("precipitation"), errors="coerce").sum(),
        "model_precip_probability_max": pd.to_numeric(day.get("precipitation_probability"), errors="coerce").max(),
        "model_precip_hours": _positive_hours(day, "precipitation"),
        "model_rain_sum": pd.to_numeric(day.get("rain"), errors="coerce").sum(),
        "model_rain_hours": _positive_hours(day, "rain"),
        "model_showers_sum": pd.to_numeric(day.get("showers"), errors="coerce").sum(),
        "model_showers_hours": _positive_hours(day, "showers"),
        "model_weather_code_max": pd.to_numeric(day.get("weather_code"), errors="coerce").max(),
        "model_has_thunderstorm_code": _has_weather_code(day, {95, 96, 99}),
        "model_has_rain_code": _has_weather_code(day, set(range(51, 68)) | set(range(80, 83)) | {95, 96, 99}),
        "model_shortwave_radiation_sum": pd.to_numeric(day.get("shortwave_radiation"), errors="coerce").sum(),
        "model_direct_radiation_sum": pd.to_numeric(day.get("direct_radiation"), errors="coerce").sum(),
        "model_diffuse_radiation_sum": pd.to_numeric(day.get("diffuse_radiation"), errors="coerce").sum(),
        "model_sunshine_duration_sum": pd.to_numeric(day.get("sunshine_duration"), errors="coerce").sum(),
        "model_cape_max": pd.to_numeric(day.get("cape"), errors="coerce").max(),
        "model_lifted_index_min": pd.to_numeric(day.get("lifted_index"), errors="coerce").min(),
        "model_wind_speed_max": pd.to_numeric(day.get("wind_speed_10m"), errors="coerce").max(),
        "model_gust_max": pd.to_numeric(day.get("wind_gusts_10m"), errors="coerce").max(),
        "model_pressure_mean": pd.to_numeric(day.get("surface_pressure"), errors="coerce").mean(),
        "model_dewpoint_mean": pd.to_numeric(day.get("dew_point_2m"), errors="coerce").mean(),
        "model_relative_humidity_mean": pd.to_numeric(day.get("relative_humidity_2m"), errors="coerce").mean(),
        "model_future_temp_max_c": pd.to_numeric(remaining_day.get("temperature_2m"), errors="coerce").max(),
        "model_future_cloud_cover_mean": pd.to_numeric(remaining_day.get("cloud_cover"), errors="coerce").mean(),
        "model_future_cloud_cover_max": pd.to_numeric(remaining_day.get("cloud_cover"), errors="coerce").max(),
        "model_future_low_cloud_cover_mean": pd.to_numeric(remaining_day.get("cloud_cover_low"), errors="coerce").mean(),
        "model_future_low_cloud_cover_max": pd.to_numeric(remaining_day.get("cloud_cover_low"), errors="coerce").max(),
        "model_future_mid_cloud_cover_mean": pd.to_numeric(remaining_day.get("cloud_cover_mid"), errors="coerce").mean(),
        "model_future_mid_cloud_cover_max": pd.to_numeric(remaining_day.get("cloud_cover_mid"), errors="coerce").max(),
        "model_future_high_cloud_cover_mean": pd.to_numeric(remaining_day.get("cloud_cover_high"), errors="coerce").mean(),
        "model_future_high_cloud_cover_max": pd.to_numeric(remaining_day.get("cloud_cover_high"), errors="coerce").max(),
        "model_future_precip_sum": pd.to_numeric(remaining_day.get("precipitation"), errors="coerce").sum(),
        "model_future_precip_probability_max": pd.to_numeric(remaining_day.get("precipitation_probability"), errors="coerce").max(),
        "model_future_precip_hours": _positive_hours(remaining_day, "precipitation"),
        "model_future_rain_sum": pd.to_numeric(remaining_day.get("rain"), errors="coerce").sum(),
        "model_future_rain_hours": _positive_hours(remaining_day, "rain"),
        "model_future_showers_sum": pd.to_numeric(remaining_day.get("showers"), errors="coerce").sum(),
        "model_future_showers_hours": _positive_hours(remaining_day, "showers"),
        "model_future_has_thunderstorm_code": _has_weather_code(remaining_day, {95, 96, 99}),
        "model_future_has_rain_code": _has_weather_code(remaining_day, set(range(51, 68)) | set(range(80, 83)) | {95, 96, 99}),
        "model_future_shortwave_radiation_sum": pd.to_numeric(remaining_day.get("shortwave_radiation"), errors="coerce").sum(),
        "model_future_direct_radiation_sum": pd.to_numeric(remaining_day.get("direct_radiation"), errors="coerce").sum(),
        "model_future_diffuse_radiation_sum": pd.to_numeric(remaining_day.get("diffuse_radiation"), errors="coerce").sum(),
        "model_future_sunshine_duration_sum": pd.to_numeric(remaining_day.get("sunshine_duration"), errors="coerce").sum(),
        "model_future_cape_max": pd.to_numeric(remaining_day.get("cape"), errors="coerce").max(),
        "model_future_lifted_index_min": pd.to_numeric(remaining_day.get("lifted_index"), errors="coerce").min(),
        "model_future_wind_speed_max": pd.to_numeric(remaining_day.get("wind_speed_10m"), errors="coerce").max(),
        "model_future_gust_max": pd.to_numeric(remaining_day.get("wind_gusts_10m"), errors="coerce").max(),
        "nearest_gridpoint_value": pd.to_numeric(day.get("temperature_2m"), errors="coerce").max(),
        "source_id": source_id,
        "source_version": source_version,
        "ingest_time_utc": ingest_time_utc,
        "raw_file_reference": raw_file_reference,
        "raw_record_hash": stable_hash(
            {
                "source_id": source_id,
                "target_date_local": target_date_local.isoformat(),
                "model_run_time_utc": model_run_time_utc.isoformat() if model_run_time_utc else None,
                "payload": payload,
            }
        ),
        "quality_flag": quality_flag,
    }


def _value_at_local_hour(day: pd.DataFrame, column: str, target_date: date, timezone_name: str, hour: int):
    local = day["valid_time_utc"].dt.tz_convert(timezone_name)
    mask = (local.dt.date == target_date) & (local.dt.hour == hour)
    if column not in day.columns:
        return None
    values = pd.to_numeric(day.loc[mask, column], errors="coerce")
    return None if values.empty else values.iloc[0]


def _positive_hours(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.empty:
        return 0.0
    return float((values.fillna(0.0) > 0.0).sum())


def _has_weather_code(frame: pd.DataFrame, codes: set[int]) -> bool:
    if "weather_code" not in frame.columns:
        return False
    values = pd.to_numeric(frame["weather_code"], errors="coerce")
    if values.empty:
        return False
    observed = set(values.dropna().astype(int).tolist())
    return bool(observed.intersection(codes))
