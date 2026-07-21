from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
DEFAULT_VARIABLES = (
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "precipitation",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
    "surface_pressure",
    "cape",
)


def fetch_previous_day1_hourly(
    *,
    latitude: float,
    longitude: float,
    model: str,
    start_date: str,
    end_date: str,
    variables: tuple[str, ...] = DEFAULT_VARIABLES,
    cache_dir: str | Path | None = None,
    chunk_days: int = 90,
    request_retries: int = 4,
) -> pd.DataFrame:
    """Fetch fixed 24-hour lead forecasts with resumable per-chunk caching."""
    first = pd.Timestamp(start_date).date()
    last = pd.Timestamp(end_date).date()
    cache = None if cache_dir is None else Path(cache_dir)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    current = first
    while current <= last:
        chunk_end = min(last, current + pd.Timedelta(days=chunk_days - 1))
        path = None if cache is None else cache / f"{model}_{current}_{chunk_end}.parquet"
        if path is not None and path.exists():
            frame = pd.read_parquet(path)
        else:
            frame = _fetch_chunk(
                latitude=latitude,
                longitude=longitude,
                model=model,
                start_date=current.isoformat(),
                end_date=chunk_end.isoformat(),
                variables=variables,
                retries=request_retries,
            )
            if path is not None:
                frame.to_parquet(path, index=False)
        parts.append(frame)
        current = chunk_end + pd.Timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["valid_time_utc"] = pd.to_datetime(out["valid_time_utc"], utc=True, errors="coerce")
    return out.dropna(subset=["valid_time_utc"]).drop_duplicates("valid_time_utc", keep="last").sort_values("valid_time_utc").reset_index(drop=True)


def build_previous_day1_snapshots(
    hourly: pd.DataFrame,
    *,
    timezone_name: str,
    issue_hours: list[int],
    prefix: str,
) -> pd.DataFrame:
    """Aggregate a fixed-lead hourly trajectory into leakage-safe intraday rows."""
    if hourly.empty:
        return pd.DataFrame()
    frame = hourly.copy()
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True, errors="coerce")
    local = frame["valid_time_utc"].dt.tz_convert(timezone_name)
    frame["target_date_local"] = local.dt.date.astype(str)
    rows: list[dict] = []
    for target_date, day in frame.groupby("target_date_local", sort=True):
        local_day = day["valid_time_utc"].dt.tz_convert(timezone_name)
        # A usable day needs most of its hourly trajectory, including daytime.
        if local_day.dt.hour.nunique() < 20 or pd.to_numeric(day["temperature_2m"], errors="coerce").notna().sum() < 20:
            continue
        day_start = pd.Timestamp(target_date, tz=timezone_name).tz_convert("UTC")
        for hour in issue_hours:
            issue = pd.Timestamp(f"{target_date} {hour:02d}:00", tz=timezone_name).tz_convert("UTC")
            future = day[day["valid_time_utc"] >= issue]
            if future.empty:
                continue
            values = _aggregate(day, future, prefix)
            values.update(
                {
                    "target_date_local": target_date,
                    "issue_time_utc": issue,
                    "local_issue_hour": float(hour),
                    "nwp_knowledge_time_utc": day_start,
                    "nwp_source_id": f"open_meteo.previous_day1.{day['model_name'].iloc[0]}",
                }
            )
            rows.append(values)
    return pd.DataFrame(rows).sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)


def _fetch_chunk(
    *,
    latitude: float,
    longitude: float,
    model: str,
    start_date: str,
    end_date: str,
    variables: tuple[str, ...],
    retries: int,
) -> pd.DataFrame:
    names = ",".join(f"{name}_previous_day1" for name in variables)
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "models": model,
        "hourly": names,
        "timezone": "UTC",
        "start_date": start_date,
        "end_date": end_date,
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=120)
            response.raise_for_status()
            hourly = pd.DataFrame(response.json().get("hourly", {}))
            if hourly.empty or "time" not in hourly:
                raise RuntimeError(f"Empty previous-runs response for {model} {start_date}..{end_date}")
            hourly["valid_time_utc"] = pd.to_datetime(hourly.pop("time"), utc=True)
            hourly = hourly.rename(columns={f"{name}_previous_day1": name for name in variables})
            hourly["model_name"] = model
            return hourly
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Previous-runs download failed for {model} {start_date}..{end_date}: {last_error}")


def _aggregate(day: pd.DataFrame, future: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_tmax_c": _max(day, "temperature_2m"),
        f"{prefix}_future_temp_max_c": _max(future, "temperature_2m"),
        f"{prefix}_cloud_cover_mean": _mean(day, "cloud_cover"),
        f"{prefix}_future_cloud_cover_mean": _mean(future, "cloud_cover"),
        f"{prefix}_precip_sum": _sum(day, "precipitation"),
        f"{prefix}_future_precip_sum": _sum(future, "precipitation"),
        f"{prefix}_shortwave_radiation_sum": _sum(day, "shortwave_radiation"),
        f"{prefix}_future_shortwave_radiation_sum": _sum(future, "shortwave_radiation"),
        f"{prefix}_wind_speed_max": _max(day, "wind_speed_10m"),
        f"{prefix}_future_wind_speed_max": _max(future, "wind_speed_10m"),
        f"{prefix}_gust_max": _max(day, "wind_gusts_10m"),
        f"{prefix}_future_gust_max": _max(future, "wind_gusts_10m"),
        f"{prefix}_dewpoint_mean": _mean(day, "dew_point_2m"),
        f"{prefix}_relative_humidity_mean": _mean(day, "relative_humidity_2m"),
        f"{prefix}_surface_pressure_mean": _mean(day, "surface_pressure"),
        f"{prefix}_cape_max": _max(day, "cape"),
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _mean(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column)
    return float(values.mean()) if values.notna().any() else float("nan")


def _sum(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column)
    return float(values.sum()) if values.notna().any() else float("nan")


def _max(frame: pd.DataFrame, column: str) -> float:
    values = _numeric(frame, column)
    return float(values.max()) if values.notna().any() else float("nan")
