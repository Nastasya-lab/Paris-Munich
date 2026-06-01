from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


def build_daily_tmax(
    observations: pd.DataFrame,
    airport_icao: str = "EDDM",
    timezone_name: str = "Europe/Berlin",
    expected_obs_count: int = 144,
) -> pd.DataFrame:
    df = observations.copy()
    if df.empty:
        return _empty_daily_tmax_frame()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["temperature_c"])
    if df.empty:
        return _empty_daily_tmax_frame()
    local_times = df["observation_time_utc"].dt.tz_convert(timezone_name)
    df["target_date_local"] = local_times.dt.date.astype(str)
    idx = df.groupby("target_date_local")["temperature_c"].idxmax()
    maxima = df.loc[idx, ["target_date_local", "temperature_c", "observation_time_utc"]].rename(
        columns={"temperature_c": "tmax_c", "observation_time_utc": "tmax_time_utc"}
    )
    counts = df.groupby("target_date_local").size().rename("obs_count")
    out = maxima.merge(counts, on="target_date_local")
    out["airport_icao"] = airport_icao
    out["station_id"] = _first_or_default(df, "station_id", "01262")
    out["timezone"] = timezone_name
    out["tmax_time_local"] = out["tmax_time_utc"].dt.tz_convert(timezone_name)
    out["expected_obs_count"] = out["target_date_local"].map(
        lambda d: int(
            round(
                (
                    local_day_bounds_utc(pd.to_datetime(d).date(), timezone_name)[1]
                    - local_day_bounds_utc(pd.to_datetime(d).date(), timezone_name)[0]
                ).total_seconds()
                / 600
            )
        )
    )
    out["missing_ratio"] = (1 - out["obs_count"] / expected_obs_count).clip(lower=0)
    out["quality_flags"] = out["missing_ratio"].map(lambda x: "low_coverage" if x > 0.2 else "ok")
    out["source_id"] = _first_or_default(df, "source_id", "dwd.10min.air_temperature.01262")
    out["source_version"] = _first_or_default(df, "source_version", "unknown")
    out["truth_data_release_time_utc"] = None
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["tmax_time_utc"] = out["tmax_time_utc"].map(lambda x: x.isoformat())
    out["tmax_time_local"] = out["tmax_time_local"].map(lambda x: x.isoformat())
    return out[
        [
            "airport_icao",
            "station_id",
            "target_date_local",
            "timezone",
            "tmax_c",
            "tmax_time_local",
            "tmax_time_utc",
            "obs_count",
            "expected_obs_count",
            "missing_ratio",
            "quality_flags",
            "source_id",
            "source_version",
            "truth_data_release_time_utc",
            "created_at_utc",
        ]
    ].sort_values("target_date_local")


def _first_or_default(df: pd.DataFrame, column: str, default: str) -> str:
    if column not in df.columns or df[column].empty:
        return default
    return str(df[column].iloc[0])


def _empty_daily_tmax_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "airport_icao",
            "station_id",
            "target_date_local",
            "timezone",
            "tmax_c",
            "tmax_time_local",
            "tmax_time_utc",
            "obs_count",
            "expected_obs_count",
            "missing_ratio",
            "quality_flags",
            "source_id",
            "source_version",
            "truth_data_release_time_utc",
            "created_at_utc",
        ]
    )
