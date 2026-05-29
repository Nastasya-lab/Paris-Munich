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
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True)
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["temperature_c"])
    local_times = df["observation_time_utc"].dt.tz_convert(timezone_name)
    df["target_date_local"] = local_times.dt.date.astype(str)
    idx = df.groupby("target_date_local")["temperature_c"].idxmax()
    maxima = df.loc[idx, ["target_date_local", "temperature_c", "observation_time_utc"]].rename(
        columns={"temperature_c": "tmax_c", "observation_time_utc": "tmax_time_utc"}
    )
    counts = df.groupby("target_date_local").size().rename("obs_count")
    out = maxima.merge(counts, on="target_date_local")
    out["airport_icao"] = airport_icao
    out["station_id"] = df.get("station_id", "01262").iloc[0]
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
    out["source_id"] = df.get("source_id", "dwd.10min.air_temperature.01262").iloc[0]
    out["source_version"] = df.get("source_version", "unknown").iloc[0]
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
