from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


def build_daily_metar_tmax(
    metar: pd.DataFrame,
    *,
    airport_icao: str,
    timezone_name: str,
    source_id: str,
    expected_reports_per_day: int = 48,
) -> pd.DataFrame:
    """Build a daily target for the maximum temperature reported by METAR/SPECI."""
    columns = [
        "airport_icao",
        "target_date_local",
        "timezone",
        "metar_tmax_c",
        "metar_tmax_time_local",
        "metar_tmax_time_utc",
        "metar_obs_count",
        "expected_metar_obs_count",
        "metar_missing_ratio",
        "has_speci",
        "source_id",
        "quality_flags",
        "created_at_utc",
    ]
    if metar.empty:
        return pd.DataFrame(columns=columns)

    df = metar.copy()
    if "observation_time_utc" not in df.columns:
        raise ValueError("METAR target input must include observation_time_utc")
    if "temperature_c" not in df.columns:
        raise ValueError("METAR target input must include temperature_c")

    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["observation_time_utc", "temperature_c"])
    if df.empty:
        return pd.DataFrame(columns=columns)

    local_time = df["observation_time_utc"].dt.tz_convert(timezone_name)
    df["target_date_local"] = local_time.dt.date.astype(str)
    df["is_speci"] = df.get("raw_metar", pd.Series("", index=df.index)).fillna("").astype(str).str.startswith("SPECI")

    idx = df.groupby("target_date_local")["temperature_c"].idxmax()
    maxima = df.loc[idx, ["target_date_local", "temperature_c", "observation_time_utc"]].rename(
        columns={"temperature_c": "metar_tmax_c", "observation_time_utc": "metar_tmax_time_utc"}
    )
    counts = df.groupby("target_date_local").size().rename("metar_obs_count")
    speci = df.groupby("target_date_local")["is_speci"].any().rename("has_speci")
    out = maxima.merge(counts, on="target_date_local").merge(speci, on="target_date_local")
    out["airport_icao"] = airport_icao
    out["timezone"] = timezone_name
    out["metar_tmax_time_local"] = out["metar_tmax_time_utc"].dt.tz_convert(timezone_name)
    out["expected_metar_obs_count"] = out["target_date_local"].map(
        lambda value: _expected_reports_for_local_day(value, timezone_name, expected_reports_per_day)
    )
    out["metar_missing_ratio"] = (1.0 - out["metar_obs_count"] / out["expected_metar_obs_count"]).clip(lower=0.0)
    out["source_id"] = source_id
    out["quality_flags"] = out["metar_missing_ratio"].map(lambda ratio: "low_coverage" if ratio > 0.35 else "ok")
    out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["metar_tmax_time_utc"] = out["metar_tmax_time_utc"].map(lambda value: value.isoformat())
    out["metar_tmax_time_local"] = out["metar_tmax_time_local"].map(lambda value: value.isoformat())
    return out[columns].sort_values("target_date_local").reset_index(drop=True)


def _expected_reports_for_local_day(date_value: str, timezone_name: str, reports_per_24h: int) -> int:
    start, end = local_day_bounds_utc(pd.to_datetime(date_value).date(), timezone_name)
    hours = (end - start).total_seconds() / 3600.0
    return max(1, int(round(reports_per_24h * hours / 24.0)))
