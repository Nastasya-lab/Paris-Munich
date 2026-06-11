from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from weather_tmax_bot.features.metar_upside_dataset import build_asof_enhanced_metar_features
from weather_tmax_bot.utils.time import ensure_utc
from weather_tmax_bot.utils.time import local_day_bounds_utc


def build_metar_features(
    metar: pd.DataFrame,
    issue_time_utc: datetime,
    target_date_local: date | None = None,
    timezone_name: str = "Europe/Berlin",
) -> dict:
    issue = ensure_utc(issue_time_utc)
    if metar.empty:
        return {"metar_missing": True, "metar_missing_last_1h": True, "metar_missing_last_3h": True}
    df = metar.copy()
    df["knowledge_time_utc"] = pd.to_datetime(df["knowledge_time_utc"], utc=True)
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True)
    df = df[df["knowledge_time_utc"] <= pd.Timestamp(issue)].sort_values("observation_time_utc")
    if df.empty:
        return {"metar_missing": True, "metar_missing_last_1h": True, "metar_missing_last_3h": True}
    last = df.iloc[-1]
    last_1h = df[df["observation_time_utc"] >= pd.Timestamp(issue) - pd.Timedelta(hours=1)]
    last_3h = df[df["observation_time_utc"] >= pd.Timestamp(issue) - pd.Timedelta(hours=3)]
    last_6h = df[df["observation_time_utc"] >= pd.Timestamp(issue) - pd.Timedelta(hours=6)]
    target_day_df = _target_day_slice(df, target_date_local, timezone_name, issue)
    direction = last.get("wind_direction_deg")
    speed = last.get("wind_speed_kt")
    rad = np.deg2rad(direction) if pd.notna(direction) else np.nan
    features = {
        "metar_missing": False,
        "last_metar_temp_c": last.get("temperature_c"),
        "last_metar_dewpoint_c": last.get("dewpoint_c"),
        "last_metar_qnh_hpa": last.get("qnh_hpa"),
        "temp_trend_1h": _trend(last_1h, "temperature_c"),
        "temp_trend_3h": _trend(last_3h, "temperature_c"),
        "temp_trend_6h": _trend(last_6h, "temperature_c"),
        "observed_max_so_far_from_metar": target_day_df["temperature_c"].max() if not target_day_df.empty else np.nan,
        "observed_min_so_far_from_metar": target_day_df["temperature_c"].min() if not target_day_df.empty else np.nan,
        "pressure_trend_3h": _trend(last_3h, "qnh_hpa"),
        "dewpoint_depression": last.get("temperature_c") - last.get("dewpoint_c") if pd.notna(last.get("dewpoint_c")) else np.nan,
        "wind_u": -speed * np.sin(rad) if pd.notna(speed) and pd.notna(rad) else np.nan,
        "wind_v": -speed * np.cos(rad) if pd.notna(speed) and pd.notna(rad) else np.nan,
        "is_cavok": bool(last.get("cavok", False)),
        "has_precip_recent": _has_weather(df, ["RA", "SHRA"]),
        "has_fog_recent": _has_weather(df, ["FG", "BR"]),
        "has_thunder_recent": _has_weather(df, ["TS"]),
        "metar_missing_last_1h": last_1h.empty,
        "metar_missing_last_3h": last_3h.empty,
        "latest_metar_time_utc": last.get("observation_time_utc"),
        "max_metar_knowledge_time_utc": last.get("knowledge_time_utc"),
        "latest_metar_source_id": last.get("source_id"),
    }
    if target_date_local is not None:
        features.update(
            build_asof_enhanced_metar_features(
                df,
                issue_time_utc=issue,
                target_date_local=target_date_local,
                timezone_name=timezone_name,
            )
        )
    return features


def _trend(df: pd.DataFrame, col: str) -> float:
    values = pd.to_numeric(df.get(col), errors="coerce").dropna()
    if len(values) < 2:
        return np.nan
    return float(values.iloc[-1] - values.iloc[0])


def _has_weather(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).dropna().astype(str).tolist())
    return any(code in text for code in codes)


def _target_day_slice(
    df: pd.DataFrame,
    target_date_local: date | None,
    timezone_name: str,
    issue_time_utc: datetime,
) -> pd.DataFrame:
    if target_date_local is None:
        return df
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    return df[
        (df["observation_time_utc"] >= pd.Timestamp(day_start))
        & (df["observation_time_utc"] <= pd.Timestamp(day_end))
        & (df["observation_time_utc"] <= pd.Timestamp(issue_time_utc))
    ]
