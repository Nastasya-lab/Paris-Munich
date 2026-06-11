from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


ENHANCED_METAR_INTRADAY_FEATURES = [
    "temp_slope_since_sunrise",
    "temp_trend_last_2_metars",
    "latest_2_metar_temp_change_c",
    "cloud_cover_proxy_latest",
    "cloud_cover_proxy_trend_last_2_metars",
    "cloud_cover_proxy_trend_2h",
    "lowest_ceiling_ft_latest",
    "ceiling_trend_last_2_metars",
    "ceiling_trend_2h",
    "dewpoint_depression_latest",
    "dewpoint_depression_trend_2h",
    "pressure_tendency_1h",
    "pressure_tendency_3h",
    "wind_dir_shift_2h_deg",
    "wind_speed_trend_2h",
    "wind_direction_latest_deg",
    "wind_speed_latest_kt",
    "rain_started_after_current_max",
    "cb_tcu_appeared_after_current_max",
    "showers_appeared_after_current_max",
    "fog_or_br_recent_metar",
    "cavok_trend_last_2_metars",
    "metar_minutes_since_current_max",
    "metar_hours_since_sunrise",
    "temp_drop_after_rain_start_c",
    "temp_drop_after_cb_tcu_c",
    "wind_direction_valid_count_2h",
]


def build_metar_remaining_upside_dataset(
    metar: pd.DataFrame,
    target: pd.DataFrame,
    *,
    airport_icao: str,
    timezone_name: str,
    local_issue_hours: list[int] | None = None,
    rain_6min: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build training rows for final METAR Tmax minus observed METAR max so far."""
    if local_issue_hours is None:
        local_issue_hours = [6, 8, 10, 12, 14, 16, 18, 20]
    if metar.empty or target.empty:
        return _empty_frame()

    metar_df = _prepare_metar(metar)
    metar_df["target_date_local"] = metar_df["observation_time_utc"].dt.tz_convert(timezone_name).dt.date.astype(str)
    metar_by_day = {str(day): group.copy() for day, group in metar_df.groupby("target_date_local", sort=False)}
    rain_df = _prepare_rain(rain_6min)
    target_df = target.copy()
    target_df = target_df[target_df.get("quality_flags", "ok").eq("ok")].copy()
    target_df["target_date_local"] = target_df["target_date_local"].astype(str)
    target_df["metar_tmax_c"] = pd.to_numeric(target_df["metar_tmax_c"], errors="coerce")
    target_df = target_df.dropna(subset=["metar_tmax_c"])

    rows: list[dict] = []
    tz = ZoneInfo(timezone_name)
    for _, target_row in target_df.iterrows():
        target_date = date.fromisoformat(str(target_row["target_date_local"]))
        day_start, day_end = local_day_bounds_utc(target_date, timezone_name)
        day_metar = metar_by_day.get(target_date.isoformat(), pd.DataFrame()).copy()
        if not day_metar.empty:
            day_metar = day_metar[
                (day_metar["observation_time_utc"] >= pd.Timestamp(day_start))
                & (day_metar["observation_time_utc"] < pd.Timestamp(day_end))
            ].copy()
        if day_metar.empty:
            continue
        for hour in local_issue_hours:
            issue_local = datetime.combine(target_date, time(hour=hour), tzinfo=tz)
            issue_utc = pd.Timestamp(issue_local).tz_convert("UTC")
            so_far = day_metar[day_metar["knowledge_time_utc"] <= issue_utc].copy()
            if so_far.empty:
                continue
            latest = so_far.iloc[-1]
            current_max = float(so_far["temperature_c"].max())
            current_max_idx = so_far["temperature_c"].idxmax()
            after_current_max = so_far.loc[so_far.index >= current_max_idx].copy()
            final_max = float(target_row["metar_tmax_c"])
            upside = max(0.0, final_max - current_max)
            row = {
                "airport_icao": airport_icao,
                "target_date_local": target_date.isoformat(),
                "timezone": timezone_name,
                "issue_time_utc": issue_utc.isoformat(),
                "local_issue_hour": float(hour),
                "final_metar_tmax_c": final_max,
                "current_metar_max_c": current_max,
                "remaining_upside_c": upside,
                "upside_ge_1c": bool(upside >= 1.0),
                "upside_ge_2c": bool(upside >= 2.0),
                "upside_ge_3c": bool(upside >= 3.0),
                "latest_metar_temp_c": float(latest["temperature_c"]),
                "drop_from_current_max_c": float(current_max - latest["temperature_c"]),
                "metar_count_so_far": int(len(so_far)),
                "metar_count_last_1h": int(_window(so_far, issue_utc, 1).shape[0]),
                "metar_count_last_3h": int(_window(so_far, issue_utc, 3).shape[0]),
                "temp_trend_1h": _trend(_window(so_far, issue_utc, 1), "temperature_c"),
                "temp_trend_3h": _trend(_window(so_far, issue_utc, 3), "temperature_c"),
                "temp_trend_6h": _trend(_window(so_far, issue_utc, 6), "temperature_c"),
                "has_rain_recent_metar": _has_weather(_window(so_far, issue_utc, 3), ["RA", "SHRA", "TSRA"]),
                "has_thunder_recent_metar": _has_weather(_window(so_far, issue_utc, 6), ["TS"]),
                "is_cavok_latest": bool(latest.get("cavok", False)),
                **_enhanced_metar_features(so_far, after_current_max, latest, issue_utc, day_start),
                "latest_metar_time_utc": latest["observation_time_utc"].isoformat(),
                "max_feature_knowledge_time_utc": so_far["knowledge_time_utc"].max().isoformat(),
                "leakage_check_passed": bool(so_far["knowledge_time_utc"].max() <= issue_utc),
            }
            row.update(_rain_features(rain_df, day_start, issue_utc))
            rows.append(row)
    if not rows:
        return _empty_frame()
    return pd.DataFrame(rows).sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)


def build_current_metar_upside_features(
    metar: pd.DataFrame,
    *,
    airport_icao: str,
    target_date_local: date,
    issue_time_utc: datetime | pd.Timestamp,
    timezone_name: str,
    rain_6min: pd.DataFrame | None = None,
) -> dict:
    """Build one as-of feature row for a live METAR Tmax forecast."""
    metar_df = _prepare_metar(metar)
    issue_utc = pd.Timestamp(issue_time_utc).tz_convert("UTC") if pd.Timestamp(issue_time_utc).tzinfo else pd.Timestamp(issue_time_utc, tz="UTC")
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    day_metar = metar_df[
        (metar_df["observation_time_utc"] >= pd.Timestamp(day_start))
        & (metar_df["observation_time_utc"] < pd.Timestamp(day_end))
        & (metar_df["knowledge_time_utc"] <= issue_utc)
    ].copy()
    if day_metar.empty:
        raise ValueError(f"No METAR observations available for {airport_icao} on {target_date_local} as of {issue_utc.isoformat()}")
    latest = day_metar.iloc[-1]
    current_max = float(day_metar["temperature_c"].max())
    current_max_idx = day_metar["temperature_c"].idxmax()
    after_current_max = day_metar.loc[day_metar.index >= current_max_idx].copy()
    tz = ZoneInfo(timezone_name)
    issue_local = issue_utc.tz_convert(tz)
    row = {
        "airport_icao": airport_icao,
        "target_date_local": target_date_local.isoformat(),
        "timezone": timezone_name,
        "issue_time_utc": issue_utc.isoformat(),
        "local_issue_hour": float(issue_local.hour),
        "current_metar_max_c": current_max,
        "latest_metar_temp_c": float(latest["temperature_c"]),
        "drop_from_current_max_c": float(current_max - latest["temperature_c"]),
        "metar_count_so_far": int(len(day_metar)),
        "metar_count_last_1h": int(_window(day_metar, issue_utc, 1).shape[0]),
        "metar_count_last_3h": int(_window(day_metar, issue_utc, 3).shape[0]),
        "temp_trend_1h": _trend(_window(day_metar, issue_utc, 1), "temperature_c"),
        "temp_trend_3h": _trend(_window(day_metar, issue_utc, 3), "temperature_c"),
        "temp_trend_6h": _trend(_window(day_metar, issue_utc, 6), "temperature_c"),
        "has_rain_recent_metar": _has_weather(_window(day_metar, issue_utc, 3), ["RA", "SHRA", "TSRA"]),
        "has_thunder_recent_metar": _has_weather(_window(day_metar, issue_utc, 6), ["TS"]),
        "is_cavok_latest": bool(latest.get("cavok", False)),
        **_enhanced_metar_features(day_metar, after_current_max, latest, issue_utc, day_start),
        "latest_metar_time_utc": latest["observation_time_utc"].isoformat(),
        "latest_metar_raw": latest.get("raw_metar"),
        "max_feature_knowledge_time_utc": day_metar["knowledge_time_utc"].max().isoformat(),
        "leakage_check_passed": bool(day_metar["knowledge_time_utc"].max() <= issue_utc),
    }
    row.update(_rain_features(_prepare_rain(rain_6min), day_start, issue_utc))
    return row


def build_asof_enhanced_metar_features(
    metar: pd.DataFrame,
    *,
    issue_time_utc: datetime | pd.Timestamp,
    target_date_local: date,
    timezone_name: str,
) -> dict:
    """Compute same-day enhanced METAR features using only as-of records."""
    metar_df = _prepare_metar(metar)
    issue_utc = (
        pd.Timestamp(issue_time_utc).tz_convert("UTC")
        if pd.Timestamp(issue_time_utc).tzinfo
        else pd.Timestamp(issue_time_utc, tz="UTC")
    )
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    day_metar = metar_df[
        (metar_df["observation_time_utc"] >= pd.Timestamp(day_start))
        & (metar_df["observation_time_utc"] < pd.Timestamp(day_end))
        & (metar_df["knowledge_time_utc"] <= issue_utc)
    ].copy()
    if day_metar.empty:
        return empty_enhanced_metar_features()
    day_metar = day_metar.reset_index(drop=True)
    latest = day_metar.iloc[-1]
    current_max_idx = day_metar["temperature_c"].idxmax()
    after_current_max = day_metar.loc[day_metar.index >= current_max_idx].copy()
    return _enhanced_metar_features(day_metar, after_current_max, latest, issue_utc, day_start)


def empty_enhanced_metar_features() -> dict:
    boolean_columns = {
        "rain_started_after_current_max",
        "cb_tcu_appeared_after_current_max",
        "showers_appeared_after_current_max",
        "fog_or_br_recent_metar",
    }
    return {
        column: (False if column in boolean_columns else float("nan"))
        for column in ENHANCED_METAR_INTRADAY_FEATURES
    }


def _prepare_metar(metar: pd.DataFrame) -> pd.DataFrame:
    df = metar.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["knowledge_time_utc"] = pd.to_datetime(df.get("knowledge_time_utc", df["observation_time_utc"]), utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    for column in ["dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt", "ceiling_ft"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["observation_time_utc", "knowledge_time_utc", "temperature_c"]).sort_values("observation_time_utc")


def _prepare_rain(rain_6min: pd.DataFrame | None) -> pd.DataFrame:
    if rain_6min is None or rain_6min.empty:
        return pd.DataFrame(columns=["observation_time_utc", "rr_mm"])
    df = rain_6min.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["rr_mm"] = pd.to_numeric(df["rr_mm"], errors="coerce").fillna(0.0)
    return df.dropna(subset=["observation_time_utc"]).sort_values("observation_time_utc")


def _window(df: pd.DataFrame, issue_utc: pd.Timestamp, hours: float) -> pd.DataFrame:
    return df[df["observation_time_utc"] >= issue_utc - pd.Timedelta(hours=hours)]


def _trend(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.iloc[-1] - values.iloc[0])


def _enhanced_metar_features(
    so_far: pd.DataFrame,
    after_current_max: pd.DataFrame,
    latest: pd.Series,
    issue_utc: pd.Timestamp,
    day_start_utc: datetime,
) -> dict:
    last_2 = so_far.tail(2)
    day_since_sunrise = so_far[so_far["observation_time_utc"] >= pd.Timestamp(day_start_utc) + pd.Timedelta(hours=5)]
    cloud_proxy = so_far.apply(_cloud_cover_proxy, axis=1)
    ceiling = pd.to_numeric(so_far.get("ceiling_ft"), errors="coerce")
    dewpoint_depression = pd.to_numeric(so_far["temperature_c"], errors="coerce") - pd.to_numeric(so_far.get("dewpoint_c"), errors="coerce")
    wind_direction = pd.to_numeric(so_far.get("wind_direction_deg"), errors="coerce")
    return {
        "temp_slope_since_sunrise": _trend(day_since_sunrise, "temperature_c"),
        "temp_trend_last_2_metars": _trend(last_2, "temperature_c"),
        "latest_2_metar_temp_change_c": _trend(last_2, "temperature_c"),
        "cloud_cover_proxy_latest": float(_cloud_cover_proxy(latest)),
        "cloud_cover_proxy_trend_last_2_metars": _series_trend(cloud_proxy.tail(2)),
        "cloud_cover_proxy_trend_2h": _series_trend(cloud_proxy.loc[_window(so_far, issue_utc, 2).index]),
        "lowest_ceiling_ft_latest": _finite_or_nan(latest.get("ceiling_ft")),
        "ceiling_trend_last_2_metars": _series_trend(ceiling.tail(2)),
        "ceiling_trend_2h": _series_trend(ceiling.loc[_window(so_far, issue_utc, 2).index]),
        "dewpoint_depression_latest": _finite_or_nan(float(latest["temperature_c"]) - _finite_or_nan(latest.get("dewpoint_c"))),
        "dewpoint_depression_trend_2h": _series_trend(dewpoint_depression.loc[_window(so_far, issue_utc, 2).index]),
        "pressure_tendency_1h": _trend(_window(so_far, issue_utc, 1), "qnh_hpa"),
        "pressure_tendency_3h": _trend(_window(so_far, issue_utc, 3), "qnh_hpa"),
        "wind_dir_shift_2h_deg": _wind_shift(_window(so_far, issue_utc, 2)),
        "wind_speed_trend_2h": _trend(_window(so_far, issue_utc, 2), "wind_speed_kt"),
        "wind_direction_latest_deg": _finite_or_nan(latest.get("wind_direction_deg")),
        "wind_speed_latest_kt": _finite_or_nan(latest.get("wind_speed_kt")),
        "rain_started_after_current_max": _has_weather(after_current_max, ["RA", "SHRA", "TSRA"]),
        "cb_tcu_appeared_after_current_max": _has_weather(after_current_max, [" CB", "CB", "TCU"]),
        "showers_appeared_after_current_max": _has_weather(after_current_max, ["SHRA", "SH", "VCSH"]),
        "fog_or_br_recent_metar": _has_weather(_window(so_far, issue_utc, 3), ["FG", "BR"]),
        "cavok_trend_last_2_metars": _series_trend(last_2.get("cavok", pd.Series(dtype=float)).astype(float)),
        "metar_minutes_since_current_max": _minutes_since_current_max(so_far, issue_utc),
        "metar_hours_since_sunrise": max(0.0, (issue_utc - (pd.Timestamp(day_start_utc) + pd.Timedelta(hours=5))).total_seconds() / 3600.0),
        "temp_drop_after_rain_start_c": _temp_drop_after_weather(so_far, issue_utc, ["RA", "SHRA", "TSRA"]),
        "temp_drop_after_cb_tcu_c": _temp_drop_after_weather(so_far, issue_utc, [" CB", "CB", "TCU"]),
        "wind_direction_valid_count_2h": int(wind_direction.loc[_window(so_far, issue_utc, 2).index].dropna().shape[0]),
    }


def _has_weather(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    return any(code in text for code in codes)


def _series_trend(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.iloc[-1] - clean.iloc[0])


def _cloud_cover_proxy(row: pd.Series) -> float:
    if bool(row.get("cavok", False)):
        return 0.0
    text = " ".join(str(row.get(column, "") or "") for column in ["cloud_layers", "raw_metar"])
    if "OVC" in text:
        return 8.0
    if "BKN" in text:
        return 6.0
    if "SCT" in text:
        return 4.0
    if "FEW" in text:
        return 2.0
    if "NSC" in text or "SKC" in text or "CLR" in text:
        return 0.0
    return float("nan")


def _wind_shift(df: pd.DataFrame) -> float:
    values = pd.to_numeric(df.get("wind_direction_deg"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    diff = abs(values[-1] - values[0]) % 360
    return float(min(diff, 360 - diff))


def _minutes_since_current_max(so_far: pd.DataFrame, issue_utc: pd.Timestamp) -> float:
    if so_far.empty:
        return float("nan")
    idx = so_far["temperature_c"].idxmax()
    max_time = pd.Timestamp(so_far.loc[idx, "observation_time_utc"])
    return float((issue_utc - max_time).total_seconds() / 60.0)


def _temp_drop_after_weather(so_far: pd.DataFrame, issue_utc: pd.Timestamp, codes: list[str]) -> float:
    if so_far.empty:
        return 0.0
    mask = so_far.get("raw_metar", pd.Series("", index=so_far.index)).fillna("").astype(str).apply(lambda text: any(code in text for code in codes))
    if not mask.any():
        return 0.0
    first_weather = so_far.loc[mask, "observation_time_utc"].min()
    after = so_far[(so_far["observation_time_utc"] >= first_weather) & (so_far["observation_time_utc"] <= issue_utc)]
    if after.empty:
        return 0.0
    return float(pd.to_numeric(after["temperature_c"], errors="coerce").max() - pd.to_numeric(after["temperature_c"], errors="coerce").iloc[-1])


def _finite_or_nan(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _rain_features(rain: pd.DataFrame, day_start_utc: datetime, issue_utc: pd.Timestamp) -> dict:
    if rain.empty:
        return {
            "rain_6min_missing": True,
            "rain_mm_last_30m": 0.0,
            "rain_mm_last_1h": 0.0,
            "rain_mm_last_3h": 0.0,
            "rain_mm_since_midnight": 0.0,
            "rain_max_6min_last_3h": 0.0,
        }
    available = rain[rain["observation_time_utc"] <= issue_utc]
    day = available[available["observation_time_utc"] >= pd.Timestamp(day_start_utc)]
    return {
        "rain_6min_missing": False,
        "rain_mm_last_30m": _rain_sum(available, issue_utc, 0.5),
        "rain_mm_last_1h": _rain_sum(available, issue_utc, 1),
        "rain_mm_last_3h": _rain_sum(available, issue_utc, 3),
        "rain_mm_since_midnight": float(day["rr_mm"].sum()) if not day.empty else 0.0,
        "rain_max_6min_last_3h": float(_window(available, issue_utc, 3)["rr_mm"].max() or 0.0),
    }


def _rain_sum(rain: pd.DataFrame, issue_utc: pd.Timestamp, hours: float) -> float:
    return float(_window(rain, issue_utc, hours)["rr_mm"].sum())


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()
