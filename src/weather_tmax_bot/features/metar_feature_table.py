from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd
import numpy as np

from weather_tmax_bot.utils.time import local_day_bounds_utc


ISSUE_HOURS_UTC = [0, 3, 6, 9, 12, 15, 18]


def build_metar_feature_table(
    metar: pd.DataFrame,
    target_dates: list[str],
    issue_hours_utc: list[int] | None = None,
    timezone_name: str = "Europe/Berlin",
) -> pd.DataFrame:
    issue_hours_utc = issue_hours_utc or ISSUE_HOURS_UTC
    if metar.empty:
        return pd.DataFrame()
    prepared = metar.copy()
    prepared["_observation_ts"] = pd.to_datetime(prepared["observation_time_utc"], utc=True)
    prepared["_knowledge_ts"] = pd.to_datetime(prepared["knowledge_time_utc"], utc=True)
    for col in ("temperature_c", "dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt"):
        if col in prepared.columns:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
    rows = []
    for target_date_str in target_dates:
        target_date = pd.to_datetime(target_date_str).date()
        day_start, day_end = local_day_bounds_utc(target_date, timezone_name)
        day = prepared[
            (prepared["_observation_ts"] >= pd.Timestamp(day_start))
            & (prepared["_observation_ts"] <= pd.Timestamp(day_end))
        ].sort_values("_knowledge_ts").copy()
        issues = pd.DataFrame(
            {
                "target_date_local": target_date.isoformat(),
                "issue_hour_utc": issue_hours_utc,
                "issue_time_utc": [
                    datetime.combine(target_date, time(hour=hour), tzinfo=timezone.utc) for hour in issue_hours_utc
                ],
            }
        )
        if day.empty:
            for _, issue_row in issues.iterrows():
                rows.append(_missing_row(issue_row))
            continue
        day["observed_max_so_far_from_metar"] = day["temperature_c"].cummax()
        day["observed_min_so_far_from_metar"] = day["temperature_c"].cummin()
        knowledge_ns = _timestamp_ns(day["_knowledge_ts"])
        observation_ns = _timestamp_ns(day["_observation_ts"])
        for _, issue_row in issues.iterrows():
            issue_ts = pd.Timestamp(issue_row["issue_time_utc"])
            idx = int(np.searchsorted(knowledge_ns, issue_ts.value, side="right") - 1)
            if idx < 0:
                rows.append(_missing_row(issue_row))
                continue
            last = day.iloc[idx]
            before_1h = _row_at(day, knowledge_ns, issue_ts - pd.Timedelta(hours=1))
            before_3h = _row_at(day, knowledge_ns, issue_ts - pd.Timedelta(hours=3))
            before_6h = _row_at(day, knowledge_ns, issue_ts - pd.Timedelta(hours=6))
            recent_start = max(0, int(np.searchsorted(observation_ns, (issue_ts - pd.Timedelta(hours=3)).value, side="left")))
            recent_3h = day.iloc[recent_start : idx + 1]
            direction = last.get("wind_direction_deg")
            speed = last.get("wind_speed_kt")
            rad = pd.NA if pd.isna(direction) else float(direction) * 3.141592653589793 / 180.0
            rows.append(
                {
                    "target_date_local": issue_row["target_date_local"],
                    "issue_time_utc": issue_row["issue_time_utc"].isoformat(),
                    "issue_hour_utc": int(issue_row["issue_hour_utc"]),
                    "metar_missing": False,
                    "last_metar_temp_c": last.get("temperature_c"),
                    "last_metar_dewpoint_c": last.get("dewpoint_c"),
                    "last_metar_qnh_hpa": last.get("qnh_hpa"),
                    "temp_trend_1h": _delta(last.get("temperature_c"), before_1h.get("temperature_c")),
                    "temp_trend_3h": _delta(last.get("temperature_c"), before_3h.get("temperature_c")),
                    "temp_trend_6h": _delta(last.get("temperature_c"), before_6h.get("temperature_c")),
                    "observed_max_so_far_from_metar": last.get("observed_max_so_far_from_metar"),
                    "observed_min_so_far_from_metar": last.get("observed_min_so_far_from_metar"),
                    "pressure_trend_3h": _delta(last.get("qnh_hpa"), before_3h.get("qnh_hpa")),
                    "dewpoint_depression": _delta(last.get("temperature_c"), last.get("dewpoint_c")),
                    "wind_u": None if pd.isna(speed) or pd.isna(rad) else -float(speed) * __import__("math").sin(rad),
                    "wind_v": None if pd.isna(speed) or pd.isna(rad) else -float(speed) * __import__("math").cos(rad),
                    "is_cavok": bool(last.get("cavok", False)),
                    "has_precip_recent": _has_recent(recent_3h, ["RA", "SHRA", "TSRA"]),
                    "has_fog_recent": _has_recent(recent_3h, ["FG", "BR"]),
                    "has_thunder_recent": _has_recent(recent_3h, ["TS", "TSRA"]),
                    "metar_missing_last_1h": _older_than(last.get("_observation_ts"), issue_ts, 1),
                    "metar_missing_last_3h": _older_than(last.get("_observation_ts"), issue_ts, 3),
                    "latest_metar_time_utc": last.get("observation_time_utc"),
                    "max_metar_knowledge_time_utc": last.get("knowledge_time_utc"),
                }
            )
    return pd.DataFrame(rows)


def build_metar_feature_table_from_files(
    metar_path: str | Path = "data/interim/metar_iem_EDDM.parquet",
    target_path: str | Path = "data/processed/daily_target.parquet",
) -> pd.DataFrame:
    metar = pd.read_parquet(metar_path)
    target = pd.read_parquet(target_path)
    target = target[target["quality_flags"] == "ok"].copy()
    return build_metar_feature_table(metar, target["target_date_local"].astype(str).tolist())


def _row_at(day: pd.DataFrame, knowledge_ns: np.ndarray, lookup_time: pd.Timestamp) -> pd.Series:
    idx = int(np.searchsorted(knowledge_ns, lookup_time.value, side="right") - 1)
    if idx < 0:
        return pd.Series(dtype=object)
    return day.iloc[idx]


def _missing_row(issue_row: pd.Series) -> dict:
    return {
        "target_date_local": issue_row["target_date_local"],
        "issue_time_utc": issue_row["issue_time_utc"].isoformat(),
        "issue_hour_utc": int(issue_row["issue_hour_utc"]),
        "metar_missing": True,
        "metar_missing_last_1h": True,
        "metar_missing_last_3h": True,
    }


def _delta(current, previous):
    if pd.isna(current) or pd.isna(previous):
        return None
    return float(current) - float(previous)


def _older_than(observation_time, issue_time, hours: int) -> bool:
    if pd.isna(observation_time):
        return True
    return pd.Timestamp(observation_time) < pd.Timestamp(issue_time) - pd.Timedelta(hours=hours)


def _has_recent(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).dropna().astype(str).tolist())
    return any(code in text for code in codes)


def _timestamp_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True).dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype("int64")
