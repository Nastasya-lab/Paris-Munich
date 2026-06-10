from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


DEFAULT_ADVECTION_STATIONS = ["LFPB", "LFPG", "LFPO"]


def wind_advection_feature_columns(stations: list[str] | tuple[str, ...] = DEFAULT_ADVECTION_STATIONS) -> list[str]:
    columns: list[str] = []
    for station in stations:
        prefix = f"adv_{station.lower()}"
        columns.extend(
            [
                f"{prefix}_available",
                f"{prefix}_wind_dir_latest_deg",
                f"{prefix}_wind_speed_latest_kt",
                f"{prefix}_wind_u_latest",
                f"{prefix}_wind_v_latest",
                f"{prefix}_wind_dir_shift_1h_deg",
                f"{prefix}_wind_dir_shift_3h_deg",
                f"{prefix}_wind_speed_trend_1h",
                f"{prefix}_wind_speed_trend_3h",
                f"{prefix}_temp_trend_1h",
                f"{prefix}_temp_trend_3h",
                f"{prefix}_dewpoint_trend_1h",
                f"{prefix}_dewpoint_trend_3h",
                f"{prefix}_dewpoint_depression_latest",
                f"{prefix}_dewpoint_depression_trend_3h",
                f"{prefix}_pressure_tendency_1h",
                f"{prefix}_pressure_tendency_3h",
                f"{prefix}_north_sector_latest",
                f"{prefix}_east_sector_latest",
                f"{prefix}_south_sector_latest",
                f"{prefix}_west_sector_latest",
                f"{prefix}_cold_advection_signal",
                f"{prefix}_warm_advection_signal",
                f"{prefix}_frontal_passage_signal",
            ]
        )
    columns.extend(
        [
            "adv_available_station_count",
            "adv_mean_wind_u_latest",
            "adv_mean_wind_v_latest",
            "adv_mean_wind_speed_latest_kt",
            "adv_mean_temp_trend_1h",
            "adv_mean_temp_trend_3h",
            "adv_mean_dewpoint_trend_3h",
            "adv_mean_pressure_tendency_3h",
            "adv_any_north_sector",
            "adv_any_south_sector",
            "adv_any_cold_advection_signal",
            "adv_any_warm_advection_signal",
            "adv_any_frontal_passage_signal",
            "adv_all_available_cold_advection_signal",
            "adv_lfpg_minus_lfpb_temp_trend_1h",
            "adv_lfpo_minus_lfpb_temp_trend_1h",
            "adv_neighbor_mean_minus_lfpb_temp_trend_1h",
            "adv_neighbor_mean_minus_lfpb_dewpoint_trend_3h",
            "adv_neighbor_mean_minus_lfpb_pressure_tendency_3h",
        ]
    )
    return columns


def build_wind_advection_features(
    station_metars: dict[str, pd.DataFrame],
    *,
    target_date_local: date,
    issue_time_utc,
    timezone_name: str,
    stations: list[str] | tuple[str, ...] = DEFAULT_ADVECTION_STATIONS,
) -> dict:
    issue = pd.Timestamp(issue_time_utc).tz_convert("UTC")
    features: dict[str, float | int | bool | str] = {}
    station_summaries: dict[str, dict] = {}
    max_knowledge_time: pd.Timestamp | None = None

    for station in stations:
        prefix = f"adv_{station.lower()}"
        station_features, summary, station_knowledge = _station_advection_features(
            _prepare_metar(station_metars.get(station, pd.DataFrame())),
            issue_time_utc=issue,
            target_date_local=target_date_local,
            timezone_name=timezone_name,
            prefix=prefix,
        )
        features.update(station_features)
        station_summaries[station] = summary
        if station_knowledge is not None:
            max_knowledge_time = station_knowledge if max_knowledge_time is None else max(max_knowledge_time, station_knowledge)

    available = [summary for summary in station_summaries.values() if summary.get("available")]
    neighbor_summaries = [station_summaries.get(station, {}) for station in stations if station != "LFPB"]
    lfpb = station_summaries.get("LFPB", {})

    features.update(
        {
            "adv_available_station_count": int(len(available)),
            "adv_mean_wind_u_latest": _mean_or_nan([s.get("wind_u_latest") for s in available]),
            "adv_mean_wind_v_latest": _mean_or_nan([s.get("wind_v_latest") for s in available]),
            "adv_mean_wind_speed_latest_kt": _mean_or_nan([s.get("wind_speed_latest_kt") for s in available]),
            "adv_mean_temp_trend_1h": _mean_or_nan([s.get("temp_trend_1h") for s in available]),
            "adv_mean_temp_trend_3h": _mean_or_nan([s.get("temp_trend_3h") for s in available]),
            "adv_mean_dewpoint_trend_3h": _mean_or_nan([s.get("dewpoint_trend_3h") for s in available]),
            "adv_mean_pressure_tendency_3h": _mean_or_nan([s.get("pressure_tendency_3h") for s in available]),
            "adv_any_north_sector": any(bool(s.get("north_sector_latest")) for s in available),
            "adv_any_south_sector": any(bool(s.get("south_sector_latest")) for s in available),
            "adv_any_cold_advection_signal": any(bool(s.get("cold_advection_signal")) for s in available),
            "adv_any_warm_advection_signal": any(bool(s.get("warm_advection_signal")) for s in available),
            "adv_any_frontal_passage_signal": any(bool(s.get("frontal_passage_signal")) for s in available),
            "adv_all_available_cold_advection_signal": bool(available) and all(bool(s.get("cold_advection_signal")) for s in available),
            "adv_lfpg_minus_lfpb_temp_trend_1h": _maybe_diff(station_summaries.get("LFPG", {}).get("temp_trend_1h"), lfpb.get("temp_trend_1h")),
            "adv_lfpo_minus_lfpb_temp_trend_1h": _maybe_diff(station_summaries.get("LFPO", {}).get("temp_trend_1h"), lfpb.get("temp_trend_1h")),
            "adv_neighbor_mean_minus_lfpb_temp_trend_1h": _maybe_diff(
                _mean_or_nan([s.get("temp_trend_1h") for s in neighbor_summaries]), lfpb.get("temp_trend_1h")
            ),
            "adv_neighbor_mean_minus_lfpb_dewpoint_trend_3h": _maybe_diff(
                _mean_or_nan([s.get("dewpoint_trend_3h") for s in neighbor_summaries]), lfpb.get("dewpoint_trend_3h")
            ),
            "adv_neighbor_mean_minus_lfpb_pressure_tendency_3h": _maybe_diff(
                _mean_or_nan([s.get("pressure_tendency_3h") for s in neighbor_summaries]), lfpb.get("pressure_tendency_3h")
            ),
            "adv_max_feature_knowledge_time_utc": (max_knowledge_time or issue).isoformat(),
            "adv_leakage_check_passed": bool((max_knowledge_time or issue) <= issue),
        }
    )
    return features


def add_wind_advection_features_to_frame(
    frame: pd.DataFrame,
    station_metars: dict[str, pd.DataFrame],
    *,
    timezone_name: str,
    stations: list[str] | tuple[str, ...] = DEFAULT_ADVECTION_STATIONS,
) -> pd.DataFrame:
    out = frame.copy()
    station_day_metars = _station_day_metar_map(station_metars, timezone_name=timezone_name, stations=stations)
    rows = [
        build_wind_advection_features(
            {
                station: station_day_metars.get(station, {}).get(str(row["target_date_local"]), pd.DataFrame())
                for station in stations
            },
            target_date_local=pd.Timestamp(str(row["target_date_local"])).date(),
            issue_time_utc=row["issue_time_utc"],
            timezone_name=timezone_name,
            stations=stations,
        )
        for _, row in out.iterrows()
    ]
    advection = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, advection], axis=1)
    out["max_feature_knowledge_time_utc"] = out["adv_max_feature_knowledge_time_utc"]
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool) & out["adv_leakage_check_passed"].fillna(False).astype(bool)
    return out


def _station_day_metar_map(
    station_metars: dict[str, pd.DataFrame],
    *,
    timezone_name: str,
    stations: list[str] | tuple[str, ...],
) -> dict[str, dict[str, pd.DataFrame]]:
    out: dict[str, dict[str, pd.DataFrame]] = {}
    for station in stations:
        df = _prepare_metar(station_metars.get(station, pd.DataFrame()))
        if df.empty:
            out[station] = {}
            continue
        df = df.copy()
        df["_target_date_local"] = df["observation_time_utc"].dt.tz_convert(timezone_name).dt.date.astype(str)
        out[station] = {str(day): group.drop(columns=["_target_date_local"]).copy() for day, group in df.groupby("_target_date_local", sort=False)}
    return out


def _station_advection_features(
    frame: pd.DataFrame,
    *,
    issue_time_utc: pd.Timestamp,
    target_date_local: date,
    timezone_name: str,
    prefix: str,
) -> tuple[dict, dict, pd.Timestamp | None]:
    empty = _empty_station_features(prefix)
    if frame.empty:
        return empty, {"available": False}, None
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    so_far = frame[
        (frame["observation_time_utc"] >= pd.Timestamp(day_start))
        & (frame["observation_time_utc"] < pd.Timestamp(day_end))
        & (frame["knowledge_time_utc"] <= issue_time_utc)
    ].copy()
    if so_far.empty:
        return empty, {"available": False}, None

    latest = so_far.iloc[-1]
    wind_dir = _finite_or_nan(latest.get("wind_direction_deg"))
    wind_speed = _finite_or_nan(latest.get("wind_speed_kt"))
    wind_u, wind_v = _wind_components(wind_dir, wind_speed)
    temp_trend_1h = _trend(_window(so_far, issue_time_utc, 1), "temperature_c")
    temp_trend_3h = _trend(_window(so_far, issue_time_utc, 3), "temperature_c")
    dewpoint_trend_1h = _trend(_window(so_far, issue_time_utc, 1), "dewpoint_c")
    dewpoint_trend_3h = _trend(_window(so_far, issue_time_utc, 3), "dewpoint_c")
    pressure_tendency_1h = _trend(_window(so_far, issue_time_utc, 1), "qnh_hpa")
    pressure_tendency_3h = _trend(_window(so_far, issue_time_utc, 3), "qnh_hpa")
    dewpoint_depression = _finite_or_nan(float(latest["temperature_c"]) - _finite_or_nan(latest.get("dewpoint_c")))
    dewpoint_depression_trend_3h = _series_trend(
        pd.to_numeric(_window(so_far, issue_time_utc, 3)["temperature_c"], errors="coerce")
        - pd.to_numeric(_window(so_far, issue_time_utc, 3).get("dewpoint_c"), errors="coerce")
    )
    north, east, south, west = _sector_flags(wind_dir)
    wind_shift_1h = _wind_shift(_window(so_far, issue_time_utc, 1))
    wind_shift_3h = _wind_shift(_window(so_far, issue_time_utc, 3))
    wind_speed_trend_1h = _trend(_window(so_far, issue_time_utc, 1), "wind_speed_kt")
    wind_speed_trend_3h = _trend(_window(so_far, issue_time_utc, 3), "wind_speed_kt")

    cold_signal = bool(
        (north or west)
        and _lt(temp_trend_3h, -1.0)
        and (_lt(dewpoint_trend_3h, -1.0) or _gt(pressure_tendency_3h, 1.0))
    )
    warm_signal = bool(
        south
        and _gt(temp_trend_3h, 0.5)
        and (_gt(dewpoint_trend_3h, -0.5) or _lt(pressure_tendency_3h, 1.5))
    )
    frontal_signal = bool(
        _gt(wind_shift_3h, 45.0)
        and (_lt(temp_trend_3h, -1.0) or _lt(dewpoint_trend_3h, -1.5))
        and (_gt(pressure_tendency_3h, 0.5) or _gt(wind_speed_trend_3h, 3.0))
    )

    features = {
        f"{prefix}_available": True,
        f"{prefix}_wind_dir_latest_deg": wind_dir,
        f"{prefix}_wind_speed_latest_kt": wind_speed,
        f"{prefix}_wind_u_latest": wind_u,
        f"{prefix}_wind_v_latest": wind_v,
        f"{prefix}_wind_dir_shift_1h_deg": wind_shift_1h,
        f"{prefix}_wind_dir_shift_3h_deg": wind_shift_3h,
        f"{prefix}_wind_speed_trend_1h": wind_speed_trend_1h,
        f"{prefix}_wind_speed_trend_3h": wind_speed_trend_3h,
        f"{prefix}_temp_trend_1h": temp_trend_1h,
        f"{prefix}_temp_trend_3h": temp_trend_3h,
        f"{prefix}_dewpoint_trend_1h": dewpoint_trend_1h,
        f"{prefix}_dewpoint_trend_3h": dewpoint_trend_3h,
        f"{prefix}_dewpoint_depression_latest": dewpoint_depression,
        f"{prefix}_dewpoint_depression_trend_3h": dewpoint_depression_trend_3h,
        f"{prefix}_pressure_tendency_1h": pressure_tendency_1h,
        f"{prefix}_pressure_tendency_3h": pressure_tendency_3h,
        f"{prefix}_north_sector_latest": north,
        f"{prefix}_east_sector_latest": east,
        f"{prefix}_south_sector_latest": south,
        f"{prefix}_west_sector_latest": west,
        f"{prefix}_cold_advection_signal": cold_signal,
        f"{prefix}_warm_advection_signal": warm_signal,
        f"{prefix}_frontal_passage_signal": frontal_signal,
    }
    summary = {
        "available": True,
        "wind_u_latest": wind_u,
        "wind_v_latest": wind_v,
        "wind_speed_latest_kt": wind_speed,
        "temp_trend_1h": temp_trend_1h,
        "temp_trend_3h": temp_trend_3h,
        "dewpoint_trend_3h": dewpoint_trend_3h,
        "pressure_tendency_3h": pressure_tendency_3h,
        "north_sector_latest": north,
        "south_sector_latest": south,
        "cold_advection_signal": cold_signal,
        "warm_advection_signal": warm_signal,
        "frontal_passage_signal": frontal_signal,
    }
    return features, summary, so_far["knowledge_time_utc"].max()


def _empty_station_features(prefix: str) -> dict:
    return {
        f"{prefix}_available": False,
        f"{prefix}_wind_dir_latest_deg": np.nan,
        f"{prefix}_wind_speed_latest_kt": np.nan,
        f"{prefix}_wind_u_latest": np.nan,
        f"{prefix}_wind_v_latest": np.nan,
        f"{prefix}_wind_dir_shift_1h_deg": np.nan,
        f"{prefix}_wind_dir_shift_3h_deg": np.nan,
        f"{prefix}_wind_speed_trend_1h": np.nan,
        f"{prefix}_wind_speed_trend_3h": np.nan,
        f"{prefix}_temp_trend_1h": np.nan,
        f"{prefix}_temp_trend_3h": np.nan,
        f"{prefix}_dewpoint_trend_1h": np.nan,
        f"{prefix}_dewpoint_trend_3h": np.nan,
        f"{prefix}_dewpoint_depression_latest": np.nan,
        f"{prefix}_dewpoint_depression_trend_3h": np.nan,
        f"{prefix}_pressure_tendency_1h": np.nan,
        f"{prefix}_pressure_tendency_3h": np.nan,
        f"{prefix}_north_sector_latest": False,
        f"{prefix}_east_sector_latest": False,
        f"{prefix}_south_sector_latest": False,
        f"{prefix}_west_sector_latest": False,
        f"{prefix}_cold_advection_signal": False,
        f"{prefix}_warm_advection_signal": False,
        f"{prefix}_frontal_passage_signal": False,
    }


def _prepare_metar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["observation_time_utc", "knowledge_time_utc", "temperature_c"])
    df = frame.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["knowledge_time_utc"] = pd.to_datetime(df.get("knowledge_time_utc", df["observation_time_utc"]), utc=True, errors="coerce")
    for column in ["temperature_c", "dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["observation_time_utc", "knowledge_time_utc", "temperature_c"]).sort_values("observation_time_utc")


def _window(df: pd.DataFrame, issue_utc: pd.Timestamp, hours: float) -> pd.DataFrame:
    return df[df["observation_time_utc"] >= issue_utc - pd.Timedelta(hours=hours)]


def _trend(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.iloc[-1] - values.iloc[0])


def _series_trend(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.iloc[-1] - clean.iloc[0])


def _wind_shift(df: pd.DataFrame) -> float:
    values = pd.to_numeric(df.get("wind_direction_deg"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    diff = abs(values[-1] - values[0]) % 360
    return float(min(diff, 360 - diff))


def _wind_components(direction_deg: float, speed_kt: float) -> tuple[float, float]:
    if not np.isfinite(direction_deg) or not np.isfinite(speed_kt):
        return float("nan"), float("nan")
    rad = np.deg2rad(direction_deg)
    return float(-speed_kt * np.sin(rad)), float(-speed_kt * np.cos(rad))


def _sector_flags(direction_deg: float) -> tuple[bool, bool, bool, bool]:
    if not np.isfinite(direction_deg):
        return False, False, False, False
    direction = direction_deg % 360
    north = direction >= 315 or direction < 45
    east = 45 <= direction < 135
    south = 135 <= direction < 225
    west = 225 <= direction < 315
    return bool(north), bool(east), bool(south), bool(west)


def _finite_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _mean_or_nan(values) -> float:
    clean = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(clean)) if clean else float("nan")


def _maybe_diff(left, right) -> float:
    if left is None or right is None:
        return float("nan")
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return float("nan")
    if not np.isfinite(left_float) or not np.isfinite(right_float):
        return float("nan")
    return left_float - right_float


def _lt(value, threshold: float) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) < threshold


def _gt(value, threshold: float) -> bool:
    return value is not None and np.isfinite(float(value)) and float(value) > threshold
