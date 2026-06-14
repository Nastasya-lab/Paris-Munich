from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


DEFAULT_SPATIAL_STATIONS = ["LFPG", "LFPO"]
EDDM_SPATIAL_STATIONS = ["EDMO", "EDMA", "ETSI", "ETSL"]
SPATIAL_STATIONS_BY_AIRPORT = {
    "EDDM": EDDM_SPATIAL_STATIONS,
    "LFPB": DEFAULT_SPATIAL_STATIONS,
}


def spatial_feature_columns(stations: list[str] | tuple[str, ...] = DEFAULT_SPATIAL_STATIONS) -> list[str]:
    columns: list[str] = []
    for station in stations:
        prefix = f"spatial_{station.lower()}"
        columns.extend(
            [
                f"{prefix}_available",
                f"{prefix}_latest_temp_c",
                f"{prefix}_current_max_c",
                f"{prefix}_drop_from_current_max_c",
                f"{prefix}_temp_trend_1h",
                f"{prefix}_temp_trend_3h",
                f"{prefix}_temp_trend_last_2_metars",
                f"{prefix}_dewpoint_depression_latest",
                f"{prefix}_cloud_cover_proxy_latest",
                f"{prefix}_has_rain_recent",
                f"{prefix}_has_thunder_recent",
                f"{prefix}_is_cavok_latest",
                f"{prefix}_age_minutes",
                f"{prefix}_count_so_far",
                f"{prefix}_count_last_1h",
                f"{prefix}_count_last_3h",
            ]
        )
    columns.extend(
        [
            "spatial_available_station_count",
            "spatial_latest_temp_mean_c",
            "spatial_latest_temp_max_c",
            "spatial_latest_temp_min_c",
            "spatial_latest_temp_spread_c",
            "spatial_current_max_mean_c",
            "spatial_current_max_max_c",
            "spatial_current_max_min_c",
            "spatial_current_max_spread_c",
            "spatial_latest_minus_base_latest_mean_c",
            "spatial_max_minus_base_current_max_mean_c",
            "spatial_latest_minus_lfpb_latest_mean_c",
            "spatial_max_minus_lfpb_current_max_mean_c",
            "spatial_any_neighbor_above_lfpb_latest",
            "spatial_any_neighbor_above_lfpb_current_max",
        ]
    )
    return columns


def build_spatial_metar_features(
    base_row: dict | pd.Series,
    neighbor_metars: dict[str, pd.DataFrame],
    *,
    target_date_local: date,
    issue_time_utc,
    timezone_name: str,
    stations: list[str] | tuple[str, ...] = DEFAULT_SPATIAL_STATIONS,
) -> dict:
    issue = pd.Timestamp(issue_time_utc).tz_convert("UTC")
    features: dict[str, float | int | bool | str | None] = {}
    latest_values: list[float] = []
    max_values: list[float] = []
    available = 0
    max_knowledge_time = _optional_timestamp(base_row.get("max_feature_knowledge_time_utc")) or issue
    base_latest = _first_optional_float(base_row, ["latest_metar_temp_c", "last_metar_temp_c"])
    base_current_max = _first_optional_float(base_row, ["current_metar_max_c", "observed_max_so_far_from_metar"])

    for station in stations:
        prefix = f"spatial_{station.lower()}"
        station_features, station_latest, station_max, station_knowledge = _station_features(
            _prepare_metar(neighbor_metars.get(station, pd.DataFrame())),
            issue_time_utc=issue,
            target_date_local=target_date_local,
            timezone_name=timezone_name,
            prefix=prefix,
        )
        features.update(station_features)
        if station_latest is not None:
            latest_values.append(station_latest)
            available += 1
        if station_max is not None:
            max_values.append(station_max)
        if station_knowledge is not None:
            max_knowledge_time = max(max_knowledge_time, station_knowledge)
        features[f"{prefix}_latest_minus_base_latest_c"] = _maybe_diff(station_latest, base_latest)
        features[f"{prefix}_max_minus_base_current_max_c"] = _maybe_diff(station_max, base_current_max)
        features[f"{prefix}_latest_minus_lfpb_latest_c"] = features[f"{prefix}_latest_minus_base_latest_c"]
        features[f"{prefix}_max_minus_lfpb_current_max_c"] = features[f"{prefix}_max_minus_base_current_max_c"]

    features["spatial_available_station_count"] = available
    features["spatial_latest_temp_mean_c"] = _mean_or_nan(latest_values)
    features["spatial_latest_temp_max_c"] = _max_or_nan(latest_values)
    features["spatial_latest_temp_min_c"] = _min_or_nan(latest_values)
    features["spatial_latest_temp_spread_c"] = _spread_or_nan(latest_values)
    features["spatial_current_max_mean_c"] = _mean_or_nan(max_values)
    features["spatial_current_max_max_c"] = _max_or_nan(max_values)
    features["spatial_current_max_min_c"] = _min_or_nan(max_values)
    features["spatial_current_max_spread_c"] = _spread_or_nan(max_values)
    features["spatial_latest_minus_base_latest_mean_c"] = _maybe_diff(features["spatial_latest_temp_mean_c"], base_latest)
    features["spatial_max_minus_base_current_max_mean_c"] = _maybe_diff(features["spatial_current_max_mean_c"], base_current_max)
    features["spatial_latest_minus_lfpb_latest_mean_c"] = features["spatial_latest_minus_base_latest_mean_c"]
    features["spatial_max_minus_lfpb_current_max_mean_c"] = features["spatial_max_minus_base_current_max_mean_c"]
    features["spatial_any_neighbor_above_lfpb_latest"] = bool(base_latest is not None and any(value > base_latest for value in latest_values))
    features["spatial_any_neighbor_above_lfpb_current_max"] = bool(
        base_current_max is not None and any(value > base_current_max for value in max_values)
    )
    features["spatial_max_feature_knowledge_time_utc"] = max_knowledge_time.isoformat()
    features["spatial_leakage_check_passed"] = bool(max_knowledge_time <= issue)
    return features


def add_spatial_metar_features_to_frame(
    frame: pd.DataFrame,
    neighbor_metars: dict[str, pd.DataFrame],
    *,
    timezone_name: str,
    stations: list[str] | tuple[str, ...] = DEFAULT_SPATIAL_STATIONS,
) -> pd.DataFrame:
    out = frame.copy()
    station_day_metars = _station_day_metar_map(neighbor_metars, timezone_name=timezone_name, stations=stations)
    rows = [
        build_spatial_metar_features(
            row,
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
    spatial = pd.DataFrame(rows, index=out.index)
    out = pd.concat([out, spatial], axis=1)
    out["max_feature_knowledge_time_utc"] = out["spatial_max_feature_knowledge_time_utc"]
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool) & out["spatial_leakage_check_passed"].fillna(False).astype(bool)
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
        out[station] = {
            str(day): group.drop(columns=["_target_date_local"]).copy()
            for day, group in df.groupby("_target_date_local", sort=False)
        }
    return out


def _station_features(
    frame: pd.DataFrame,
    *,
    issue_time_utc: pd.Timestamp,
    target_date_local: date,
    timezone_name: str,
    prefix: str,
) -> tuple[dict, float | None, float | None, pd.Timestamp | None]:
    features = {
        f"{prefix}_available": False,
        f"{prefix}_latest_temp_c": np.nan,
        f"{prefix}_current_max_c": np.nan,
        f"{prefix}_drop_from_current_max_c": np.nan,
        f"{prefix}_temp_trend_1h": np.nan,
        f"{prefix}_temp_trend_3h": np.nan,
        f"{prefix}_temp_trend_last_2_metars": np.nan,
        f"{prefix}_dewpoint_depression_latest": np.nan,
        f"{prefix}_cloud_cover_proxy_latest": np.nan,
        f"{prefix}_has_rain_recent": False,
        f"{prefix}_has_thunder_recent": False,
        f"{prefix}_is_cavok_latest": False,
        f"{prefix}_age_minutes": np.nan,
        f"{prefix}_count_so_far": 0,
        f"{prefix}_count_last_1h": 0,
        f"{prefix}_count_last_3h": 0,
    }
    if frame.empty:
        return features, None, None, None
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    so_far = frame[
        (frame["observation_time_utc"] >= pd.Timestamp(day_start))
        & (frame["observation_time_utc"] < pd.Timestamp(day_end))
        & (frame["knowledge_time_utc"] <= issue_time_utc)
    ].copy()
    if so_far.empty:
        return features, None, None, None
    latest = so_far.iloc[-1]
    latest_temp = float(latest["temperature_c"])
    current_max = float(so_far["temperature_c"].max())
    features.update(
        {
            f"{prefix}_available": True,
            f"{prefix}_latest_temp_c": latest_temp,
            f"{prefix}_current_max_c": current_max,
            f"{prefix}_drop_from_current_max_c": float(current_max - latest_temp),
            f"{prefix}_temp_trend_1h": _trend(_window(so_far, issue_time_utc, 1), "temperature_c"),
            f"{prefix}_temp_trend_3h": _trend(_window(so_far, issue_time_utc, 3), "temperature_c"),
            f"{prefix}_temp_trend_last_2_metars": _trend(so_far.tail(2), "temperature_c"),
            f"{prefix}_dewpoint_depression_latest": _finite_or_nan(latest_temp - _finite_or_nan(latest.get("dewpoint_c"))),
            f"{prefix}_cloud_cover_proxy_latest": _cloud_cover_proxy(latest),
            f"{prefix}_has_rain_recent": _has_weather(_window(so_far, issue_time_utc, 3), ["RA", "SHRA", "TSRA"]),
            f"{prefix}_has_thunder_recent": _has_weather(_window(so_far, issue_time_utc, 6), ["TS"]),
            f"{prefix}_is_cavok_latest": bool(latest.get("cavok", False)),
            f"{prefix}_age_minutes": float((issue_time_utc - latest["knowledge_time_utc"]).total_seconds() / 60.0),
            f"{prefix}_count_so_far": int(len(so_far)),
            f"{prefix}_count_last_1h": int(_window(so_far, issue_time_utc, 1).shape[0]),
            f"{prefix}_count_last_3h": int(_window(so_far, issue_time_utc, 3).shape[0]),
        }
    )
    return features, latest_temp, current_max, so_far["knowledge_time_utc"].max()


def _prepare_metar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["observation_time_utc", "knowledge_time_utc", "temperature_c"])
    df = frame.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["knowledge_time_utc"] = pd.to_datetime(df.get("knowledge_time_utc", df["observation_time_utc"]), utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    for column in ["dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt", "ceiling_ft"]:
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


def _has_weather(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    return any(code in text for code in codes)


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


def _optional_timestamp(value) -> pd.Timestamp | None:
    try:
        if value is None or pd.isna(value):
            return None
        return pd.Timestamp(value).tz_convert("UTC")
    except (TypeError, ValueError):
        return None


def _optional_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_optional_float(row: dict | pd.Series, keys: list[str]) -> float | None:
    for key in keys:
        value = _optional_float(row.get(key))
        if value is not None:
            return value
    return None


def _finite_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _min_or_nan(values: list[float]) -> float:
    return float(np.min(values)) if values else float("nan")


def _spread_or_nan(values: list[float]) -> float:
    return float(np.max(values) - np.min(values)) if len(values) >= 2 else float("nan")


def _maybe_diff(left, right: float | None) -> float:
    if left is None or right is None or pd.isna(left):
        return float("nan")
    return float(left) - float(right)
