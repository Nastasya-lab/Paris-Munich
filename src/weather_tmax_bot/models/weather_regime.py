from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REGIME_LABELS = [
    "clear_heating",
    "cloud_limited",
    "frontal_rain",
    "convective",
    "late_clearing",
    "cold_advection",
    "heatwave",
]


@dataclass(frozen=True)
class WeatherRegimePrediction:
    label: str
    scores: dict[str, float]


def detect_weather_regime(row: dict | pd.Series) -> WeatherRegimePrediction:
    """Classify a forecast row into a physically interpretable weather regime.

    This detector deliberately uses only as-of feature columns that are already
    present in the LFPB METAR/NWP dataset. It is a transparent research layer,
    not a replacement for the probabilistic model.
    """

    cloud_now = _cloud_fraction(_num(row, "cloud_cover_proxy_latest", 0.0))
    cloud_future = _cloud_fraction(_first_num(row, ["model_future_cloud_cover_mean", "model_cloud_cover_mean"], 0.0))
    radiation_future = _num(row, "model_future_shortwave_radiation_sum", 0.0)
    if radiation_future == 0.0:
        radiation_future = _num(row, "model_shortwave_radiation_sum", 0.0)
    precip_future = _first_num(row, ["model_future_precip_sum", "model_precip_sum"], 0.0)
    precip_total = _num(row, "model_precip_sum", 0.0) + precip_future
    temp_trend_1h = _num(row, "temp_trend_1h", 0.0)
    temp_trend_3h = _num(row, "temp_trend_3h", 0.0)
    temp_slope = _num(row, "temp_slope_since_sunrise", 0.0)
    current_max = _first_num(row, ["current_metar_max_c", "observed_max_so_far_from_metar"], np.nan)
    last_temp = _first_num(row, ["latest_metar_temp_c", "last_metar_temp_c"], np.nan)
    inferred_drop = 0.0 if np.isnan(current_max) or np.isnan(last_temp) else max(0.0, current_max - last_temp)
    drop = _num(row, "drop_from_current_max_c", inferred_drop)
    dewpoint_dep = _num(row, "dewpoint_depression_latest", 0.0)
    dewpoint_dep_trend = _num(row, "dewpoint_depression_trend_2h", 0.0)
    pressure_3h = _num(row, "pressure_tendency_3h", 0.0)
    wind_shift = abs(_num(row, "wind_dir_shift_2h_deg", 0.0))
    model_tmax = _num(row, "model_tmax_c", np.nan)
    future_minus_current = _num(row, "nwp_future_minus_current_max_c", np.nan)
    if np.isnan(future_minus_current) and not np.isnan(model_tmax) and not np.isnan(current_max):
        future_minus_current = model_tmax - current_max

    rain_recent = (
        _flag(row, "has_rain_recent_metar")
        or _flag(row, "has_precip_recent")
        or _flag(row, "rain_started_after_current_max")
    )
    thunder_recent = _flag(row, "has_thunder_recent_metar") or _flag(row, "has_thunder_recent")
    convective_markers = (
        thunder_recent
        or _flag(row, "cb_tcu_appeared_after_current_max")
        or _flag(row, "showers_appeared_after_current_max")
    )
    cavok = _flag(row, "is_cavok_latest")
    rain_after_peak = _flag(row, "rain_started_after_current_max")

    cloud_improving = cloud_now >= 0.45 and cloud_future <= max(0.35, cloud_now - 0.20)
    still_heating = temp_trend_1h > 0.2 or temp_slope > 0.20 or (not np.isnan(future_minus_current) and future_minus_current > 0.8)
    dry_heating = dewpoint_dep >= 6.0 and dewpoint_dep_trend >= -1.0
    low_cloud = max(cloud_now, cloud_future) <= 0.35
    high_cloud = max(cloud_now, cloud_future) >= 0.65
    low_radiation = radiation_future <= 450.0
    good_radiation = radiation_future >= 900.0
    rainy = rain_recent or precip_future >= 0.3 or precip_total >= 0.8
    sharp_drop = drop >= 2.5 or temp_trend_3h <= -1.8

    scores = {
        "convective": 0.25
        + 0.45 * float(convective_markers)
        + 0.20 * float(rain_recent and sharp_drop)
        + 0.10 * float(wind_shift >= 80),
        "frontal_rain": 0.15
        + 0.35 * float(rainy)
        + 0.25 * float(sharp_drop)
        + 0.15 * float(high_cloud)
        + 0.10 * float(pressure_3h <= -0.8),
        "late_clearing": 0.10
        + 0.35 * float(cloud_improving)
        + 0.25 * float(good_radiation)
        + 0.15 * float(precip_future <= 0.2)
        + 0.15 * float(still_heating),
        "cloud_limited": 0.15
        + 0.35 * float(high_cloud)
        + 0.25 * float(low_radiation)
        + 0.15 * float(temp_trend_3h <= 0.5)
        + 0.10 * float(dewpoint_dep <= 5.0),
        "cold_advection": 0.10
        + 0.30 * float(temp_trend_3h <= -1.5)
        + 0.20 * float(dewpoint_dep_trend <= -1.5)
        + 0.20 * float(pressure_3h >= 0.8)
        + 0.20 * float((not np.isnan(future_minus_current)) and future_minus_current <= 0.2),
        "heatwave": 0.05
        + 0.45 * float((not np.isnan(model_tmax)) and model_tmax >= 30.0)
        + 0.25 * float((not np.isnan(current_max)) and current_max >= 28.0)
        + 0.15 * float(low_cloud)
        + 0.10 * float(not rainy),
        "clear_heating": 0.15
        + 0.30 * float(cavok or low_cloud)
        + 0.20 * float(good_radiation)
        + 0.20 * float(still_heating)
        + 0.10 * float(dry_heating)
        + 0.05 * float(not rainy),
    }

    # Keep regimes mutually interpretable: rain after the current max should not
    # be classified as pure clear heating even if the NWP still has radiation.
    if rain_after_peak:
        scores["clear_heating"] *= 0.45
        scores["heatwave"] *= 0.70
    if convective_markers:
        scores["frontal_rain"] *= 0.75
    if rainy and high_cloud and not convective_markers:
        scores["convective"] *= 0.60
    if rainy and sharp_drop and high_cloud and not convective_markers:
        scores["frontal_rain"] = min(1.0, scores["frontal_rain"] + 0.20)
        scores["cloud_limited"] *= 0.55
    if cavok and not rainy and low_cloud:
        scores["cloud_limited"] *= 0.55
        scores["frontal_rain"] *= 0.50

    scores = {key: float(np.clip(value, 0.0, 1.0)) for key, value in scores.items()}
    label = max(REGIME_LABELS, key=lambda key: scores.get(key, 0.0))
    return WeatherRegimePrediction(label=label, scores=scores)


def add_weather_regime_columns(frame: pd.DataFrame, prefix: str = "weather_regime") -> pd.DataFrame:
    out = frame.copy()
    predictions = [detect_weather_regime(row) for _, row in out.iterrows()]
    out[prefix] = [item.label for item in predictions]
    for label in REGIME_LABELS:
        out[f"{prefix}_{label}_score"] = [item.scores.get(label, 0.0) for item in predictions]
        out[f"{prefix}_is_{label}"] = [item.label == label for item in predictions]
    return out


def _num(row: dict | pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if value is None or pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_num(row: dict | pd.Series, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        value = _num(row, key, np.nan)
        if not np.isnan(value):
            return value
    return float(default)


def _cloud_fraction(value: float) -> float:
    """Normalize METAR octas or NWP percent cloud values to 0..1."""
    if value is None or np.isnan(value):
        return 0.0
    if value > 8.0:
        return float(np.clip(value / 100.0, 0.0, 1.0))
    if value > 1.0:
        return float(np.clip(value / 8.0, 0.0, 1.0))
    return float(np.clip(value, 0.0, 1.0))


def _flag(row: dict | pd.Series, key: str) -> bool:
    value = row.get(key, False)
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)
