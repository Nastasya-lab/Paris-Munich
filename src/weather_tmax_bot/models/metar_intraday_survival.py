from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution


@dataclass(frozen=True)
class MetarIntradaySurvivalAdjustment:
    distribution: TmaxDistribution
    active: bool
    details: dict


def apply_metar_intraday_survival_layer(
    distribution: TmaxDistribution,
    feature_row: dict | pd.Series,
    *,
    historical_dataset: pd.DataFrame | None = None,
    historical_dataset_path: str | Path | None = None,
    strength: float = 0.92,
    max_threshold_c: int = 6,
    min_context_rows: int = 30,
) -> MetarIntradaySurvivalAdjustment:
    """Constrain same-day METAR Tmax upside using intraday survival signals.

    The trained METAR/ICON model estimates the final daily maximum. This layer is
    deliberately a post-processor: it only reduces implausible upside mass when
    as-of evidence says the current METAR maximum is unlikely to be exceeded.
    """
    observed_max = _optional_float(feature_row, "current_metar_max_c")
    if observed_max is None:
        return MetarIntradaySurvivalAdjustment(
            distribution=distribution,
            active=False,
            details={"active": False, "reason": "missing_current_metar_max_c"},
        )

    observed_bin = int(np.ceil(observed_max))
    base = _truncate_below_observed_bin(distribution, observed_bin)
    max_upside = max(1, int(max(base.bins_c.max() - observed_bin, max_threshold_c)))
    original_survival = _distribution_to_survival(base, observed_bin, max_upside)

    priors = _historical_survival_priors(
        feature_row,
        historical_dataset=historical_dataset,
        historical_dataset_path=historical_dataset_path,
        max_threshold_c=max_upside,
        min_context_rows=min_context_rows,
    )
    context_caps = _context_survival_caps(feature_row, observed_bin=observed_bin, max_threshold_c=max_upside)
    target_caps = {
        threshold: min(priors.get(threshold, 1.0), context_caps.get(threshold, 1.0))
        for threshold in range(1, max_upside + 1)
    }
    dynamic_strength = float(np.clip(strength * _phase_strength(feature_row), 0.0, 1.0))
    adjusted_survival = {}
    for threshold in range(1, max_upside + 1):
        original = float(np.clip(original_survival.get(threshold, 0.0), 0.0, 1.0))
        cap = float(np.clip(target_caps.get(threshold, 1.0), 0.0, 1.0))
        adjusted_survival[threshold] = original - dynamic_strength * max(0.0, original - cap)
    rebound_guard = _convective_rebound_guard(
        feature_row,
        original_survival=original_survival,
        adjusted_survival=adjusted_survival,
        max_threshold_c=max_upside,
    )
    if rebound_guard["active"]:
        for threshold, floor in rebound_guard["floors"].items():
            threshold = int(threshold)
            if threshold <= max_upside:
                adjusted_survival[threshold] = max(float(adjusted_survival.get(threshold, 0.0)), float(floor))

    adjusted_values = np.minimum.accumulate(
        np.clip([adjusted_survival[threshold] for threshold in range(1, max_upside + 1)], 0.0, 1.0)
    )
    adjusted_survival = {
        threshold: float(adjusted_values[threshold - 1]) for threshold in range(1, max_upside + 1)
    }
    adjusted = _survival_to_distribution(adjusted_survival, observed_bin, max_upside)
    details = {
        "active": True,
        "layer_version": "lfpb_metar_intraday_survival_v2_rebound_guard",
        "observed_max_bin_c": observed_bin,
        "observed_max_so_far_c": float(observed_max),
        "local_issue_hour": _optional_float(feature_row, "local_issue_hour"),
        "latest_metar_temp_c": _optional_float(feature_row, "latest_metar_temp_c"),
        "drop_from_current_max_c": _optional_float(feature_row, "drop_from_current_max_c"),
        "has_rain_recent_metar": bool(feature_row.get("has_rain_recent_metar", False)),
        "nwp_future_minus_current_max_c": _optional_float(feature_row, "nwp_future_minus_current_max_c"),
        "model_future_temp_max_c": _optional_float(feature_row, "model_future_temp_max_c"),
        "strength_requested": float(strength),
        "phase_strength": _phase_strength(feature_row),
        "effective_strength": dynamic_strength,
        "original_probability_upside_ge_1c": original_survival.get(1, 0.0),
        "adjusted_probability_upside_ge_1c": adjusted_survival.get(1, 0.0),
        "original_probability_upside_ge_2c": original_survival.get(2, 0.0),
        "adjusted_probability_upside_ge_2c": adjusted_survival.get(2, 0.0),
        "original_probability_upside_ge_3c": original_survival.get(3, 0.0),
        "adjusted_probability_upside_ge_3c": adjusted_survival.get(3, 0.0),
        "historical_survival_priors": {str(k): float(v) for k, v in priors.items() if k <= 3},
        "context_survival_caps": {str(k): float(v) for k, v in context_caps.items() if k <= 3},
        "target_survival_caps": {str(k): float(v) for k, v in target_caps.items() if k <= 3},
        "rebound_guard": rebound_guard,
        "formula": "survival_cap_blend_from_historical_hourly_prior_and_live_context",
    }
    return MetarIntradaySurvivalAdjustment(distribution=adjusted, active=True, details=details)


def _historical_survival_priors(
    feature_row: dict | pd.Series,
    *,
    historical_dataset: pd.DataFrame | None,
    historical_dataset_path: str | Path | None,
    max_threshold_c: int,
    min_context_rows: int,
) -> dict[int, float]:
    frame = historical_dataset
    if frame is None and historical_dataset_path is not None and Path(historical_dataset_path).exists():
        frame = pd.read_parquet(historical_dataset_path)
    if frame is None or frame.empty or "remaining_upside_c" not in frame.columns:
        return {threshold: 1.0 for threshold in range(1, max_threshold_c + 1)}

    data = frame.copy()
    data["remaining_upside_c"] = pd.to_numeric(data["remaining_upside_c"], errors="coerce")
    data["local_issue_hour"] = pd.to_numeric(data.get("local_issue_hour"), errors="coerce")
    data = data.dropna(subset=["remaining_upside_c", "local_issue_hour"])
    if data.empty:
        return {threshold: 1.0 for threshold in range(1, max_threshold_c + 1)}
    data["season"] = data.get("season", pd.Series(index=data.index, dtype=object)).fillna(
        pd.to_datetime(data.get("target_date_local"), errors="coerce").dt.month.map(_season)
    )
    target_season = _season(pd.Timestamp(str(feature_row.get("target_date_local"))).month)
    local_hour = _optional_float(feature_row, "local_issue_hour")
    if local_hour is None:
        local_hour = 12.0
    context = data[data["season"] == target_season]
    if len(context) < min_context_rows:
        context = data
    return _interpolated_hourly_survival(context, local_hour, max_threshold_c)


def _interpolated_hourly_survival(frame: pd.DataFrame, local_hour: float, max_threshold_c: int) -> dict[int, float]:
    hours = sorted(float(hour) for hour in frame["local_issue_hour"].dropna().unique())
    if not hours:
        return {threshold: 1.0 for threshold in range(1, max_threshold_c + 1)}
    lower = max((hour for hour in hours if hour <= local_hour), default=hours[0])
    upper = min((hour for hour in hours if hour >= local_hour), default=hours[-1])
    lower_probs = _survival_for_group(frame[frame["local_issue_hour"] == lower], max_threshold_c)
    upper_probs = _survival_for_group(frame[frame["local_issue_hour"] == upper], max_threshold_c)
    if lower == upper:
        return lower_probs
    weight = float((local_hour - lower) / (upper - lower))
    return {
        threshold: float((1.0 - weight) * lower_probs[threshold] + weight * upper_probs[threshold])
        for threshold in range(1, max_threshold_c + 1)
    }


def _survival_for_group(group: pd.DataFrame, max_threshold_c: int) -> dict[int, float]:
    upside = pd.to_numeric(group["remaining_upside_c"], errors="coerce").dropna()
    if upside.empty:
        return {threshold: 1.0 for threshold in range(1, max_threshold_c + 1)}
    return {threshold: float((upside >= threshold).mean()) for threshold in range(1, max_threshold_c + 1)}


def _context_survival_caps(feature_row: dict | pd.Series, *, observed_bin: int, max_threshold_c: int) -> dict[int, float]:
    future_delta = _optional_float(feature_row, "nwp_future_minus_current_max_c")
    if future_delta is None:
        model_future = _optional_float(feature_row, "model_future_temp_max_c")
        current_max = _optional_float(feature_row, "current_metar_max_c")
        future_delta = None if model_future is None or current_max is None else model_future - current_max
    drop = _optional_float(feature_row, "drop_from_current_max_c") or 0.0
    trend_1h = _optional_float(feature_row, "temp_trend_1h")
    trend_3h = _optional_float(feature_row, "temp_trend_3h")
    rain_recent = bool(feature_row.get("has_rain_recent_metar", False))
    local_hour = _optional_float(feature_row, "local_issue_hour") or 12.0

    caps = {}
    for threshold in range(1, max_threshold_c + 1):
        if future_delta is None:
            nwp_cap = 1.0
        else:
            # Probability-like cap that falls quickly when future NWP max is
            # below the amount required to beat the observed METAR maximum.
            nwp_cap = _sigmoid((future_delta - (threshold - 0.35)) / 0.55)
        trend_multiplier = 1.0
        if drop >= 1.0:
            trend_multiplier *= float(np.exp(-0.30 * min(drop, 6.0)))
        if trend_1h is not None and trend_1h <= -1.0:
            trend_multiplier *= 0.70
        if trend_3h is not None and trend_3h <= -2.0:
            trend_multiplier *= 0.65
        if rain_recent:
            trend_multiplier *= 0.55 if drop >= 1.0 else 0.75
        if local_hour >= 17:
            trend_multiplier *= 0.80
        if local_hour >= 18:
            trend_multiplier *= 0.65
        caps[threshold] = float(np.clip(nwp_cap * trend_multiplier, 0.0, 1.0))
    return caps


def _phase_strength(feature_row: dict | pd.Series) -> float:
    hour = _optional_float(feature_row, "local_issue_hour")
    if hour is None:
        return 0.35
    if hour < 10:
        return 0.10
    if hour < 12:
        return 0.20
    if hour < 14:
        return 0.35
    if hour < 16:
        return 0.55
    if hour < 18:
        return 0.80
    return 0.95


def _convective_rebound_guard(
    feature_row: dict | pd.Series,
    *,
    original_survival: dict[int, float],
    adjusted_survival: dict[int, float],
    max_threshold_c: int,
) -> dict:
    """Keep midday convective rebounds from being mistaken for a finished day.

    The normal intraday layer is intentionally conservative after rain/CB/TCU.
    That is good late in the day, but around early afternoon a shower can create
    a temporary temperature dip followed by a fast rebound. In that narrow case
    we apply a floor to +1C/+2C upside probabilities instead of fully trusting
    the shutdown cap.
    """
    local_hour = _optional_float(feature_row, "local_issue_hour")
    if local_hour is None or not (11.5 <= local_hour <= 15.5):
        return _inactive_rebound_guard("outside_midday_rebound_window")

    current_max = _optional_float(feature_row, "current_metar_max_c")
    latest_temp = _optional_float(feature_row, "latest_metar_temp_c")
    if current_max is None or latest_temp is None:
        return _inactive_rebound_guard("missing_current_or_latest_temperature")

    trend_1h = _optional_float(feature_row, "temp_trend_1h")
    trend_last_2 = _optional_float(feature_row, "temp_trend_last_2_metars")
    trend_3h = _optional_float(feature_row, "temp_trend_3h")
    minutes_since_max = _optional_float(feature_row, "metar_minutes_since_current_max")
    future_delta = _optional_float(feature_row, "nwp_future_minus_current_max_c")
    if future_delta is None:
        model_future = _optional_float(feature_row, "model_future_temp_max_c")
        future_delta = None if model_future is None else model_future - current_max

    convective_signal = any(
        bool(feature_row.get(key, False))
        for key in [
            "has_rain_recent_metar",
            "rain_started_after_current_max",
            "cb_tcu_appeared_after_current_max",
            "showers_appeared_after_current_max",
        ]
    )
    strong_rebound = max(
        trend_1h if trend_1h is not None else -99.0,
        trend_last_2 if trend_last_2 is not None else -99.0,
    )
    near_current_max = latest_temp >= current_max - 0.35
    recent_or_new_max = minutes_since_max is None or minutes_since_max <= 45.0

    if not convective_signal:
        return _inactive_rebound_guard("no_convective_or_rain_signal")
    if strong_rebound < 2.0:
        return _inactive_rebound_guard("no_strong_temperature_rebound")
    if not near_current_max:
        return _inactive_rebound_guard("latest_temperature_not_back_near_current_max")
    if not recent_or_new_max:
        return _inactive_rebound_guard("current_max_not_recent")

    floor_1c = 0.34
    if strong_rebound >= 3.0:
        floor_1c += 0.08
    if local_hour < 14.5:
        floor_1c += 0.04
    if trend_3h is not None and trend_3h >= 1.5:
        floor_1c += 0.04
    if future_delta is not None:
        if future_delta < -1.0:
            floor_1c -= 0.12
        elif future_delta < -0.25:
            floor_1c -= 0.05
        elif future_delta > 0.5:
            floor_1c += 0.05
    floor_1c = float(np.clip(floor_1c, 0.24, 0.55))

    # Keep +2C possible but modest: the guard is a rebound safety net, not a
    # heat-spike booster.
    floor_2c = float(np.clip(floor_1c * 0.18, 0.03, 0.12))
    floors = {1: floor_1c}
    if max_threshold_c >= 2:
        floors[2] = min(floor_2c, floor_1c)

    changed = any(float(adjusted_survival.get(threshold, 0.0)) < floor for threshold, floor in floors.items())
    return {
        "active": bool(changed),
        "eligible": True,
        "reason": "convective_temporary_dip_rebound_guard",
        "floors": {str(key): float(value) for key, value in floors.items()},
        "strong_rebound_c": float(strong_rebound),
        "trend_1h_c": None if trend_1h is None else float(trend_1h),
        "trend_last_2_metars_c": None if trend_last_2 is None else float(trend_last_2),
        "trend_3h_c": None if trend_3h is None else float(trend_3h),
        "minutes_since_current_max": None if minutes_since_max is None else float(minutes_since_max),
        "nwp_future_minus_current_max_c": None if future_delta is None else float(future_delta),
        "original_probability_upside_ge_1c": float(original_survival.get(1, 0.0)),
        "pre_guard_adjusted_probability_upside_ge_1c": float(adjusted_survival.get(1, 0.0)),
    }


def _inactive_rebound_guard(reason: str) -> dict:
    return {"active": False, "eligible": False, "reason": reason, "floors": {}}


def _distribution_to_survival(distribution: TmaxDistribution, observed_bin: int, max_upside: int) -> dict[int, float]:
    return {
        threshold: float(distribution.probabilities[distribution.bins_c >= observed_bin + threshold].sum())
        for threshold in range(1, max_upside + 1)
    }


def _survival_to_distribution(survival: dict[int, float], observed_bin: int, max_upside: int) -> TmaxDistribution:
    values = np.minimum.accumulate(
        np.clip([survival.get(threshold, 0.0) for threshold in range(1, max_upside + 1)], 0.0, 1.0)
    )
    probs = np.empty(max_upside + 1, dtype=float)
    probs[0] = 1.0 - values[0]
    if max_upside > 1:
        probs[1:-1] = values[:-1] - values[1:]
    probs[-1] = values[-1]
    return TmaxDistribution(np.arange(observed_bin, observed_bin + max_upside + 1), probs)


def _truncate_below_observed_bin(distribution: TmaxDistribution, observed_bin: int) -> TmaxDistribution:
    probs = distribution.probabilities.copy()
    below = distribution.bins_c < observed_bin
    removed = float(probs[below].sum())
    probs[below] = 0.0
    mask = distribution.bins_c == observed_bin
    if not mask.any():
        bins = np.arange(observed_bin, max(int(distribution.bins_c.max()), observed_bin) + 1)
        new_probs = np.zeros(len(bins), dtype=float)
        lookup = {int(bin_c): float(prob) for bin_c, prob in zip(distribution.bins_c, probs)}
        for idx, bin_c in enumerate(bins):
            new_probs[idx] = lookup.get(int(bin_c), 0.0)
        new_probs[0] += removed
        return TmaxDistribution(bins, new_probs)
    probs[mask] += removed
    return TmaxDistribution(distribution.bins_c, probs)


def _optional_float(row: dict | pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _season(month: int | float | None) -> str:
    if month is None or pd.isna(month):
        return "unknown"
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))
