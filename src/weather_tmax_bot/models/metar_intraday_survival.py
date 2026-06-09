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
    adjusted_values = np.minimum.accumulate(
        np.clip([adjusted_survival[threshold] for threshold in range(1, max_upside + 1)], 0.0, 1.0)
    )
    adjusted_survival = {
        threshold: float(adjusted_values[threshold - 1]) for threshold in range(1, max_upside + 1)
    }
    adjusted = _survival_to_distribution(adjusted_survival, observed_bin, max_upside)
    details = {
        "active": True,
        "layer_version": "lfpb_metar_intraday_survival_v1",
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
