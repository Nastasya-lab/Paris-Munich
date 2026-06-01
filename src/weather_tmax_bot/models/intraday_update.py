from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.intraday_survival_prior import (
    adjust_upside_probability,
    build_daily_first_metar_max,
    build_seasonal_hourly_survival_table,
    lookup_survival_prior,
)
from weather_tmax_bot.models.distribution import TmaxDistribution

LOCAL_TZ = ZoneInfo("Europe/Berlin")
DEFAULT_METAR_SURVIVAL_SOURCE = Path("data/interim/metar_iem_EDDM.parquet")
SHADOW_SURVIVAL_FORMULA = "cap_blend"
SHADOW_SURVIVAL_STRENGTH = 0.75
SHADOW_SURVIVAL_MIN_LOCAL_HOUR = 17.0
NWP_LOCAL_TEMP_COLUMNS = {
    8: "model_temp_at_08_local",
    11: "model_temp_at_11_local",
    14: "model_temp_at_14_local",
    17: "model_temp_at_17_local",
}
WARM_MONTHS = frozenset({5, 6, 7, 8, 9})
SEASONAL_SHADOW_WEIGHT_CURVES = {
    "warm": (
        (0.0, 0.00),
        (8.0, 0.03),
        (10.0, 0.08),
        (12.0, 0.22),
        (14.0, 0.45),
        (15.5, 0.58),
        (17.0, 0.72),
        (19.0, 0.82),
        (23.0, 0.88),
    ),
    "cool": (
        (0.0, 0.00),
        (8.0, 0.03),
        (10.0, 0.10),
        (12.0, 0.28),
        (14.0, 0.52),
        (15.5, 0.66),
        (17.0, 0.78),
        (19.0, 0.84),
        (23.0, 0.88),
    ),
}


@dataclass(frozen=True)
class IntradayUpdateResult:
    distribution: TmaxDistribution
    details: dict


def apply_intraday_update(
    base_distribution: TmaxDistribution,
    feature_row: dict,
    target_date: date,
    issue_time_utc: datetime,
    *,
    training_dataset_path: str | Path = "data/processed/training_dataset.parquet",
    daily_target_path: str | Path = "data/processed/daily_target.parquet",
    training_frame: pd.DataFrame | None = None,
    daily_target_frame: pd.DataFrame | None = None,
    survival_prior_frame: pd.DataFrame | None = None,
    min_rows: int = 40,
    blend_weight_profile: str = "production",
) -> IntradayUpdateResult:
    """Blend a full-day prior with a same-day remaining-upside model.

    The production NWP residual model is intentionally left untouched. This
    layer only activates for same-local-day forecasts with observed METAR
    context, then estimates how much additional Tmax upside remains.
    """
    base_payload = _compact_payload(base_distribution)
    observed_max = _float_or_nan(feature_row.get("observed_max_so_far_from_metar"))
    last_temp = _float_or_nan(feature_row.get("last_metar_temp_c"))
    local_issue = issue_time_utc.astimezone(LOCAL_TZ)
    local_hour = local_issue.hour + local_issue.minute / 60
    details = {
        "active": False,
        "reason": None,
        "base_model": base_payload,
        "local_issue_hour": local_hour,
    }

    if local_issue.date() != target_date:
        details["reason"] = "not_same_local_day"
        return IntradayUpdateResult(base_distribution, _with_final(details, base_distribution))
    if np.isnan(observed_max) or np.isnan(last_temp):
        details["reason"] = "missing_observed_metar_context"
        return IntradayUpdateResult(base_distribution, _with_final(details, base_distribution))

    training = _load_intraday_training(training_dataset_path, training_frame=training_frame)
    if len(training) < min_rows:
        details["reason"] = "insufficient_intraday_training_rows"
        details["training_rows"] = len(training)
        return IntradayUpdateResult(base_distribution, _with_final(details, base_distribution))

    candidates = _select_candidates(training, month=int(feature_row.get("month") or target_date.month), issue_hour=issue_time_utc.hour)
    if len(candidates) < min_rows:
        details["reason"] = "insufficient_matching_intraday_rows"
        details["training_rows"] = len(candidates)
        return IntradayUpdateResult(base_distribution, _with_final(details, base_distribution))

    drop_from_max = max(0.0, observed_max - last_temp)
    nwp_future_max = _nwp_future_max_from_feature_row(feature_row, local_hour)
    nwp_future_upside = None if np.isnan(nwp_future_max) else max(0.0, nwp_future_max - observed_max)
    weights = _candidate_weights(candidates, feature_row, drop_from_max)
    raw_peak_probability = _weighted_mean(candidates["peak_already_passed"].to_numpy(dtype=float), weights)
    timing_prior = _timing_peak_passed_prior(
        daily_target_path=daily_target_path,
        daily_target_frame=daily_target_frame,
        target_date=target_date,
        local_hour=local_hour,
    )
    peak_probability = _contextual_peak_probability(
        raw_peak_probability=raw_peak_probability,
        timing_prior=timing_prior,
        local_hour=local_hour,
        drop_from_max=drop_from_max,
        nwp_future_upside=nwp_future_upside,
        has_precip=bool(feature_row.get("has_precip_recent", False)),
        has_thunder=bool(feature_row.get("has_thunder_recent", False)),
        temp_trend_3h=_float_or_nan(feature_row.get("temp_trend_3h")),
    )
    shadow_survival_table = pd.DataFrame()
    shadow_survival_prior = None
    if blend_weight_profile == "seasonal_shadow":
        shadow_survival_table = survival_prior_frame if survival_prior_frame is not None else _load_default_survival_prior_table()
        if not shadow_survival_table.empty:
            shadow_survival_prior = lookup_survival_prior(shadow_survival_table, month=target_date.month, local_hour=local_hour)

    scenario_metadata = _scenario_tracking_metadata(
        feature_row=feature_row,
        local_hour=local_hour,
        drop_from_max=drop_from_max,
        nwp_future_upside=nwp_future_upside,
    )
    phase_metadata = _classify_intraday_phase(
        target_month=target_date.month,
        local_hour=local_hour,
        drop_from_max=drop_from_max,
        nwp_future_upside=nwp_future_upside,
        survival_prior=shadow_survival_prior,
        scenario_metadata=scenario_metadata,
    )
    intraday_dist = _remaining_upside_distribution(
        observed_max=observed_max,
        increases=candidates["future_increase_c"].to_numpy(dtype=float),
        weights=weights,
        peak_probability=peak_probability,
        local_hour=local_hour,
        drop_from_max=drop_from_max,
        nwp_future_upside=nwp_future_upside,
    )
    blend_weight, blend_metadata = _resolve_intraday_blend_weight(
        profile=blend_weight_profile,
        target_month=target_date.month,
        issue_hour_utc=issue_time_utc.hour,
        local_hour=local_hour,
        peak_probability=peak_probability,
        drop_from_max=drop_from_max,
        survival_prior=shadow_survival_prior,
        nwp_future_upside=nwp_future_upside,
        phase_metadata=phase_metadata,
    )
    final_dist = _blend_distributions(base_distribution, intraday_dist, blend_weight).truncate_below(observed_max)
    survival_metadata = {}
    if blend_weight_profile == "seasonal_shadow":
        final_dist, survival_metadata = _apply_shadow_survival_prior(
            final_dist,
            target_date=target_date,
            local_hour=local_hour,
            observed_max=observed_max,
            survival_prior_frame=shadow_survival_table,
            survival_prior=shadow_survival_prior,
        )

    details.update(
        {
            "active": True,
            "reason": "same_day_intraday_update_applied",
            "training_rows": int(len(candidates)),
            "observed_max_so_far_c": observed_max,
            "last_metar_temp_c": last_temp,
            "drop_from_observed_max_c": drop_from_max,
            "raw_peak_passed_probability": raw_peak_probability,
            "timing_peak_passed_prior": timing_prior,
            "peak_passed_probability": peak_probability,
            "nwp_future_max_sampled_hours_c": None if np.isnan(nwp_future_max) else nwp_future_max,
            "nwp_future_upside_c": nwp_future_upside,
            **scenario_metadata,
            **phase_metadata,
            "intraday_blend_weight": blend_weight,
            **blend_metadata,
            **survival_metadata,
            "intraday_model": _compact_payload(intraday_dist),
        }
    )
    return IntradayUpdateResult(final_dist, _with_final(details, final_dist))


def _load_intraday_training(path: str | Path, *, training_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    if training_frame is None:
        p = Path(path)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_parquet(p)
    else:
        df = training_frame.copy()
    required = {"tmax_c", "observed_max_so_far_from_metar", "last_metar_temp_c", "issue_hour_utc", "month"}
    if missing := required.difference(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["tmax_c"] = pd.to_numeric(out["tmax_c"], errors="coerce")
    out["observed_max_so_far_from_metar"] = pd.to_numeric(out["observed_max_so_far_from_metar"], errors="coerce")
    out["last_metar_temp_c"] = pd.to_numeric(out["last_metar_temp_c"], errors="coerce")
    out = out[out["tmax_c"].notna() & out["observed_max_so_far_from_metar"].notna() & out["last_metar_temp_c"].notna()]
    out["future_increase_c"] = (out["tmax_c"] - out["observed_max_so_far_from_metar"]).clip(lower=0.0)
    out["peak_already_passed"] = out["future_increase_c"] <= 0.5
    out["drop_from_observed_max_c"] = (out["observed_max_so_far_from_metar"] - out["last_metar_temp_c"]).clip(lower=0.0)
    for col in ("temp_trend_3h", "has_precip_recent", "has_thunder_recent"):
        if col not in out.columns:
            out[col] = np.nan if col == "temp_trend_3h" else False
    return out


def _select_candidates(training: pd.DataFrame, *, month: int, issue_hour: int) -> pd.DataFrame:
    month_distance = np.minimum((training["month"].astype(int) - month).abs(), 12 - (training["month"].astype(int) - month).abs())
    hour_distance = (training["issue_hour_utc"].astype(int) - issue_hour).abs()
    selected = training[(month_distance <= 1) & (hour_distance <= 3)].copy()
    if len(selected) >= 40:
        return selected
    selected = training[hour_distance <= 3].copy()
    return selected if len(selected) >= 40 else training.copy()


def _candidate_weights(candidates: pd.DataFrame, feature_row: dict, drop_from_max: float) -> np.ndarray:
    weights = np.ones(len(candidates), dtype=float)
    candidate_drop = pd.to_numeric(candidates["drop_from_observed_max_c"], errors="coerce").fillna(0).to_numpy(dtype=float)
    weights *= np.exp(-np.abs(candidate_drop - drop_from_max) / 4.0)

    current_trend = _float_or_nan(feature_row.get("temp_trend_3h"))
    if not np.isnan(current_trend):
        candidate_trend = pd.to_numeric(candidates.get("temp_trend_3h"), errors="coerce").fillna(0).to_numpy(dtype=float)
        weights *= np.exp(-np.abs(candidate_trend - current_trend) / 5.0)

    current_max = _float_or_nan(feature_row.get("observed_max_so_far_from_metar"))
    if not np.isnan(current_max):
        candidate_max = pd.to_numeric(candidates["observed_max_so_far_from_metar"], errors="coerce").fillna(current_max).to_numpy(dtype=float)
        weights *= np.exp(-np.abs(candidate_max - current_max) / 8.0)

    for flag in ("has_precip_recent", "has_thunder_recent"):
        if bool(feature_row.get(flag, False)) and flag in candidates.columns:
            weights *= np.where(candidates[flag].astype(bool).to_numpy(), 1.5, 0.85)
    return np.clip(weights, 1e-6, None)


def _timing_peak_passed_prior(
    *,
    daily_target_path: str | Path,
    target_date: date,
    local_hour: float,
    daily_target_frame: pd.DataFrame | None = None,
) -> float | None:
    if daily_target_frame is None:
        path = Path(daily_target_path)
        if not path.exists():
            return None
        target = pd.read_parquet(path)
    else:
        target = daily_target_frame.copy()
    if "tmax_time_local" not in target.columns or "target_date_local" not in target.columns:
        return None
    df = target.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce")
    df = df[df["target_date_local"].dt.month == target_date.month]
    if df.empty:
        return None
    # Keep the local wall-clock hour across DST transitions. Converting the
    # mixed +01:00/+02:00 offsets to UTC would answer a different question.
    tmax_hour = df["tmax_time_local"].map(_local_wall_clock_hour)
    valid = tmax_hour.notna()
    if valid.sum() < 20:
        return None
    return float((tmax_hour[valid] <= local_hour).mean())


def _contextual_peak_probability(
    *,
    raw_peak_probability: float,
    timing_prior: float | None,
    local_hour: float,
    drop_from_max: float,
    nwp_future_upside: float | None,
    has_precip: bool,
    has_thunder: bool,
    temp_trend_3h: float,
) -> float:
    probability = raw_peak_probability if timing_prior is None else 0.65 * raw_peak_probability + 0.35 * timing_prior
    if local_hour < 8:
        probability = min(probability, 0.25)
    elif local_hour < 10:
        probability = min(probability, 0.45)
    elif local_hour < 12 and nwp_future_upside is not None and nwp_future_upside >= 2.0:
        probability *= 0.75

    sharp_drop = drop_from_max >= 5.0
    weather_break = has_precip or has_thunder or (not np.isnan(temp_trend_3h) and temp_trend_3h <= -3.0)
    if local_hour >= 14 and sharp_drop and weather_break:
        probability = max(probability, 0.85)
    if local_hour >= 16 and sharp_drop:
        probability = max(probability, 0.92)
    if nwp_future_upside is not None and nwp_future_upside <= 0.5 and local_hour >= 14:
        probability = max(probability, 0.80)
    if nwp_future_upside is not None and nwp_future_upside >= 4.0 and local_hour < 13:
        probability *= 0.7
    return float(np.clip(probability, 0.02, 0.99))


def _remaining_upside_distribution(
    *,
    observed_max: float,
    increases: np.ndarray,
    weights: np.ndarray,
    peak_probability: float,
    local_hour: float,
    drop_from_max: float,
    nwp_future_upside: float | None,
) -> TmaxDistribution:
    adjusted = np.asarray(increases, dtype=float).copy()
    shrink = np.clip(0.15 + 0.85 * (1.0 - peak_probability), 0.12, 1.0)
    if local_hour >= 13 or drop_from_max >= 4:
        adjusted *= shrink
    if nwp_future_upside is not None:
        allowance = 0.75 + 1.5 * (1.0 - peak_probability)
        cap = max(0.5, nwp_future_upside + allowance)
        adjusted = np.minimum(adjusted, cap)
    values = observed_max + np.clip(adjusted, 0.0, None)
    return _weighted_empirical_distribution(values, weights)


def _intraday_blend_weight(*, local_hour: float, peak_probability: float) -> float:
    if local_hour < 8:
        base = 0.20
    elif local_hour < 11:
        base = 0.35
    elif local_hour < 14:
        base = 0.55
    elif local_hour < 16:
        base = 0.75
    else:
        base = 0.90
    if local_hour < 10:
        return float(min(0.45, max(base, 0.35 * peak_probability)))
    return float(np.clip(max(base, 0.9 * peak_probability), 0.20, 0.95))


def _resolve_intraday_blend_weight(
    *,
    profile: str,
    target_month: int,
    issue_hour_utc: int,
    local_hour: float,
    peak_probability: float,
    drop_from_max: float,
    survival_prior: float | None = None,
    nwp_future_upside: float | None = None,
    phase_metadata: dict | None = None,
) -> tuple[float, dict]:
    if profile == "production":
        weight = _intraday_blend_weight(local_hour=local_hour, peak_probability=peak_probability)
        return weight, {
            "blend_weight_profile": "production_dynamic_v1",
            "shadow_mode": False,
        }
    if profile != "seasonal_shadow":
        raise ValueError(f"unknown intraday blend weight profile: {profile}")

    season = "warm" if target_month in WARM_MONTHS else "cool"
    base_weight = _hour_curve_weight(season=season, local_hour=local_hour)
    late_gate = float(np.clip((local_hour - 12.0) / 5.0, 0.0, 1.0))
    peak_component = late_gate * 0.85 * peak_probability
    survival_component = None
    if survival_prior is not None:
        survival_component = late_gate * (0.25 + 0.65 * (1.0 - float(np.clip(survival_prior, 0.0, 1.0))))
    nwp_future_component = 0.0
    if nwp_future_upside is not None and local_hour < 13.0 and nwp_future_upside >= 3.0:
        # Morning and late-morning showers should not make the shadow model
        # over-trust observed maxima when NWP still has substantial heating.
        nwp_future_component = -0.18
    weight_candidates = [base_weight, peak_component]
    if survival_component is not None:
        weight_candidates.append(survival_component)
    weight = max(weight_candidates) + nwp_future_component
    override_weight = 0.95 if season == "warm" else 0.85
    late_drop_override = local_hour >= 14.0 and drop_from_max >= 5.0
    if late_drop_override:
        weight = max(weight, override_weight)
    if not late_drop_override:
        weight = float(np.clip(weight, 0.0, 0.92))
    phase = (phase_metadata or {}).get("forecast_phase")
    if phase == "morning_prior":
        morning_cap = 0.08 if nwp_future_upside is not None and nwp_future_upside >= 3.0 else 0.20
        weight = min(weight, morning_cap)
    elif phase == "midday_update":
        weight = min(weight, 0.70)
    elif phase == "late_nowcast":
        weight = max(weight, 0.65)
    return float(weight), {
        "blend_weight_profile": "seasonal_hour_aware_challenger_v2",
        "shadow_mode": True,
        "seasonal_profile": season,
        "seasonal_weight_group": "local_hour_curve",
        "seasonal_base_weight": float(base_weight),
        "seasonal_survival_prior_for_weight": None if survival_prior is None else float(survival_prior),
        "seasonal_survival_weight_component": None if survival_component is None else float(survival_component),
        "peak_probability_weight_component": float(peak_component),
        "nwp_future_weight_adjustment": float(nwp_future_component),
        "late_drop_override_active": bool(late_drop_override),
        "late_drop_override_weight": float(override_weight) if late_drop_override else None,
    }


def _scenario_tracking_metadata(
    *,
    feature_row: dict,
    local_hour: float,
    drop_from_max: float,
    nwp_future_upside: float | None,
) -> dict:
    temp_trend_3h = _float_or_nan(feature_row.get("temp_trend_3h"))
    metar_weather_break = (
        bool(feature_row.get("has_precip_recent", False))
        or bool(feature_row.get("has_thunder_recent", False))
        or drop_from_max >= 3.0
        or (not np.isnan(temp_trend_3h) and temp_trend_3h <= -2.5)
    )
    taf_adverse_weather = any(
        bool(feature_row.get(flag, False))
        for flag in (
            "taf_has_rain",
            "taf_has_shower",
            "taf_has_thunder",
            "taf_has_fog",
            "taf_has_snow",
            "taf_prob30_bad_weather",
            "taf_prob40_bad_weather",
        )
    )
    nwp_future_precip = _float_or_nan(feature_row.get("model_future_precip_sum"))
    nwp_future_cloud = _float_or_nan(feature_row.get("model_future_cloud_cover_mean"))
    nwp_future_wind = _float_or_nan(feature_row.get("model_future_wind_speed_max"))
    nwp_future_gust = _float_or_nan(feature_row.get("model_future_gust_max"))
    nwp_adverse_components = []
    if not np.isnan(nwp_future_precip) and nwp_future_precip >= 0.5:
        nwp_adverse_components.append("future_precip")
    if not np.isnan(nwp_future_cloud) and nwp_future_cloud >= 85.0:
        nwp_adverse_components.append("future_cloud")
    if not np.isnan(nwp_future_wind) and nwp_future_wind >= 35.0:
        nwp_adverse_components.append("future_wind")
    if not np.isnan(nwp_future_gust) and nwp_future_gust >= 50.0:
        nwp_adverse_components.append("future_gust")
    nwp_adverse_weather = bool(nwp_adverse_components)
    if nwp_future_upside is None:
        nwp_future_heating_signal = "unknown"
    elif nwp_future_upside >= 3.0:
        nwp_future_heating_signal = "strong_future_heating"
    elif nwp_future_upside >= 1.0:
        nwp_future_heating_signal = "some_future_heating"
    else:
        nwp_future_heating_signal = "little_future_heating"

    if local_hour < 13.0 and metar_weather_break and nwp_future_upside is not None and nwp_future_upside >= 3.0:
        scenario = "temporary_disruption_possible"
    elif local_hour >= 14.0 and metar_weather_break and (nwp_future_upside is None or nwp_future_upside <= 1.0):
        scenario = "heating_cutoff_likely"
    elif metar_weather_break and taf_adverse_weather and nwp_adverse_weather:
        scenario = "multi_source_adverse_weather"
    elif taf_adverse_weather and metar_weather_break:
        scenario = "taf_and_metar_adverse"
    elif nwp_future_upside is not None and nwp_future_upside >= 2.0:
        scenario = "nwp_still_supports_higher_tmax"
    else:
        scenario = "near_observed_track"

    return {
        "scenario_tracking": scenario,
        "metar_weather_break_signal": bool(metar_weather_break),
        "taf_adverse_weather_signal": bool(taf_adverse_weather),
        "nwp_adverse_weather_signal": bool(nwp_adverse_weather),
        "nwp_adverse_weather_components": nwp_adverse_components,
        "nwp_future_heating_signal": nwp_future_heating_signal,
        "temp_trend_3h_for_phase_c": None if np.isnan(temp_trend_3h) else float(temp_trend_3h),
    }


def _classify_intraday_phase(
    *,
    target_month: int,
    local_hour: float,
    drop_from_max: float,
    nwp_future_upside: float | None,
    survival_prior: float | None,
    scenario_metadata: dict,
) -> dict:
    season = "warm" if target_month in WARM_MONTHS else "cool"
    metar_break = bool(scenario_metadata.get("metar_weather_break_signal"))
    nwp_adverse_weather = bool(scenario_metadata.get("nwp_adverse_weather_signal"))
    reasons = []

    if local_hour < 11.0:
        phase = "morning_prior"
        reasons.append("local_hour_before_11")
        if nwp_future_upside is not None and nwp_future_upside >= 3.0:
            reasons.append("nwp_future_heating_available")
    else:
        late_cutoff_signal = (
            local_hour >= 14.0
            and metar_break
            and drop_from_max >= 3.0
            and (nwp_future_upside is None or nwp_future_upside <= 1.0 or nwp_adverse_weather)
        )
        survival_late_signal = (
            local_hour >= 15.5
            and survival_prior is not None
            and survival_prior <= 0.12
            and (nwp_future_upside is None or nwp_future_upside <= 1.5)
        )
        if local_hour >= 16.0 or late_cutoff_signal or survival_late_signal:
            phase = "late_nowcast"
            if local_hour >= 16.0:
                reasons.append("local_hour_ge_16")
            if late_cutoff_signal:
                reasons.append("metar_break_with_little_nwp_upside")
            if survival_late_signal:
                reasons.append("low_historical_peak_survival")
        else:
            phase = "midday_update"
            reasons.append("main_heating_window")
            if nwp_future_upside is not None and nwp_future_upside >= 1.0:
                reasons.append("nwp_still_allows_upside")

    return {
        "forecast_phase": phase,
        "phase_reason": ",".join(reasons),
        "phase_season": season,
    }


def _hour_curve_weight(*, season: str, local_hour: float) -> float:
    curve = SEASONAL_SHADOW_WEIGHT_CURVES[season]
    hours = np.array([point[0] for point in curve], dtype=float)
    weights = np.array([point[1] for point in curve], dtype=float)
    return float(np.interp(local_hour, hours, weights))


def _apply_shadow_survival_prior(
    distribution: TmaxDistribution,
    *,
    target_date: date,
    local_hour: float,
    observed_max: float,
    survival_prior_frame: pd.DataFrame | None = None,
    survival_prior: float | None = None,
) -> tuple[TmaxDistribution, dict]:
    metadata = {
        "survival_adjustment_active": False,
        "survival_adjustment_formula": SHADOW_SURVIVAL_FORMULA,
        "survival_adjustment_strength": SHADOW_SURVIVAL_STRENGTH,
        "survival_adjustment_min_local_hour": SHADOW_SURVIVAL_MIN_LOCAL_HOUR,
    }
    if local_hour < SHADOW_SURVIVAL_MIN_LOCAL_HOUR:
        metadata["survival_adjustment_reason"] = "before_min_local_hour"
        return distribution, metadata
    prior = survival_prior
    if prior is None:
        table = survival_prior_frame if survival_prior_frame is not None else _load_default_survival_prior_table()
        if table.empty:
            metadata["survival_adjustment_reason"] = "survival_prior_unavailable"
            return distribution, metadata
        prior = lookup_survival_prior(table, month=target_date.month, local_hour=local_hour)
    adjustment = adjust_upside_probability(
        distribution,
        observed_max_so_far_c=observed_max,
        survival_prior=prior,
        formula=SHADOW_SURVIVAL_FORMULA,
        strength=SHADOW_SURVIVAL_STRENGTH,
    )
    metadata.update(
        {
            "survival_adjustment_active": True,
            "survival_adjustment_reason": "local_ge17_cap_blend_applied",
            "seasonal_survival_prior": float(prior),
            "survival_original_upside_probability": adjustment.original_upside_probability,
            "survival_adjusted_upside_probability": adjustment.adjusted_upside_probability,
            "survival_observed_max_bin_c": adjustment.observed_max_bin_c,
        }
    )
    return adjustment.distribution, metadata


@lru_cache(maxsize=1)
def _load_default_survival_prior_table() -> pd.DataFrame:
    if not DEFAULT_METAR_SURVIVAL_SOURCE.exists():
        return pd.DataFrame()
    metar = pd.read_parquet(DEFAULT_METAR_SURVIVAL_SOURCE)
    daily = build_daily_first_metar_max(metar)
    if daily.empty:
        return pd.DataFrame()
    return build_seasonal_hourly_survival_table(daily, train_before=date(2026, 1, 1))


def _nwp_future_max_from_feature_row(feature_row: dict, local_hour: float) -> float:
    future_max = _float_or_nan(feature_row.get("model_future_temp_max_c"))
    if not np.isnan(future_max):
        return float(future_max)
    values = []
    for hour, column in NWP_LOCAL_TEMP_COLUMNS.items():
        value = _float_or_nan(feature_row.get(column))
        if hour > local_hour and not np.isnan(value):
            values.append(value)
    return float(max(values)) if values else float("nan")


def _blend_distributions(base: TmaxDistribution, intraday: TmaxDistribution, intraday_weight: float) -> TmaxDistribution:
    bins = np.arange(min(base.bins_c.min(), intraday.bins_c.min()), max(base.bins_c.max(), intraday.bins_c.max()) + 1)
    base_probs = _align_probs(base, bins)
    intraday_probs = _align_probs(intraday, bins)
    probs = (1.0 - intraday_weight) * base_probs + intraday_weight * intraday_probs
    return TmaxDistribution(bins, probs)


def _weighted_empirical_distribution(samples: np.ndarray, weights: np.ndarray, bin_min: int = -35, bin_max: int = 45) -> TmaxDistribution:
    bins = np.arange(bin_min, bin_max + 1)
    rounded = np.rint(np.asarray(samples, dtype=float)).astype(int)
    weights = np.asarray(weights, dtype=float)
    probs = np.array([weights[rounded == bin_c].sum() for bin_c in bins], dtype=float)
    if probs.sum() <= 0:
        probs[np.argmin(np.abs(bins - np.nanmedian(samples)))] = 1.0
    return TmaxDistribution(bins, probs)


def _align_probs(dist: TmaxDistribution, bins: np.ndarray) -> np.ndarray:
    out = np.zeros(len(bins), dtype=float)
    lookup = {int(bin_c): idx for idx, bin_c in enumerate(bins)}
    for bin_c, probability in zip(dist.bins_c, dist.probabilities):
        out[lookup[int(bin_c)]] = probability
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = weights.sum()
    if total <= 0:
        return float(np.nanmean(values))
    return float(np.sum(values * weights) / total)


def _compact_payload(dist: TmaxDistribution) -> dict:
    payload = dist.to_payload()
    material = {
        key: value
        for key, value in payload["probabilities_by_integer_c"].items()
        if float(value) >= 0.005
    }
    payload["probabilities_by_integer_c"] = material
    return payload


def _with_final(details: dict, dist: TmaxDistribution) -> dict:
    details["final_model"] = _compact_payload(dist)
    return details


def _float_or_nan(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _local_wall_clock_hour(value) -> float:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return float("nan")
    if pd.isna(timestamp):
        return float("nan")
    return float(timestamp.hour + timestamp.minute / 60)
