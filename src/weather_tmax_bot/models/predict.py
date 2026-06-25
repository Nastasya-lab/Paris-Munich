from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from weather_tmax_bot.features.build_features import build_feature_row
from weather_tmax_bot.features.metar_upside_dataset import build_current_metar_upside_features
from weather_tmax_bot.features.nwp_features import build_nwp_features
from weather_tmax_bot.features.spatial_metar import SPATIAL_STATIONS_BY_AIRPORT, build_spatial_metar_features
from weather_tmax_bot.features.wind_advection import EDDM_ADVECTION_STATIONS, build_wind_advection_features
from weather_tmax_bot.models.baselines import ClimatologyBaseline
from weather_tmax_bot.models.disagreement import assess_model_disagreement
from weather_tmax_bot.models.distribution import project_unimodal_distribution, unimodal_violation_count
from weather_tmax_bot.models.extrapolation import detect_feature_extrapolation
from weather_tmax_bot.models.intraday_update import apply_intraday_update
from weather_tmax_bot.models.model_registry import load_model, resolve_active_artifacts
from weather_tmax_bot.models.phase_arbitration import build_phase_arbitrated_candidate
from weather_tmax_bot.models.safe_blend import build_blended_shadow_candidate
from weather_tmax_bot.temporal.freshness import assess_feature_freshness
from weather_tmax_bot.temporal.source_compatibility import assess_source_compatibility
from weather_tmax_bot.utils.time import local_day_bounds_utc

EDDM_WIND_ADVECTION_MODEL_PATH = Path("data/models/eddm_metar_tmax_icon_d2_spatial_wind_advection_v1.joblib")
EDDM_WIND_ADVECTION_METADATA_PATH = Path("data/models/eddm_metar_tmax_icon_d2_spatial_wind_advection_v1.metadata.json")
EDDM_WIND_ADVECTION_VARIANT = "shadow_spatial_wind_advection"
EDDM_WIND_ADVECTION_LOCAL_HOUR_START = 12
EDDM_WIND_ADVECTION_LOCAL_HOUR_END = 18
EDDM_UNIMODAL_VARIANT = "shadow_unimodal_pmf"
EDDM_UNIMODAL_VERSION = "eddm_unimodal_projection_shadow_v1"


def predict_with_climatology(
    target_date: date,
    daily_target_path: str | Path = "data/processed/daily_target.parquet",
    observed_max_so_far: float | None = None,
):
    if Path(daily_target_path).exists():
        targets = pd.read_parquet(daily_target_path)
    else:
        targets = _synthetic_targets()
    return ClimatologyBaseline().fit(targets).predict_distribution(target_date, observed_max_so_far)


def predict_best_available(
    airport: str,
    target_date: date,
    issue_time_utc: datetime,
    daily_target_path: str | Path = "data/processed/daily_target.parquet",
    model_path: str | Path = "data/models/quantile_mvp.joblib",
) -> tuple[object, dict]:
    warnings = []
    active = resolve_active_artifacts(fallback_model_path=model_path)
    resolved_model_path = active["model_path"]
    if resolved_model_path is not None and Path(resolved_model_path).exists():
        metar = _load_metar_for_issue(airport, target_date, issue_time_utc)
        spatial_metars = _load_spatial_metars_for_issue(airport, target_date, issue_time_utc)
        taf = _load_taf_for_issue(airport, issue_time_utc)
        nwp = _load_nwp_for_issue(target_date, issue_time_utc)
        feature_row = build_feature_row(
            airport_icao=airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date,
            metar=metar,
            spatial_metars=spatial_metars,
            taf=taf,
            nwp=nwp,
        )
        model = load_model(resolved_model_path)
        runtime_model_version = active.get("active_model_version") or Path(resolved_model_path).stem
        if _is_metar_tmax_model(model):
            try:
                return _predict_metar_tmax_model(
                    model=model,
                    active_model_version=runtime_model_version,
                    airport=airport,
                    target_date=target_date,
                    issue_time_utc=issue_time_utc,
                    metar=metar,
                    spatial_metars=spatial_metars,
                    nwp=nwp,
                )
            except (FileNotFoundError, ValueError, KeyError) as exc:
                warnings.append(f"Active METAR Tmax model could not use current features: {exc}; legacy Munich model fallback used.")
                legacy_path = Path("data/models/nwp_residual_icon_d2_20260531.joblib")
                if legacy_path.exists():
                    model = load_model(legacy_path)
                    runtime_model_version = legacy_path.stem
                else:
                    warnings.append("Legacy Munich model unavailable; climatology fallback used.")
                    dist = predict_with_climatology(target_date, daily_target_path=daily_target_path)
                    payload = dist.to_payload()
                    return dist, {
                        "model_version": "climatology_mvp",
                        "feature_snapshot": {
                            "airport_icao": airport,
                            "target_date_local": target_date.isoformat(),
                            "metar_missing": metar.empty,
                            "nwp_missing": nwp.empty,
                            "forecast_variants": {
                                "production_champion": {
                                    "description": "Climatology fallback distribution.",
                                    "distribution": payload,
                                    "metadata": {"variant_version": "climatology_mvp"},
                                }
                            },
                        },
                        "warnings": warnings,
                    }
        observed_max = feature_row.get("observed_max_so_far_from_metar")
        try:
            dist = model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed_max)
        except ValueError as exc:
            warnings.append(f"Active model could not use current features: {exc}; climatology fallback used.")
            dist = predict_with_climatology(target_date, daily_target_path=daily_target_path, observed_max_so_far=observed_max)
        extrapolation = detect_feature_extrapolation(feature_row, model)
        feature_row["extrapolation"] = extrapolation
        warnings.extend(extrapolation["warnings"])
        calibrator_path = active["calibrator_path"]
        if calibrator_path is not None and calibrator_path.exists():
            calibrator = load_model(calibrator_path)
            dist = calibrator.transform(dist).truncate_below(observed_max)
            warnings.append("Validation-fitted spread calibration applied.")
        base_dist = dist
        intraday = apply_intraday_update(base_dist, feature_row, target_date, issue_time_utc)
        shadow_intraday = apply_intraday_update(
            base_dist,
            feature_row,
            target_date,
            issue_time_utc,
            blend_weight_profile="seasonal_shadow",
        )
        ml_shadow_dist, ml_shadow_details = _predict_intraday_ml_shadow(feature_row)
        dist = intraday.distribution
        freshness = assess_feature_freshness(feature_row, issue_time_utc)
        feature_row["freshness"] = freshness["statuses"]
        warnings.extend(freshness["warnings"])
        source_compatibility = assess_source_compatibility(feature_row)
        feature_row["source_compatibility"] = source_compatibility["sources"]
        warnings.extend(source_compatibility["warnings"])
        component_variants = {
            "production_champion": {
                "description": "Operational intraday distribution before optional late-day promotion.",
                "distribution": dist.to_payload(),
                "metadata": {
                    "variant_version": "production_dynamic_v1",
                    **intraday.details,
                },
            },
            "base_prior": {
                "description": "Full-day prior before same-day intraday update.",
                "distribution": base_dist.to_payload(),
            },
            "shadow_seasonal_intraday": {
                "description": "Phase-aware intraday shadow challenger; never used as the operational forecast.",
                "distribution": shadow_intraday.distribution.to_payload(),
                "metadata": {
                    "variant_version": "phase_aware_intraday_challenger_v3",
                    **shadow_intraday.details,
                },
            },
        }
        if ml_shadow_dist is not None:
            ml_calibrated = ml_shadow_details.get("calibration_status") == "contextual_out_of_fold_survival_calibrated"
            component_variants["shadow_intraday_ml"] = {
                "description": (
                    "Preliminary calibrated ordinal remaining-upside ML shadow challenger."
                    if ml_calibrated
                    else "Preliminary raw ordinal remaining-upside ML shadow challenger; calibration gate did not accept the latest candidate."
                ),
                "distribution": ml_shadow_dist.to_payload(),
                "metadata": {
                    "variant_version": "intraday_ml_core_challenger_v1",
                    **ml_shadow_details,
                },
            }
        pre_promotion_disagreement = assess_model_disagreement(component_variants)
        blended_shadow = build_blended_shadow_candidate(
            dist,
            shadow_intraday.distribution,
            phase_details=shadow_intraday.details,
            ml_shadow_details=ml_shadow_details,
            model_disagreement=pre_promotion_disagreement,
            source_compatibility=feature_row["source_compatibility"],
            freshness=feature_row["freshness"],
        )
        component_variants["shadow_safe_blend"] = {
            "description": "Conservative smooth blended shadow candidate; never used as the operational forecast.",
            "distribution": blended_shadow.distribution.to_payload(),
            "metadata": blended_shadow.details,
        }
        phase_arbitrated = build_phase_arbitrated_candidate(
            champion=dist,
            safe_blend=blended_shadow.distribution,
            seasonal_shadow=shadow_intraday.distribution,
            ml_shadow=ml_shadow_dist,
            local_hour=float(intraday.details.get("local_issue_hour") or 0.0),
        )
        late_day_promotion = _late_day_promotion_payload(phase_arbitrated.details, dist)
        if phase_arbitrated.details.get("selected_variant") == "shadow_intraday_ml":
            dist = phase_arbitrated.distribution
            late_day_promotion.update(
                {
                    "active": True,
                    "status": "promoted_to_production",
                    "promoted_expected_tmax_c": dist.expected_tmax_c,
                    "promoted_distribution": dist.to_payload(),
                }
            )
            warnings.append("Late-day ML remaining-upside component promoted into production forecast.")
        component_variants["production_champion"] = {
            "description": "Operational distribution returned to users.",
            "distribution": dist.to_payload(),
            "metadata": {
                "variant_version": (
                    "production_dynamic_late_day_ml_v2" if late_day_promotion.get("active") else "production_dynamic_v1"
                ),
                **intraday.details,
                "late_day_promotion": late_day_promotion,
            },
        }
        feature_row["forecast_variants"] = {
            "production_champion": component_variants["production_champion"],
        }
        model_disagreement = assess_model_disagreement(component_variants)
        feature_row["model_disagreement"] = model_disagreement
        feature_row["component_variants"] = component_variants
        feature_row["growth_potential"] = _growth_potential_payload(ml_shadow_details)
        feature_row["intraday_update"] = intraday.details
        feature_row["shadow_intraday_update"] = shadow_intraday.details
        feature_row["forecast_components"] = {
            "base_model": intraday.details.get("base_model"),
            "intraday_update": {
                key: value
                for key, value in intraday.details.items()
                if key not in {"base_model", "intraday_model", "final_model"}
            },
            "intraday_model": intraday.details.get("intraday_model"),
            "final_model": intraday.details.get("final_model"),
            "late_day_promotion": late_day_promotion,
            "shadow_mode": {
                "name": "phase_aware_intraday_challenger_v3",
                "status": "shadow_only_does_not_affect_operational_forecast",
                "intraday_update": {
                    key: value
                    for key, value in shadow_intraday.details.items()
                    if key not in {"base_model", "intraday_model", "final_model"}
                },
                "final_model": shadow_intraday.details.get("final_model"),
                "comparison_to_champion": _distribution_comparison(shadow_intraday.distribution, dist),
            },
            "ml_shadow_mode": {
                "name": "intraday_ml_core_challenger_v1",
                "status": (
                    "shadow_only_preliminary_calibrated_does_not_affect_operational_forecast"
                    if ml_shadow_details.get("calibration_status") == "contextual_out_of_fold_survival_calibrated"
                    else "shadow_only_preliminary_raw_calibration_candidate_rejected_does_not_affect_operational_forecast"
                ),
                "details": ml_shadow_details,
                "final_model": None if ml_shadow_dist is None else ml_shadow_dist.to_payload(),
                "comparison_to_champion": None if ml_shadow_dist is None else _distribution_comparison(ml_shadow_dist, dist),
            },
            "model_disagreement": model_disagreement,
            "blended_shadow_mode": {
                "name": "blended_shadow_candidate_v1",
                "status": "shadow_only_does_not_affect_operational_forecast",
                "details": blended_shadow.details,
                "final_model": blended_shadow.distribution.to_payload(),
                "comparison_to_champion": _distribution_comparison(blended_shadow.distribution, dist),
            },
        }
        if intraday.details.get("active"):
            warnings.append("Intraday update applied: base prior blended with current METAR/TAF/NWP remaining-day signal.")
        if model.__class__.__name__ == "QuantileTmaxModel":
            warnings.append("Quantile MVP model used; calibration layer is still preliminary.")
        else:
            warnings.append(f"{model.__class__.__name__} model used; monitor NWP source availability and residual calibration.")
        if feature_row.get("nwp_missing", True):
            warnings.append("NWP forecast-as-issued archive not yet available.")
        if feature_row.get("taf_missing", True):
            warnings.append("TAF missing for this as-of feature view.")
        return dist, {
            "model_version": runtime_model_version,
            "feature_snapshot": feature_row,
            "warnings": warnings,
        }
    dist = predict_with_climatology(target_date, daily_target_path=daily_target_path)
    warnings.append("Quantile model unavailable; climatology MVP distribution used.")
    return dist, {"model_version": "climatology_mvp", "feature_snapshot": {}, "warnings": warnings}


def _is_metar_tmax_model(model) -> bool:
    return model.__class__.__name__ in {"IconD2MetarTmaxEnsemble", "MetarTmaxUpsideModel"}


def _predict_metar_tmax_model(
    *,
    model,
    active_model_version: str,
    airport: str,
    target_date: date,
    issue_time_utc: datetime,
    metar: pd.DataFrame,
    spatial_metars: dict[str, pd.DataFrame],
    nwp: pd.DataFrame,
) -> tuple[object, dict]:
    timezone_name = "Europe/Berlin" if airport.upper() == "EDDM" else "Europe/Paris"
    feature_row = build_current_metar_upside_features(
        metar,
        airport_icao=airport,
        target_date_local=target_date,
        issue_time_utc=issue_time_utc,
        timezone_name=timezone_name,
    )
    _annotate_metar_source(feature_row, metar)
    feature_row["taf_not_required"] = True
    feature_row["taf_missing"] = False
    feature_row["issue_schedule_offset_minutes"] = _local_issue_schedule_offset_minutes(
        issue_time_utc,
        timezone_name,
        scheduled_hours=[6, 8, 10, 12, 14, 16, 18, 20],
    )
    nwp_features = build_nwp_features(nwp, issue_time_utc)
    _add_metar_tmax_nwp_relative_features(nwp_features, feature_row)
    feature_row.update(nwp_features)
    feature_row["max_feature_knowledge_time_utc"] = _max_timestamp_string(
        feature_row.get("max_feature_knowledge_time_utc"),
        feature_row.get("max_nwp_knowledge_time_utc"),
    )
    if hasattr(model, "residuals_by_hour") and pd.isna(feature_row.get("model_tmax_c")):
        raise FileNotFoundError("ICON-D2 features are required for the active METAR Tmax ensemble")
    stations = list(SPATIAL_STATIONS_BY_AIRPORT.get(airport.upper(), ()))
    if stations:
        spatial_features = build_spatial_metar_features(
            feature_row,
            spatial_metars,
            target_date_local=target_date,
            issue_time_utc=issue_time_utc,
            timezone_name=timezone_name,
            stations=stations,
        )
        feature_row.update(spatial_features)
        feature_row["max_feature_knowledge_time_utc"] = _max_timestamp_string(
            feature_row.get("max_feature_knowledge_time_utc"),
            spatial_features.get("spatial_max_feature_knowledge_time_utc"),
        )
        feature_row["leakage_check_passed"] = bool(feature_row.get("leakage_check_passed", False)) and bool(
            spatial_features.get("spatial_leakage_check_passed", False)
        )
    dist = model.predict_distribution(feature_row)
    unimodal_candidate = _build_unimodal_shadow_candidate(
        champion=dist,
        active_model_version=active_model_version,
        feature_row=feature_row,
    )
    wind_advection_candidate = _predict_eddm_wind_advection_candidate(
        airport=airport,
        target_date=target_date,
        issue_time_utc=issue_time_utc,
        base_feature_row=feature_row,
        metar=metar,
        spatial_metars=spatial_metars,
        champion=dist,
    )
    extrapolation = detect_feature_extrapolation(feature_row, model)
    feature_row["extrapolation"] = extrapolation
    freshness = assess_feature_freshness(feature_row, issue_time_utc)
    feature_row["freshness"] = freshness["statuses"]
    source_compatibility = assess_source_compatibility(feature_row)
    feature_row["source_compatibility"] = source_compatibility["sources"]
    feature_row["target"] = "METAR_Tmax"
    feature_row["target_description"] = "daily maximum temperature reported by EDDM METAR"
    feature_row["forecast_variants"] = {
        "production_champion": {
            "description": "Operational EDDM METAR Tmax distribution.",
            "distribution": dist.to_payload(),
            "metadata": {
                "variant_version": active_model_version,
                "local_issue_hour": feature_row.get("local_issue_hour"),
                "target": "METAR_Tmax",
            },
        }
    }
    feature_row["forecast_variants"][EDDM_UNIMODAL_VARIANT] = {
        "description": "Shadow-only EDDM least-squares unimodal PMF projection; does not affect operational forecast.",
        "distribution": unimodal_candidate["forecast"],
        "metadata": unimodal_candidate["metadata"],
    }
    if wind_advection_candidate.get("active") and wind_advection_candidate.get("forecast"):
        feature_row["forecast_variants"][EDDM_WIND_ADVECTION_VARIANT] = {
            "description": "Shadow-only EDDM spatial + wind/advection candidate; does not affect operational forecast.",
            "distribution": wind_advection_candidate["forecast"],
            "metadata": {
                "variant_version": wind_advection_candidate.get("model_version"),
                "local_issue_hour": wind_advection_candidate.get("local_issue_hour"),
                "target": "METAR_Tmax",
                "active_local_hour_window": wind_advection_candidate.get("active_local_hour_window"),
            },
        }
    feature_row["forecast_components"] = {
        "base_model": {
            "model_version": active_model_version,
            "expected_tmax_c": dist.expected_tmax_c,
            "target": "METAR_Tmax",
        },
        "metar_tmax_model": {
            "active": True,
            "model_version": active_model_version,
            "local_issue_hour": feature_row.get("local_issue_hour"),
            "current_metar_max_c": feature_row.get("current_metar_max_c"),
            "latest_metar_temp_c": feature_row.get("latest_metar_temp_c"),
            "model_tmax_c": feature_row.get("model_tmax_c"),
            "spatial_available_station_count": feature_row.get("spatial_available_station_count"),
        },
        "spatial_wind_advection_candidate": wind_advection_candidate,
        "unimodal_shadow_candidate": unimodal_candidate,
    }
    warnings = [
        *extrapolation.get("warnings", []),
        *freshness.get("warnings", []),
        *source_compatibility.get("warnings", []),
        "EDDM METAR Tmax model used; target is METAR Tmax, not DWD official Tmax.",
    ]
    if feature_row.get("nwp_missing", True):
        warnings.append("NWP forecast-as-issued archive not yet available.")
    return dist, {
        "model_version": active_model_version,
        "feature_snapshot": feature_row,
        "warnings": warnings,
    }


def _add_metar_tmax_nwp_relative_features(nwp_features: dict, metar_features: dict) -> None:
    model_tmax = nwp_features.get("model_tmax_c")
    current_max = metar_features.get("current_metar_max_c")
    future = nwp_features.get("model_future_temp_max_c")
    nwp_features["nwp_model_minus_current_max_c"] = (
        None if pd.isna(model_tmax) or pd.isna(current_max) else float(model_tmax) - float(current_max)
    )
    nwp_features["nwp_future_minus_current_max_c"] = (
        None if pd.isna(future) or pd.isna(current_max) else float(future) - float(current_max)
    )


def _build_unimodal_shadow_candidate(*, champion, active_model_version: str, feature_row: dict) -> dict:
    shadow = project_unimodal_distribution(champion)
    forecast = shadow.to_payload()
    champion_payload = champion.to_payload()
    local_issue_hour = feature_row.get("local_issue_hour")
    expected_delta = shadow.expected_tmax_c - champion.expected_tmax_c
    ge_30_delta = (
        forecast["threshold_probabilities"].get("ge_30", 0.0)
        - champion_payload["threshold_probabilities"].get("ge_30", 0.0)
    )
    metadata = {
        "variant_version": EDDM_UNIMODAL_VERSION,
        "status": "shadow_only_does_not_affect_operational_forecast",
        "local_issue_hour": local_issue_hour,
        "forecast_phase": _forecast_phase(local_issue_hour),
        "projection_method": "least_squares_unimodal_projection_by_candidate_mode",
        "temperature": None,
        "temperature_source": "disabled_for_eddm_after_backtest_quality_check",
        "champion_model_version": active_model_version,
        "champion_unimodal_violation_count": unimodal_violation_count(champion),
        "shadow_unimodal_violation_count": unimodal_violation_count(shadow),
        "expected_tmax_delta_c": expected_delta,
    }
    return {
        "enabled": True,
        "active": True,
        "status": "shadow_only_does_not_affect_operational_forecast",
        "variant": EDDM_UNIMODAL_VARIANT,
        "model_version": EDDM_UNIMODAL_VERSION,
        "champion_model_version": active_model_version,
        "forecast": forecast,
        "champion_shape": _shape_summary(champion),
        "shadow_shape": _shape_summary(shadow),
        "comparison_to_champion": {
            "expected_tmax_delta_c": expected_delta,
            "most_likely_integer_delta_c": shadow.most_likely_integer_c - champion.most_likely_integer_c,
            "ge_30_probability_delta": ge_30_delta,
        },
        "metadata": metadata,
    }


def _shape_summary(distribution) -> dict:
    probs = distribution.probabilities
    deep_valleys = 0
    if len(probs) >= 3:
        for idx in range(1, len(probs) - 1):
            left, center, right = probs[idx - 1], probs[idx], probs[idx + 1]
            if min(left, right) >= 0.08 and center <= 0.60 * min(left, right):
                deep_valleys += 1
    return {
        "unimodal_violation_count": unimodal_violation_count(distribution),
        "deep_valley_count": int(deep_valleys),
        "max_adjacent_probability_jump": float(abs(pd.Series(probs).diff()).max()) if len(probs) >= 2 else 0.0,
    }


def _forecast_phase(local_issue_hour) -> str:
    if local_issue_hour is None or pd.isna(local_issue_hour):
        return "unknown"
    hour = float(local_issue_hour)
    if hour < 10:
        return "morning"
    if hour < 14:
        return "midday"
    if hour < 18:
        return "afternoon"
    return "evening"


def _predict_eddm_wind_advection_candidate(
    *,
    airport: str,
    target_date: date,
    issue_time_utc: datetime,
    base_feature_row: dict,
    metar: pd.DataFrame,
    spatial_metars: dict[str, pd.DataFrame],
    champion,
) -> dict:
    local_hour = int(pd.Timestamp(issue_time_utc).tz_convert("Europe/Berlin").hour)
    active_window = [EDDM_WIND_ADVECTION_LOCAL_HOUR_START, EDDM_WIND_ADVECTION_LOCAL_HOUR_END]
    base = {
        "enabled": airport.upper() == "EDDM",
        "active": False,
        "status": "shadow_only_does_not_affect_operational_forecast",
        "active_local_hour_window": active_window,
        "local_issue_hour": local_hour,
        "model_version": None,
        "reason": None,
    }
    if airport.upper() != "EDDM":
        return {**base, "enabled": False, "reason": "airport_not_supported"}
    if not (EDDM_WIND_ADVECTION_LOCAL_HOUR_START <= local_hour <= EDDM_WIND_ADVECTION_LOCAL_HOUR_END):
        return {**base, "reason": "outside_spatial_wind_advection_local_hour_window"}
    if not EDDM_WIND_ADVECTION_MODEL_PATH.exists():
        return {**base, "reason": f"missing_model:{EDDM_WIND_ADVECTION_MODEL_PATH}"}
    try:
        station_metars = {"EDDM": metar, **spatial_metars}
        advection_features = build_wind_advection_features(
            station_metars,
            target_date_local=target_date,
            issue_time_utc=issue_time_utc,
            timezone_name="Europe/Berlin",
            stations=EDDM_ADVECTION_STATIONS,
            target_station="EDDM",
        )
        if not advection_features.get("adv_leakage_check_passed", False):
            return {**base, "reason": "wind_advection_leakage_check_failed"}
        model = joblib.load(EDDM_WIND_ADVECTION_MODEL_PATH)
        metadata = _load_json(EDDM_WIND_ADVECTION_METADATA_PATH)
        feature_row = {**base_feature_row, **advection_features}
        dist = model.predict_distribution(feature_row)
        return {
            **base,
            "active": True,
            "reason": "active_midday_spatial_wind_advection_candidate",
            "model_version": metadata.get("model_version", getattr(model, "model_version", EDDM_WIND_ADVECTION_MODEL_PATH.stem)),
            "forecast": dist.to_payload(),
            "comparison_to_champion": _distribution_comparison(dist, champion),
            "wind_advection_features": {
                "available_station_count": advection_features.get("adv_available_station_count"),
                "mean_wind_speed_latest_kt": advection_features.get("adv_mean_wind_speed_latest_kt"),
                "mean_temp_trend_1h": advection_features.get("adv_mean_temp_trend_1h"),
                "mean_temp_trend_3h": advection_features.get("adv_mean_temp_trend_3h"),
                "mean_dewpoint_trend_3h": advection_features.get("adv_mean_dewpoint_trend_3h"),
                "mean_pressure_tendency_3h": advection_features.get("adv_mean_pressure_tendency_3h"),
                "any_cold_advection_signal": advection_features.get("adv_any_cold_advection_signal"),
                "any_warm_advection_signal": advection_features.get("adv_any_warm_advection_signal"),
                "any_frontal_passage_signal": advection_features.get("adv_any_frontal_passage_signal"),
                "neighbor_mean_minus_eddm_temp_trend_1h": advection_features.get("adv_neighbor_mean_minus_eddm_temp_trend_1h"),
                "max_feature_knowledge_time_utc": advection_features.get("adv_max_feature_knowledge_time_utc"),
            },
            "advection_stations": {
                station: {
                    "available": advection_features.get(f"adv_{station.lower()}_available"),
                    "wind_dir_latest_deg": advection_features.get(f"adv_{station.lower()}_wind_dir_latest_deg"),
                    "wind_speed_latest_kt": advection_features.get(f"adv_{station.lower()}_wind_speed_latest_kt"),
                    "temp_trend_1h": advection_features.get(f"adv_{station.lower()}_temp_trend_1h"),
                    "dewpoint_trend_3h": advection_features.get(f"adv_{station.lower()}_dewpoint_trend_3h"),
                    "pressure_tendency_3h": advection_features.get(f"adv_{station.lower()}_pressure_tendency_3h"),
                    "cold_advection_signal": advection_features.get(f"adv_{station.lower()}_cold_advection_signal"),
                    "warm_advection_signal": advection_features.get(f"adv_{station.lower()}_warm_advection_signal"),
                    "frontal_passage_signal": advection_features.get(f"adv_{station.lower()}_frontal_passage_signal"),
                }
                for station in EDDM_ADVECTION_STATIONS
            },
        }
    except Exception as exc:
        return {**base, "reason": f"spatial_wind_advection_candidate_unavailable:{exc}"}


def _annotate_metar_source(feature_row: dict, metar: pd.DataFrame) -> None:
    latest_time = pd.to_datetime(feature_row.get("latest_metar_time_utc"), utc=True, errors="coerce")
    if pd.isna(latest_time) or metar.empty:
        feature_row["metar_missing"] = True
        return
    frame = metar.copy()
    frame["observation_time_utc"] = pd.to_datetime(frame["observation_time_utc"], utc=True, errors="coerce")
    matches = frame[frame["observation_time_utc"] == latest_time]
    latest = matches.iloc[-1] if not matches.empty else frame.sort_values("observation_time_utc").iloc[-1]
    feature_row["latest_metar_source_id"] = latest.get("source_id")
    feature_row["latest_metar_record"] = {
        "observation_time_utc": _json_safe_value(latest.get("observation_time_utc")),
        "knowledge_time_utc": _json_safe_value(latest.get("knowledge_time_utc")),
        "ingest_time_utc": _json_safe_value(latest.get("ingest_time_utc")),
        "temperature_c": _json_safe_value(latest.get("temperature_c")),
        "dewpoint_c": _json_safe_value(latest.get("dewpoint_c")),
        "raw_metar": _json_safe_value(latest.get("raw_metar")),
        "source_id": _json_safe_value(latest.get("source_id")),
    }
    feature_row["max_metar_knowledge_time_utc"] = feature_row.get("max_feature_knowledge_time_utc")
    feature_row["metar_missing"] = False


def _json_safe_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _max_timestamp_string(*values) -> str | None:
    timestamps = pd.to_datetime([value for value in values if value is not None], utc=True, errors="coerce")
    timestamps = [timestamp for timestamp in timestamps if not pd.isna(timestamp)]
    if not timestamps:
        return None
    return max(pd.Timestamp(timestamp) for timestamp in timestamps).isoformat()


def _local_issue_schedule_offset_minutes(issue_time_utc: datetime, timezone_name: str, scheduled_hours: list[int]) -> float:
    issue_local = pd.Timestamp(issue_time_utc).tz_convert(timezone_name)
    if issue_local.minute == 30 and issue_local.second == 0:
        return 0.0
    actual = issue_local.hour * 60 + issue_local.minute + issue_local.second / 60.0
    scheduled = [hour * 60 for hour in scheduled_hours]
    return float(min(abs(actual - value) for value in scheduled))


def _predict_intraday_ml_shadow(feature_row: dict, model_path: str | Path = "data/models/intraday_ml_core_challenger_v1.joblib"):
    path = Path(model_path)
    if not path.exists():
        return None, {"active": False, "reason": "intraday_ml_artifact_unavailable"}
    try:
        model = joblib.load(path)
        return model.predict_distribution(feature_row)
    except (AttributeError, ImportError, ModuleNotFoundError, ValueError, TypeError) as exc:
        return None, {"active": False, "reason": f"intraday_ml_prediction_unavailable: {exc}"}


def _growth_potential_payload(ml_shadow_details: dict) -> dict:
    return {
        "source": "shadow_intraday_ml_remaining_upside",
        "active": bool(ml_shadow_details.get("active")),
        "probability_peak_already_passed": ml_shadow_details.get("probability_peak_already_passed"),
        "probability_upside_ge_1c": ml_shadow_details.get("probability_upside_ge_1c"),
        "probability_upside_ge_2c": ml_shadow_details.get("probability_upside_ge_2c"),
        "probability_upside_ge_3c": ml_shadow_details.get("probability_upside_ge_3c"),
        "calibration_status": ml_shadow_details.get("calibration_status"),
        "reason": ml_shadow_details.get("reason"),
    }


def _late_day_promotion_payload(phase_details: dict, champion_dist) -> dict:
    return {
        "active": False,
        "status": "not_promoted",
        "promotion_rule": "promote_shadow_intraday_ml_when_phase_arbitration_selects_late_day_ml",
        "selected_variant": phase_details.get("selected_variant"),
        "selection_reason": phase_details.get("selection_reason"),
        "local_issue_hour": phase_details.get("local_issue_hour"),
        "pre_promotion_expected_tmax_c": champion_dist.expected_tmax_c,
    }


def _synthetic_targets() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    seasonal = 13 + 12 * pd.Series((2 * 3.14159 * (doy - 200) / 366)).map(lambda x: __import__("math").cos(x)).to_numpy()
    deterministic_variation = 2.5 * pd.Series((2 * 3.14159 * (doy * 7 + dates.year.to_numpy()) / 29)).map(
        lambda x: __import__("math").sin(x)
    ).to_numpy()
    return pd.DataFrame(
        {
            "airport_icao": "EDDM",
            "target_date_local": dates.date.astype(str),
            "tmax_c": seasonal + deterministic_variation,
            "source_id": "synthetic.climatology.placeholder",
        }
    )


def _distribution_comparison(challenger, champion) -> dict:
    return {
        "expected_tmax_delta_c": challenger.expected_tmax_c - champion.expected_tmax_c,
        "median_tmax_delta_c": challenger.median_tmax_c - champion.median_tmax_c,
        "most_likely_integer_delta_c": challenger.most_likely_integer_c - champion.most_likely_integer_c,
        "ge_25_probability_delta": challenger.threshold_ge(25) - champion.threshold_ge(25),
        "ge_30_probability_delta": challenger.threshold_ge(30) - champion.threshold_ge(30),
    }


def _load_optional(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_metar_for_issue(airport: str, target_date: date, issue_time_utc: datetime) -> pd.DataFrame:
    frames = []
    start, _ = local_day_bounds_utc(target_date, "Europe/Berlin")
    context_start = min(pd.Timestamp(start), pd.Timestamp(issue_time_utc) - pd.Timedelta(hours=24))
    context_end = pd.Timestamp(issue_time_utc)
    for path in (f"data/interim/metar_iem_{airport}.parquet", f"data/forecasts/awc_metar_live_{airport}.parquet"):
        df = _load_metar_context(path, context_start, context_end)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    metar = pd.concat(frames, ignore_index=True)
    times = pd.to_datetime(metar["observation_time_utc"], utc=True)
    return metar[(times >= context_start) & (times <= context_end)].copy()


def _load_spatial_metars_for_issue(airport: str, target_date: date, issue_time_utc: datetime) -> dict[str, pd.DataFrame]:
    stations = SPATIAL_STATIONS_BY_AIRPORT.get(airport.upper(), ())
    if not stations:
        return {}
    return {
        station: frame
        for station in stations
        if not (frame := _load_metar_for_issue(station, target_date, issue_time_utc)).empty
    }


def _load_metar_context(path: str | Path, context_start: pd.Timestamp, context_end: pd.Timestamp) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    columns = [
        "station",
        "observation_time_utc",
        "knowledge_time_utc",
        "source_id",
        "raw_metar",
        "temperature_c",
        "dewpoint_c",
        "qnh_hpa",
        "wind_direction_deg",
        "wind_speed_kt",
        "gust_kt",
        "cloud_layers",
        "ceiling_ft",
        "cavok",
    ]
    try:
        return pd.read_parquet(
            p,
            columns=columns,
            filters=[
                ("observation_time_utc", ">=", context_start.to_pydatetime()),
                ("observation_time_utc", "<=", context_end.to_pydatetime()),
            ],
        )
    except (KeyError, ValueError, TypeError):
        df = pd.read_parquet(p)
        missing = [column for column in columns if column not in df.columns]
        for column in missing:
            df[column] = None
        times = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
        return df.loc[(times >= context_start) & (times <= context_end), columns].copy()


def _load_nwp_for_issue(target_date: date, issue_time_utc: datetime) -> pd.DataFrame:
    frames = [
        frame
        for path in (
            "data/forecasts/open_meteo_archive.parquet",
            "data/forecasts/open_meteo_single_runs_icon_d2.parquet",
        )
        if not (frame := _load_optional(path)).empty
    ]
    if not frames:
        return pd.DataFrame()
    nwp = pd.concat(frames, ignore_index=True)
    nwp = nwp[nwp["target_date_local"].astype(str) == target_date.isoformat()].copy()
    if nwp.empty:
        return nwp
    availability = pd.to_datetime(nwp["model_availability_time_utc"], utc=True)
    return nwp[availability <= pd.Timestamp(issue_time_utc)].copy()


def _load_taf_for_issue(airport: str, issue_time_utc: datetime) -> pd.DataFrame:
    frames = []
    for path in (f"data/interim/taf_iem_{airport}.parquet", f"data/forecasts/awc_taf_live_{airport}.parquet"):
        df = _load_optional(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    taf = pd.concat(frames, ignore_index=True)
    if "knowledge_time_utc" not in taf.columns:
        return taf
    knowledge = pd.to_datetime(taf["knowledge_time_utc"], utc=True, errors="coerce")
    return taf[knowledge <= pd.Timestamp(issue_time_utc)].copy()
