from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from weather_tmax_bot.features.build_features import build_feature_row
from weather_tmax_bot.models.baselines import ClimatologyBaseline
from weather_tmax_bot.models.disagreement import assess_model_disagreement
from weather_tmax_bot.models.extrapolation import detect_feature_extrapolation
from weather_tmax_bot.models.intraday_update import apply_intraday_update
from weather_tmax_bot.models.model_registry import load_model, resolve_active_artifacts
from weather_tmax_bot.models.phase_arbitration import build_phase_arbitrated_candidate
from weather_tmax_bot.models.safe_blend import build_blended_shadow_candidate
from weather_tmax_bot.temporal.freshness import assess_feature_freshness
from weather_tmax_bot.temporal.source_compatibility import assess_source_compatibility
from weather_tmax_bot.utils.time import local_day_bounds_utc


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
        taf = _load_taf_for_issue(airport, issue_time_utc)
        nwp = _load_nwp_for_issue(target_date, issue_time_utc)
        feature_row = build_feature_row(
            airport_icao=airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date,
            metar=metar,
            taf=taf,
            nwp=nwp,
        )
        model = load_model(resolved_model_path)
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
            "model_version": active.get("active_model_version") or Path(resolved_model_path).stem,
            "feature_snapshot": feature_row,
            "warnings": warnings,
        }
    dist = predict_with_climatology(target_date, daily_target_path=daily_target_path)
    warnings.append("Quantile model unavailable; climatology MVP distribution used.")
    return dist, {"model_version": "climatology_mvp", "feature_snapshot": {}, "warnings": warnings}


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
