from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from weather_tmax_bot.bot.forecast_log import log_forecast
from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.data.open_meteo import fetch_open_meteo_live_extract
from weather_tmax_bot.features.metar_upside_dataset import build_current_metar_upside_features
from weather_tmax_bot.features.nwp_features import build_nwp_features
from weather_tmax_bot.features.spatial_metar import DEFAULT_SPATIAL_STATIONS, build_spatial_metar_features
from weather_tmax_bot.features.wind_advection import DEFAULT_ADVECTION_STATIONS, build_wind_advection_features
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_intraday_survival import apply_metar_intraday_survival_layer
from weather_tmax_bot.notifications.telegram import notify_if_configured
from weather_tmax_bot.operations.refresh import refresh_awc_live
from weather_tmax_bot.utils.time import parse_issue_time, to_local_date


AIRPORT = "LFPB"
TIMEZONE = "Europe/Paris"
LATITUDE = 48.969444
LONGITUDE = 2.441389
MODEL_PATH = Path("data/models/lfpb_metar_tmax_upside_v1.joblib")
METADATA_PATH = Path("data/models/lfpb_metar_tmax_upside_v1.metadata.json")
LIVE_NWP_PATH = Path("data/forecasts/open_meteo_archive_LFPB.parquet")
ENHANCED_ICON_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_icon_d2_LFPB_enhanced.parquet")
HISTORICAL_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
ECMWF_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_ecmwf_ifs_LFPB.parquet")
AROME_NWP_PATHS = [
    Path("data/forecasts/open_meteo_single_runs_meteofrance_arome_france_hd_LFPB_holdout_analysis.parquet"),
    Path("data/forecasts/open_meteo_single_runs_meteofrance_arome_france_hd_LFPB_analysis.parquet"),
]
SURVIVAL_DATASET_PATH = Path("data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
SPATIAL_CANDIDATE_MODEL_PATH = Path("data/models/lfpb_metar_tmax_icon_d2_spatial_wind_advection_v1.joblib")
SPATIAL_CANDIDATE_METADATA_PATH = Path("data/models/lfpb_metar_tmax_icon_d2_spatial_wind_advection_v1.metadata.json")
SPATIAL_CANDIDATE_LOCAL_HOUR_START = 12
SPATIAL_CANDIDATE_LOCAL_HOUR_END = 18


def main() -> None:
    args = _parse_args()
    refresh_summary = None
    issue_is_now = args.issue_time in (None, "now")
    if args.auto_refresh and issue_is_now:
        refresh_summary = {"awc": refresh_awc_live(args.airport)}
        if args.spatial_candidate:
            refresh_summary["spatial_awc"] = _refresh_spatial_awc_live()
        if args.refresh_nwp:
            refresh_summary["open_meteo_nwp"] = _refresh_open_meteo_live(args.airport, None)
    issue_time_utc = parse_issue_time(args.issue_time)
    target_date = date.fromisoformat(args.target_date) if args.target_date else to_local_date(issue_time_utc, TIMEZONE)
    if args.auto_refresh and not issue_is_now:
        refresh_summary = {"awc": refresh_awc_live(args.airport)}
        if args.spatial_candidate:
            refresh_summary["spatial_awc"] = _refresh_spatial_awc_live()
        if args.refresh_nwp:
            refresh_summary["open_meteo_nwp"] = _refresh_open_meteo_live(args.airport, target_date)
    metar = _load_metar(args.airport)
    model = joblib.load(args.model_path)
    metadata = _load_json(args.metadata_path)
    feature_row = build_current_metar_upside_features(
        metar,
        airport_icao=args.airport,
        target_date_local=target_date,
        issue_time_utc=issue_time_utc,
        timezone_name=TIMEZONE,
    )
    nwp_features = _load_nwp_features(target_date, issue_time_utc)
    _add_nwp_relative_features(nwp_features, feature_row)
    feature_row.update(nwp_features)
    feature_row["max_feature_knowledge_time_utc"] = _max_timestamp_string(
        feature_row.get("max_feature_knowledge_time_utc"),
        feature_row.get("max_nwp_knowledge_time_utc"),
    )
    if hasattr(model, "residuals_by_hour") and pd.isna(feature_row.get("model_tmax_c")):
        raise FileNotFoundError(
            "ICON-aware LFPB METAR Tmax model requires NWP features; "
            "run with --auto-refresh --refresh-nwp or provide an Open-Meteo archive."
        )
    base_distribution = model.predict_distribution(feature_row)
    survival_adjustment = apply_metar_intraday_survival_layer(
        base_distribution,
        feature_row,
        historical_dataset_path=SURVIVAL_DATASET_PATH,
    )
    distribution = survival_adjustment.distribution
    feature_row["production_expected_tmax_c"] = distribution.expected_tmax_c
    spatial_candidate = _predict_spatial_candidate(
        enabled=args.spatial_candidate,
        target_date=target_date,
        issue_time_utc=issue_time_utc,
        base_feature_row=feature_row,
    )
    base_production_distribution = distribution
    base_production_model_version = metadata.get("model_version", "lfpb_metar_tmax_upside_v1")
    production_selection = {
        "selected": "base_icon_d2",
        "reason": "spatial_candidate_not_promoted",
        "promote_spatial_candidate_requested": bool(args.promote_spatial_candidate),
    }
    if args.promote_spatial_candidate and spatial_candidate.get("active"):
        promoted = _distribution_from_payload(spatial_candidate.get("forecast"))
        if promoted is not None:
            distribution = promoted
            production_selection = {
                "selected": "spatial_wind_advection_candidate",
                "reason": "active_spatial_candidate_promoted_by_job_flag",
                "promote_spatial_candidate_requested": True,
                "base_model_version": base_production_model_version,
                "promoted_model_version": spatial_candidate.get("model_version"),
                "base_expected_tmax_c": base_production_distribution.expected_tmax_c,
                "promoted_expected_tmax_c": distribution.expected_tmax_c,
            }
            feature_row["production_expected_tmax_c"] = distribution.expected_tmax_c
    production_model_version = (
        spatial_candidate.get("model_version")
        if production_selection["selected"] == "spatial_wind_advection_candidate"
        else base_production_model_version
    )
    nwp_source_diagnostics = _build_nwp_source_diagnostics(feature_row)
    forecast_id = None
    if args.log:
        forecast_id = log_forecast(
            airport=args.airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date,
            distribution=distribution,
            feature_snapshot={
                **feature_row,
                "data_sources_used": ["awc.metar.live.LFPB", "iem.metar.archive.LFPB", feature_row.get("latest_nwp_source_id")],
                "target": "METAR_Tmax",
                "model_family": "metar_tmax_remaining_upside",
                "intraday_survival_layer": survival_adjustment.details,
                "base_forecast_before_intraday_survival": base_distribution.to_payload(),
                "base_production_before_spatial_promotion": base_production_distribution.to_payload(),
                "production_selection": production_selection,
                "nwp_source_diagnostics": nwp_source_diagnostics,
                "spatial_candidate": spatial_candidate,
            },
            model_version=production_model_version,
        )
    payload = {
        "forecast_id": forecast_id,
        "airport": args.airport,
        "target": "METAR_Tmax",
        "target_description": "daily maximum temperature reported by METAR",
        "target_date_local": target_date.isoformat(),
        "timezone": TIMEZONE,
        "issue_time_utc": issue_time_utc.isoformat(),
        "model_version": production_model_version,
        "calibration": (metadata.get("calibration_metadata") or {}).get("calibration_method", "unknown"),
        "calibration_attached": _calibration_attached(model),
        "forecast": distribution.to_payload(),
        "base_forecast_before_intraday_survival": base_distribution.to_payload(),
        "base_production_before_spatial_promotion": base_production_distribution.to_payload(),
        "intraday_survival_layer": survival_adjustment.details,
        "production_selection": production_selection,
        "spatial_candidate": spatial_candidate,
        "metar_signal": {
            "latest_metar_time_utc": feature_row.get("latest_metar_time_utc"),
            "latest_metar_temp_c": feature_row.get("latest_metar_temp_c"),
            "current_metar_max_c": feature_row.get("current_metar_max_c"),
            "drop_from_current_max_c": feature_row.get("drop_from_current_max_c"),
            "metar_count_so_far": feature_row.get("metar_count_so_far"),
            "temp_trend_1h": feature_row.get("temp_trend_1h"),
            "temp_trend_3h": feature_row.get("temp_trend_3h"),
            "temp_trend_last_2_metars": feature_row.get("temp_trend_last_2_metars"),
            "temp_slope_since_sunrise": feature_row.get("temp_slope_since_sunrise"),
            "pressure_tendency_3h": feature_row.get("pressure_tendency_3h"),
            "dewpoint_depression_latest": feature_row.get("dewpoint_depression_latest"),
            "cloud_cover_proxy_latest": feature_row.get("cloud_cover_proxy_latest"),
            "cloud_cover_proxy_trend_2h": feature_row.get("cloud_cover_proxy_trend_2h"),
            "metar_minutes_since_current_max": feature_row.get("metar_minutes_since_current_max"),
            "has_rain_recent_metar": feature_row.get("has_rain_recent_metar"),
            "rain_started_after_current_max": feature_row.get("rain_started_after_current_max"),
            "cb_tcu_appeared_after_current_max": feature_row.get("cb_tcu_appeared_after_current_max"),
            "showers_appeared_after_current_max": feature_row.get("showers_appeared_after_current_max"),
            "latest_metar_raw": feature_row.get("latest_metar_raw"),
        },
        "data_lineage": {
            "max_feature_knowledge_time_utc": feature_row.get("max_feature_knowledge_time_utc"),
            "source": "AWC live METAR if refreshed; local AWC/IEM METAR archive fallback",
            "latest_nwp_source_id": feature_row.get("latest_nwp_source_id"),
            "selected_nwp_source_label": feature_row.get("selected_nwp_source_label"),
            "available_nwp_source_labels": feature_row.get("available_nwp_source_labels"),
            "nwp_tmax_spread_c": feature_row.get("nwp_tmax_spread_c"),
            "nwp_tmax_by_source": feature_row.get("nwp_tmax_by_source"),
            "nwp_source_diagnostics": nwp_source_diagnostics,
            "max_nwp_knowledge_time_utc": feature_row.get("max_nwp_knowledge_time_utc"),
            "model_tmax_c": feature_row.get("model_tmax_c"),
            "model_future_temp_max_c": feature_row.get("model_future_temp_max_c"),
            "leakage_check_passed": feature_row.get("leakage_check_passed"),
        },
        "refresh_summary": refresh_summary,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    text = _format_message(payload)
    print(text)
    print(f"\nWrote {report_path}")
    if args.notify:
        result = notify_if_configured(text)
        print(json.dumps({"telegram": result}, indent=2, default=str))


def _load_metar(airport: str) -> pd.DataFrame:
    frames = []
    for path in [
        Path(f"data/forecasts/awc_metar_live_{airport}.parquet"),
        Path(f"data/interim/metar_iem_{airport}.parquet"),
    ]:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No METAR data found for {airport}; run with --auto-refresh or download historical METAR first")
    frame = pd.concat(frames, ignore_index=True)
    if "raw_record_hash" in frame.columns:
        frame = frame.drop_duplicates(subset=["raw_record_hash"], keep="last")
    else:
        frame = frame.drop_duplicates(subset=["observation_time_utc", "raw_metar"], keep="last")
    return frame


def _refresh_spatial_awc_live() -> dict:
    summary = {}
    for station in DEFAULT_SPATIAL_STATIONS:
        try:
            summary[station] = refresh_awc_live(station)
        except Exception as exc:
            summary[station] = {"error": str(exc)}
    return summary


def _predict_spatial_candidate(
    *,
    enabled: bool,
    target_date: date,
    issue_time_utc,
    base_feature_row: dict,
) -> dict:
    local_hour = int(pd.Timestamp(issue_time_utc).tz_convert(TIMEZONE).hour)
    active_window = [SPATIAL_CANDIDATE_LOCAL_HOUR_START, SPATIAL_CANDIDATE_LOCAL_HOUR_END]
    base = {
        "enabled": bool(enabled),
        "active": False,
        "active_local_hour_window": active_window,
        "local_issue_hour": local_hour,
        "model_version": None,
        "reason": None,
    }
    if not enabled:
        base["reason"] = "spatial_candidate_disabled"
        return base
    if not (SPATIAL_CANDIDATE_LOCAL_HOUR_START <= local_hour <= SPATIAL_CANDIDATE_LOCAL_HOUR_END):
        base["reason"] = "outside_spatial_candidate_local_hour_window"
        return base
    if not SPATIAL_CANDIDATE_MODEL_PATH.exists():
        base["reason"] = f"missing_model:{SPATIAL_CANDIDATE_MODEL_PATH}"
        return base
    try:
        neighbor_metars = {station: _load_metar(station) for station in DEFAULT_SPATIAL_STATIONS}
        spatial_features = build_spatial_metar_features(
            base_feature_row,
            neighbor_metars,
            target_date_local=target_date,
            issue_time_utc=issue_time_utc,
            timezone_name=TIMEZONE,
            stations=DEFAULT_SPATIAL_STATIONS,
        )
        station_metars = {"LFPB": _load_metar("LFPB"), **neighbor_metars}
        advection_features = build_wind_advection_features(
            station_metars,
            target_date_local=target_date,
            issue_time_utc=issue_time_utc,
            timezone_name=TIMEZONE,
            stations=DEFAULT_ADVECTION_STATIONS,
        )
        spatial_row = {**base_feature_row, **spatial_features, **advection_features}
        if not spatial_features.get("spatial_leakage_check_passed", False):
            base["reason"] = "spatial_leakage_check_failed"
            return base
        if not advection_features.get("adv_leakage_check_passed", False):
            base["reason"] = "wind_advection_leakage_check_failed"
            return base
        if int(spatial_features.get("spatial_available_station_count") or 0) <= 0:
            base["reason"] = "no_spatial_neighbor_metar_available_as_of_issue_time"
            return base
        model = joblib.load(SPATIAL_CANDIDATE_MODEL_PATH)
        metadata = _load_json(SPATIAL_CANDIDATE_METADATA_PATH)
        raw_distribution = model.predict_distribution(spatial_row)
        survival_adjustment = apply_metar_intraday_survival_layer(
            raw_distribution,
            spatial_row,
            historical_dataset_path=SURVIVAL_DATASET_PATH,
        )
        final_distribution = survival_adjustment.distribution
        production_expected = base_feature_row.get("production_expected_tmax_c")
        return {
            **base,
            "active": True,
            "reason": "active_midday_spatial_wind_advection_candidate",
            "model_version": metadata.get("model_version", getattr(model, "model_version", "spatial_wind_advection_candidate")),
            "forecast": final_distribution.to_payload(),
            "forecast_before_intraday_survival": raw_distribution.to_payload(),
            "intraday_survival_layer": survival_adjustment.details,
            "spatial_features": {
                "available_station_count": spatial_features.get("spatial_available_station_count"),
                "latest_temp_mean_c": spatial_features.get("spatial_latest_temp_mean_c"),
                "latest_temp_max_c": spatial_features.get("spatial_latest_temp_max_c"),
                "current_max_mean_c": spatial_features.get("spatial_current_max_mean_c"),
                "current_max_max_c": spatial_features.get("spatial_current_max_max_c"),
                "latest_minus_lfpb_latest_mean_c": spatial_features.get("spatial_latest_minus_lfpb_latest_mean_c"),
                "max_minus_lfpb_current_max_mean_c": spatial_features.get("spatial_max_minus_lfpb_current_max_mean_c"),
                "any_neighbor_above_lfpb_latest": spatial_features.get("spatial_any_neighbor_above_lfpb_latest"),
                "any_neighbor_above_lfpb_current_max": spatial_features.get("spatial_any_neighbor_above_lfpb_current_max"),
                "max_feature_knowledge_time_utc": spatial_features.get("spatial_max_feature_knowledge_time_utc"),
            },
            "wind_advection_features": {
                "available_station_count": advection_features.get("adv_available_station_count"),
                "mean_wind_speed_latest_kt": advection_features.get("adv_mean_wind_speed_latest_kt"),
                "mean_temp_trend_1h": advection_features.get("adv_mean_temp_trend_1h"),
                "mean_temp_trend_3h": advection_features.get("adv_mean_temp_trend_3h"),
                "mean_dewpoint_trend_3h": advection_features.get("adv_mean_dewpoint_trend_3h"),
                "mean_pressure_tendency_3h": advection_features.get("adv_mean_pressure_tendency_3h"),
                "any_north_sector": advection_features.get("adv_any_north_sector"),
                "any_south_sector": advection_features.get("adv_any_south_sector"),
                "any_cold_advection_signal": advection_features.get("adv_any_cold_advection_signal"),
                "any_warm_advection_signal": advection_features.get("adv_any_warm_advection_signal"),
                "any_frontal_passage_signal": advection_features.get("adv_any_frontal_passage_signal"),
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
                for station in DEFAULT_ADVECTION_STATIONS
            },
            "neighbor_stations": {
                station: {
                    "available": spatial_features.get(f"spatial_{station.lower()}_available"),
                    "latest_temp_c": spatial_features.get(f"spatial_{station.lower()}_latest_temp_c"),
                    "current_max_c": spatial_features.get(f"spatial_{station.lower()}_current_max_c"),
                    "age_minutes": spatial_features.get(f"spatial_{station.lower()}_age_minutes"),
                    "has_rain_recent": spatial_features.get(f"spatial_{station.lower()}_has_rain_recent"),
                }
                for station in DEFAULT_SPATIAL_STATIONS
            },
            "expected_delta_vs_production_c": (
                None
                if production_expected is None or pd.isna(production_expected)
                else final_distribution.expected_tmax_c - float(production_expected)
            ),
        }
    except Exception as exc:
        return {**base, "reason": f"spatial_candidate_unavailable:{exc}"}


def _refresh_open_meteo_live(airport: str, target_date_local: date | None) -> dict:
    target = target_date_local or date.today()
    rows = fetch_open_meteo_live_extract(
        airport_icao=airport,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        target_date_local=target,
        timezone_name=TIMEZONE,
    )
    if rows.empty:
        return {"rows_fetched": 0, "archive_rows": _parquet_rows(LIVE_NWP_PATH)}
    NWPArchive(LIVE_NWP_PATH).append_extract(rows)
    return {"rows_fetched": len(rows), "archive_rows": _parquet_rows(LIVE_NWP_PATH)}


def _load_nwp_features(target_date_local: date, issue_time_utc) -> dict:
    candidates = []
    source_specs = [
        ("live_icon_d2", LIVE_NWP_PATH),
        ("enhanced_icon_d2", ENHANCED_ICON_NWP_PATH),
        ("historical_icon_d2", HISTORICAL_NWP_PATH),
        ("ecmwf_ifs", ECMWF_NWP_PATH),
        *[(f"arome_france_hd_{idx}", path) for idx, path in enumerate(AROME_NWP_PATHS, start=1)],
    ]
    for label, path in source_specs:
        candidate = _nwp_candidate_from_path(label, path, target_date_local, issue_time_utc)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return {
            "nwp_missing": True,
            "model_tmax_c": None,
            "available_nwp_source_labels": [],
            "nwp_tmax_by_source": {},
            "nwp_tmax_spread_c": None,
        }
    selected = candidates[0]
    features = dict(selected["features"])
    tmax_by_source = {
        item["label"]: item["features"].get("model_tmax_c")
        for item in candidates
        if not pd.isna(item["features"].get("model_tmax_c"))
    }
    tmax_values = [float(value) for value in tmax_by_source.values() if not pd.isna(value)]
    features.update(
        {
            "selected_nwp_source_label": selected["label"],
            "available_nwp_source_labels": [item["label"] for item in candidates],
            "available_nwp_source_ids": [item["features"].get("latest_nwp_source_id") for item in candidates],
            "nwp_tmax_by_source": tmax_by_source,
            "nwp_tmax_spread_c": (max(tmax_values) - min(tmax_values)) if len(tmax_values) >= 2 else 0.0,
            "nwp_tmax_mean_c": sum(tmax_values) / len(tmax_values) if tmax_values else None,
            "nwp_fallback_used": selected["label"] not in {"live_icon_d2", "enhanced_icon_d2", "historical_icon_d2"},
        }
    )
    return features


def _nwp_candidate_from_path(label: str, path: Path, target_date_local: date, issue_time_utc) -> dict | None:
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if frame.empty:
        return None
    if "airport_icao" in frame.columns:
        frame = frame[frame["airport_icao"].fillna(AIRPORT) == AIRPORT].copy()
    if "target_date_local" not in frame.columns:
        return None
    frame = frame[frame["target_date_local"].astype(str) == target_date_local.isoformat()].copy()
    features = build_nwp_features(frame, issue_time_utc)
    if features.get("nwp_missing") or pd.isna(features.get("model_tmax_c")):
        return None
    return {"label": label, "path": str(path), "features": features}


def _add_nwp_relative_features(nwp_features: dict, metar_features: dict) -> None:
    model_tmax = nwp_features.get("model_tmax_c")
    current_max = metar_features.get("current_metar_max_c")
    future = nwp_features.get("model_future_temp_max_c")
    nwp_features["nwp_model_minus_current_max_c"] = (
        None if pd.isna(model_tmax) or pd.isna(current_max) else float(model_tmax) - float(current_max)
    )
    nwp_features["nwp_future_minus_current_max_c"] = (
        None if pd.isna(future) or pd.isna(current_max) else float(future) - float(current_max)
    )


def _build_nwp_source_diagnostics(feature_row: dict) -> dict:
    selected = feature_row.get("selected_nwp_source_label")
    labels = feature_row.get("available_nwp_source_labels") or []
    spread = feature_row.get("nwp_tmax_spread_c")
    fallback_used = bool(feature_row.get("nwp_fallback_used"))
    if not labels:
        return {
            "level": "missing",
            "reason": "no_nwp_source_available",
            "diagnostic_only": True,
            "recommended_uncertainty_padding_c": None,
        }

    spread_value = None if spread is None or pd.isna(spread) else float(spread)
    level = "low"
    reasons = []
    padding = 0.0
    if fallback_used:
        level = "moderate"
        padding = max(padding, 0.5)
        reasons.append("non_icon_fallback_selected")
    if spread_value is not None:
        if spread_value >= 2.5:
            level = "high"
            padding = max(padding, 1.0)
            reasons.append("large_cross_source_tmax_spread")
        elif spread_value >= 1.5 and level != "high":
            level = "moderate"
            padding = max(padding, 0.5)
            reasons.append("moderate_cross_source_tmax_spread")
    if selected not in {"live_icon_d2", "enhanced_icon_d2", "historical_icon_d2"} and level == "low":
        level = "moderate"
        padding = max(padding, 0.5)
        reasons.append("source_family_shift")

    return {
        "level": level,
        "reason": "+".join(reasons) if reasons else "primary_icon_source_selected",
        "diagnostic_only": True,
        "recommended_uncertainty_padding_c": padding,
    }


def _parquet_rows(path: Path) -> int:
    return 0 if not path.exists() else len(pd.read_parquet(path))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _distribution_from_payload(payload: dict | None) -> TmaxDistribution | None:
    probabilities = (payload or {}).get("probabilities_by_integer_c")
    if not probabilities:
        return None
    bins = [int(key) for key in probabilities.keys()]
    probs = [float(value) for value in probabilities.values()]
    return TmaxDistribution(bins, probs)


def _calibration_attached(model) -> bool:
    direct = getattr(model, "calibrator", None)
    if direct is not None:
        return True
    base = getattr(model, "base_model", None)
    if getattr(base, "calibrator", None) is not None:
        return True
    ml_model = getattr(model, "ml_model", None)
    return bool(getattr(ml_model, "calibrator", None))


def _format_message(payload: dict) -> str:
    forecast = payload["forecast"]
    signal = payload["metar_signal"]
    survival = payload.get("intraday_survival_layer") or {}
    thresholds = forecast["threshold_probabilities"]
    bins = {
        int(bin_c): float(probability)
        for bin_c, probability in forecast["probabilities_by_integer_c"].items()
        if float(probability) >= 0.01
    }
    bin_text = "\n".join(f"{bin_c:+d} °C: <b>{probability:.1%}</b>" for bin_c, probability in sorted(bins.items()))
    if not bin_text:
        bin_text = "Нет корзин выше 1%."
    return "\n".join(
        [
            f"<b>Прогноз METAR Tmax: {payload['airport']}</b>",
            f"Дата: <b>{payload['target_date_local']}</b>",
            f"Цель: максимум температуры, который покажут METAR за день",
            f"Выпуск UTC: <code>{payload['issue_time_utc']}</code>",
            f"ID прогноза: <code>{payload.get('forecast_id') or 'не логировался'}</code>",
            "",
            "<b>Температурный прогноз</b>",
            f"Ожидаемый METAR Tmax: <b>{forecast['expected_tmax_c']:.1f} °C</b>",
            f"Медиана: {forecast['median_tmax_c']:.1f} °C",
            f"Самая вероятная корзина: <b>{forecast['most_likely_integer_c']} °C</b>",
            f"Интервал 80%: {forecast['intervals']['80'][0]:.1f}...{forecast['intervals']['80'][1]:.1f} °C",
            "",
            "<b>Вероятности по градусам</b>",
            bin_text,
            "",
            "<b>Вероятности событий</b>",
            f"Не ниже +20 °C: {thresholds['ge_20']:.1%}",
            f"Не ниже +25 °C: {thresholds['ge_25']:.1%}",
            f"Не ниже +30 °C: {thresholds['ge_30']:.1%}",
            "",
            "<b>METAR-сигнал</b>",
            f"Последняя температура: {float(signal['latest_metar_temp_c']):.1f} °C",
            f"Текущий максимум по METAR: {float(signal['current_metar_max_c']):.1f} °C",
            f"Падение от максимума: {float(signal['drop_from_current_max_c']):.1f} °C",
            f"METAR за день: {int(signal['metar_count_so_far'])}",
            f"Тренд 1ч: {_fmt_float(signal.get('temp_trend_1h'))} °C",
            f"Тренд 3ч: {_fmt_float(signal.get('temp_trend_3h'))} °C",
            f"Последние 2 METAR: {_fmt_float(signal.get('temp_trend_last_2_metars'))} °C",
            f"Тренд с утра: {_fmt_float(signal.get('temp_slope_since_sunrise'))} °C",
            f"Давление 3ч: {_fmt_float(signal.get('pressure_tendency_3h'))} hPa",
            f"Дефицит точки росы: {_fmt_float(signal.get('dewpoint_depression_latest'))} °C",
            f"Облачность proxy: {_fmt_float(signal.get('cloud_cover_proxy_latest'))}/8, тренд 2ч {_fmt_float(signal.get('cloud_cover_proxy_trend_2h'))}",
            f"Минут с текущего максимума: {_fmt_float(signal.get('metar_minutes_since_current_max'))}",
            f"Дождь недавно: {'да' if signal.get('has_rain_recent_metar') else 'нет'}",
            f"Дождь после максимума: {'да' if signal.get('rain_started_after_current_max') else 'нет'}",
            f"CB/TCU после максимума: {'да' if signal.get('cb_tcu_appeared_after_current_max') else 'нет'}",
            f"Ливни после максимума: {'да' if signal.get('showers_appeared_after_current_max') else 'нет'}",
            "",
            "<b>Intraday survival</b>",
            f"Слой активен: {'да' if survival.get('active') else 'нет'}",
            f"Шанс роста минимум на +1 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_1c'))}",
            f"Шанс роста минимум на +2 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_2c'))}",
            f"Шанс роста минимум на +3 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_3c'))}",
            f"До коррекции +1 °C: {_fmt_percent(survival.get('original_probability_upside_ge_1c'))}",
            f"Вес коррекции: {_fmt_percent(survival.get('effective_strength'))}",
            *_format_rebound_guard_lines(survival),
            "",
            *_format_spatial_candidate_lines(payload),
            "<b>Калибровка</b>",
            f"Статус: {'включена' if payload.get('calibration_attached') else 'не включена'}",
            f"Метод: <code>{payload.get('calibration')}</code>",
            "",
            "<b>Данные</b>",
            f"Последний METAR: <code>{signal.get('latest_metar_time_utc')}</code>",
            *_format_nwp_source_lines(payload),
            f"Max knowledge time: <code>{payload['data_lineage'].get('max_feature_knowledge_time_utc')}</code>",
            f"Leakage check: {'ok' if payload['data_lineage'].get('leakage_check_passed') else 'failed'}",
            "",
            f"<code>{signal.get('latest_metar_raw')}</code>",
        ]
    )


def _fmt_float(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):+.1f}"


def _format_nwp_source_lines(payload: dict) -> list[str]:
    lineage = payload.get("data_lineage") or {}
    selected = lineage.get("selected_nwp_source_label") or "unknown"
    labels = lineage.get("available_nwp_source_labels") or []
    diagnostics = lineage.get("nwp_source_diagnostics") or {}
    available = ", ".join(str(label) for label in labels) if labels else "none"
    level = diagnostics.get("level") or "unknown"
    reason = diagnostics.get("reason") or "unknown"
    padding = diagnostics.get("recommended_uncertainty_padding_c")
    padding_text = "n/a" if padding is None or pd.isna(padding) else f"{float(padding):.1f} C"
    return [
        f"NWP source: <code>{selected}</code> (available: <code>{available}</code>)",
        f"NWP Tmax: {_fmt_float(lineage.get('model_tmax_c'))} °C",
        f"NWP future max: {_fmt_float(lineage.get('model_future_temp_max_c'))} °C",
        f"NWP Tmax spread: {_fmt_float(lineage.get('nwp_tmax_spread_c'))} °C",
        f"NWP uncertainty: <code>{level}</code>, padding note {padding_text}, <code>{reason}</code>",
    ]


def _format_rebound_guard_lines(survival: dict) -> list[str]:
    guard = (survival or {}).get("rebound_guard") or {}
    if not guard:
        return []
    if not guard.get("active"):
        return []
    floors = guard.get("floors") or {}
    floor_1c = floors.get("1")
    return [
        "Rebound guard: <b>active</b>",
        f"Rebound signal: +{float(guard.get('strong_rebound_c', 0.0)):.1f} °C",
        f"Minimum P(+1 °C): {_fmt_percent(floor_1c)}",
    ]


def _format_spatial_candidate_lines(payload: dict) -> list[str]:
    candidate = payload.get("spatial_candidate") or {}
    selection = payload.get("production_selection") or {}
    promoted = selection.get("selected") == "spatial_wind_advection_candidate"
    if not candidate.get("enabled", False):
        return []
    if not candidate.get("active", False):
        reason = str(candidate.get("reason") or "")
        if reason == "outside_spatial_candidate_local_hour_window":
            return []
        return [
            "<b>Spatial + wind/advection candidate</b>",
            "Кандидат не влияет на основной прогноз.",
            f"Статус: не активен ({reason or 'причина неизвестна'})",
            "",
        ]
    forecast = candidate.get("forecast") or {}
    thresholds = forecast.get("threshold_probabilities") or {}
    spatial = candidate.get("spatial_features") or {}
    advection = candidate.get("wind_advection_features") or {}
    neighbors = candidate.get("neighbor_stations") or {}
    advection_stations = candidate.get("advection_stations") or {}
    delta = candidate.get("expected_delta_vs_production_c")
    neighbor_lines = []
    for station, info in neighbors.items():
        if not info.get("available"):
            neighbor_lines.append(f"{station}: нет свежих данных")
            continue
        neighbor_lines.append(
            f"{station}: сейчас {_fmt_signed_plain(info.get('latest_temp_c'))} °C, "
            f"max {_fmt_signed_plain(info.get('current_max_c'))} °C, "
            f"возраст {_fmt_plain(info.get('age_minutes'))} мин"
        )
    advection_lines = []
    for station, info in advection_stations.items():
        if not info.get("available"):
            advection_lines.append(f"{station}: wind/advection н/д")
            continue
        signals = []
        if info.get("cold_advection_signal"):
            signals.append("cold")
        if info.get("warm_advection_signal"):
            signals.append("warm")
        if info.get("frontal_passage_signal"):
            signals.append("front")
        signal_text = ", ".join(signals) if signals else "neutral"
        advection_lines.append(
            f"{station}: ветер {_fmt_plain(info.get('wind_dir_latest_deg'))}°/{_fmt_plain(info.get('wind_speed_latest_kt'))} kt, "
            f"T1h {_fmt_signed_plain(info.get('temp_trend_1h'))} °C, "
            f"Td3h {_fmt_signed_plain(info.get('dewpoint_trend_3h'))} °C, "
            f"QNH3h {_fmt_signed_plain(info.get('pressure_tendency_3h'))} hPa, {signal_text}"
        )
    return [
        "<b>Spatial + wind/advection candidate</b>",
        (
            "Использован как основной прогноз в этом выпуске. Активен только 12:00-18:00 по Парижу."
            if promoted
            else "Заменяет прежний spatial-кандидат LFPG/LFPO. Не влияет на основной прогноз. Активен только 12:00-18:00 по Парижу."
        ),
        f"Ожидаемый METAR Tmax: <b>{float(forecast.get('expected_tmax_c', 0.0)):.1f} °C</b> ({_fmt_delta(delta)} к production)",
        f"Медиана: {float(forecast.get('median_tmax_c', 0.0)):.1f} °C",
        f"Самая вероятная корзина: <b>{forecast.get('most_likely_integer_c')} °C</b>",
        f"80% интервал: {float((forecast.get('intervals') or {}).get('80', [0.0, 0.0])[0]):.1f}...{float((forecast.get('intervals') or {}).get('80', [0.0, 0.0])[1]):.1f} °C",
        f"P(Tmax >= 25 °C): {float(thresholds.get('ge_25', 0.0)):.1%}",
        f"P(Tmax >= 30 °C): {float(thresholds.get('ge_30', 0.0)):.1%}",
        f"Соседних станций доступно: {int(spatial.get('available_station_count') or 0)}",
        f"Средняя текущая температура соседей: {_fmt_signed_plain(spatial.get('latest_temp_mean_c'))} °C",
        f"Средний максимум соседей: {_fmt_signed_plain(spatial.get('current_max_mean_c'))} °C",
        f"Advection stations: {int(advection.get('available_station_count') or 0)}",
        f"Средний ветер: {_fmt_plain(advection.get('mean_wind_speed_latest_kt'))} kt",
        f"Средний тренд T 1ч/3ч: {_fmt_signed_plain(advection.get('mean_temp_trend_1h'))}/{_fmt_signed_plain(advection.get('mean_temp_trend_3h'))} °C",
        f"Средний тренд Td 3ч: {_fmt_signed_plain(advection.get('mean_dewpoint_trend_3h'))} °C",
        f"Средний тренд QNH 3ч: {_fmt_signed_plain(advection.get('mean_pressure_tendency_3h'))} hPa",
        f"Сигналы cold/warm/front: {'да' if advection.get('any_cold_advection_signal') else 'нет'}/{'да' if advection.get('any_warm_advection_signal') else 'нет'}/{'да' if advection.get('any_frontal_passage_signal') else 'нет'}",
        *(neighbor_lines or ["Соседи: нет данных"]),
        *(advection_lines or ["Wind/advection: нет данных"]),
        "",
    ]


def _fmt_plain(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):.0f}"


def _fmt_signed_plain(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):.1f}"


def _fmt_delta(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):+.1f} °C"


def _fmt_percent(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):.1%}"


def _max_timestamp_string(*values) -> str | None:
    stamps = [pd.Timestamp(value) for value in values if value is not None and not pd.isna(value)]
    if not stamps:
        return None
    return max(stamps).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LFPB daily METAR Tmax distribution.")
    parser.add_argument("--airport", default=AIRPORT)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--auto-refresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refresh-nwp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--spatial-candidate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--promote-spatial-candidate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--report-path", default="data/reports/latest_lfpb_metar_tmax_prediction.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
