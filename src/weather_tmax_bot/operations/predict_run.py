from __future__ import annotations

from datetime import date, datetime

from weather_tmax_bot.bot.forecast_log import log_forecast
from weather_tmax_bot.bot.lineage import build_data_lineage
from weather_tmax_bot.models.predict import predict_best_available
from weather_tmax_bot.operations.acceptance import evaluate_forecast_acceptance
from weather_tmax_bot.operations.quality import assess_forecast_quality
from weather_tmax_bot.operations.refresh import refresh_operational_data


def run_prediction(
    airport: str,
    target_date_local: date,
    issue_time_utc: datetime,
    log: bool = True,
    mode: str = "cli",
) -> dict:
    dist, meta = predict_best_available(airport, target_date_local, issue_time_utc)
    feature_snapshot = meta.get("feature_snapshot", {})
    lineage = build_data_lineage(feature_snapshot)
    quality = assess_forecast_quality(feature_snapshot, meta.get("warnings", []))
    acceptance = evaluate_forecast_acceptance(distribution=dist, forecast_quality=quality)
    feature_snapshot["forecast_quality"] = quality
    feature_snapshot["forecast_acceptance"] = acceptance
    forecast_id = None
    if log:
        forecast_id = log_forecast(
            airport=airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date_local,
            distribution=dist,
            model_version=meta["model_version"],
            feature_snapshot={
                "data_sources_used": ["dwd.10min.air_temperature.01262", "climatology_baseline"],
                **feature_snapshot,
                "mode": mode,
            },
        )
    warnings = [
        *meta.get("warnings", []),
        *([] if forecast_id is None else [f"forecast logged as {forecast_id}"]),
    ]
    return {
        "distribution": dist,
        "metadata": meta,
        "feature_snapshot": feature_snapshot,
        "data_lineage": lineage,
        "forecast_id": forecast_id,
        "warnings": warnings,
        "forecast_quality": quality,
        "forecast_acceptance": acceptance,
    }


def run_prediction_with_optional_refresh(
    airport: str,
    target_date_local: date,
    issue_time_utc: datetime,
    auto_refresh: bool = False,
    refresh_awc: bool = True,
    refresh_nwp: bool = True,
    log: bool = True,
    mode: str = "cli",
) -> dict:
    refresh_summary = None
    if auto_refresh:
        refresh_summary = refresh_operational_data(
            airport=airport,
            target_date_local=target_date_local,
            refresh_awc=refresh_awc,
            refresh_nwp=refresh_nwp,
        )
    result = run_prediction(
        airport=airport,
        target_date_local=target_date_local,
        issue_time_utc=issue_time_utc,
        log=log,
        mode=mode,
    )
    result["refresh_summary"] = refresh_summary
    return result
