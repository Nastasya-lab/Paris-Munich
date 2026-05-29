from __future__ import annotations

from datetime import date

from fastapi import FastAPI

from weather_tmax_bot.bot.forecast_log import log_forecast
from weather_tmax_bot.bot.lineage import build_data_lineage, model_info
from weather_tmax_bot.evaluation.first_analysis import build_first_analysis
from weather_tmax_bot.evaluation.monitoring import build_monitoring_summary
from weather_tmax_bot.evaluation.operational_api import operational_monitoring_payload
from weather_tmax_bot.models.registry_health import registry_health
from weather_tmax_bot.models.predict import predict_best_available
from weather_tmax_bot.operations.acceptance import evaluate_forecast_acceptance
from weather_tmax_bot.operations.refresh import refresh_operational_data
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.pending_truth import pending_truth_status, run_pending_truth_cron
from weather_tmax_bot.operations.quality import assess_forecast_quality
from weather_tmax_bot.operations.run_report import operational_prediction_payload
from weather_tmax_bot.operations.workflow import run_operational_cycle
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate
from weather_tmax_bot.utils.time import parse_issue_time

app = FastAPI(title="Weather Tmax Bot")


@app.get("/predict")
def predict(airport: str = "EDDM", target_date: date | None = None, issue_time: str = "now"):
    issue = parse_issue_time(issue_time)
    target = target_date or issue.date()
    dist, meta = predict_best_available(airport, target, issue)
    feature_snapshot = meta.get("feature_snapshot", {})
    quality = assess_forecast_quality(feature_snapshot, meta.get("warnings", []))
    acceptance = evaluate_forecast_acceptance(distribution=dist, forecast_quality=quality)
    feature_snapshot["forecast_quality"] = quality
    feature_snapshot["forecast_acceptance"] = acceptance
    forecast_id = log_forecast(
        airport=airport,
        issue_time_utc=issue,
        target_date_local=target,
        distribution=dist,
        model_version=meta["model_version"],
        feature_snapshot={
            "data_sources_used": ["dwd.10min.air_temperature.01262", "climatology_baseline"],
            **feature_snapshot,
            "mode": "api",
        },
    )
    payload = dist.to_payload()
    payload.update(
        {
            "airport": airport,
            "target_date_local": target.isoformat(),
            "issue_time_utc": issue.isoformat(),
            "timezone": "Europe/Berlin",
            "model_version": meta["model_version"],
            "data_mode": "as_of_knowledge_view",
            "data_lineage": build_data_lineage(feature_snapshot),
            "data_freshness": feature_snapshot.get("freshness", {}),
            "extrapolation": feature_snapshot.get("extrapolation", {}),
            "source_compatibility": feature_snapshot.get("source_compatibility", {}),
            "forecast_quality": quality,
            "forecast_acceptance": acceptance,
            "forecast_id": forecast_id,
            "warnings": meta.get("warnings", []),
        }
    )
    return payload


@app.post("/operational-cycle")
def operational_cycle(
    airport: str = "EDDM",
    target_date: date | None = None,
    issue_time: str = "now",
    auto_refresh: bool = True,
    refresh_awc: bool = True,
    refresh_nwp: bool = True,
    log: bool = True,
    update_reports: bool = True,
):
    issue = parse_issue_time(issue_time)
    target = target_date or issue.date()
    return run_operational_cycle(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=log,
        update_reports=update_reports,
        mode="api_operational_cycle",
    )


@app.post("/predict-operational")
def predict_operational(
    airport: str = "EDDM",
    target_date: date | None = None,
    issue_time: str = "now",
    auto_refresh: bool = True,
    refresh_awc: bool = True,
    refresh_nwp: bool = True,
    log: bool = True,
):
    issue = parse_issue_time(issue_time)
    target = target_date or issue.date()
    result = run_prediction_with_optional_refresh(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=log,
        mode="api_operational",
    )
    payload = operational_prediction_payload(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        result=result,
    )
    payload["timezone"] = "Europe/Berlin"
    payload["data_mode"] = "as_of_knowledge_view"
    return payload


@app.get("/health")
def health():
    return {"status": "ok", "service": "weather_tmax_bot"}


@app.get("/model-info")
def get_model_info():
    return model_info()


@app.get("/registry-health")
def get_registry_health():
    return registry_health()


@app.get("/data-freshness-health")
def data_freshness_health(fail_on_missing: bool = False, fail_on_stale: bool = True):
    return evaluate_freshness_gate(fail_on_missing=fail_on_missing, fail_on_stale=fail_on_stale)


@app.get("/monitoring-summary")
def monitoring_summary():
    return build_monitoring_summary()


@app.get("/operational-monitoring")
def operational_monitoring():
    return operational_monitoring_payload()


@app.get("/first-analysis")
def first_analysis():
    return build_first_analysis()


@app.post("/prepare-operational-run")
def prepare_operational_run(
    airport: str = "EDDM",
    target_date: date | None = None,
    skip_awc: bool = False,
    skip_nwp: bool = False,
):
    return refresh_operational_data(
        airport=airport,
        target_date_local=target_date,
        refresh_awc=not skip_awc,
        refresh_nwp=not skip_nwp,
    )


@app.get("/pending-truth")
def get_pending_truth(as_of_date: date | None = None, min_lag_days: int = 1):
    return pending_truth_status(as_of_date=as_of_date, min_lag_days=min_lag_days)


@app.post("/pending-truth-cron")
def post_pending_truth_cron(
    fetch: bool = False,
    as_of_date: date | None = None,
    min_lag_days: int = 1,
    update_reports: bool = True,
):
    return run_pending_truth_cron(
        fetch=fetch,
        as_of_date=as_of_date,
        min_lag_days=min_lag_days,
        update_reports=update_reports,
    )
