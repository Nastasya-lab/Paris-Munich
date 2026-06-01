from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report
from weather_tmax_bot.evaluation.monitoring import write_monitoring_report
from weather_tmax_bot.evaluation.operational_monitoring import build_operational_monitoring_tables
from weather_tmax_bot.evaluation.outcomes import build_forecast_outcome_status
from weather_tmax_bot.evaluation.promotion_gate import write_shadow_promotion_gate_report
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.run_report import operational_prediction_payload, write_operational_prediction_report


def run_operational_cycle(
    *,
    airport: str,
    target_date_local: date,
    issue_time_utc: datetime,
    auto_refresh: bool = True,
    refresh_awc: bool = True,
    refresh_nwp: bool = True,
    log: bool = True,
    update_reports: bool = True,
    report_path: str | Path = "data/reports/latest_operational_prediction.json",
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    target_path: str | Path = "data/processed/daily_target.parquet",
    outcome_status_path: str | Path = "data/reports/forecast_outcome_status.parquet",
    reports_dir: str | Path = "data/reports",
    mode: str = "operational_cycle",
) -> dict:
    prediction = run_prediction_with_optional_refresh(
        airport=airport,
        target_date_local=target_date_local,
        issue_time_utc=issue_time_utc,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=log,
        mode=mode,
    )
    prediction_payload = operational_prediction_payload(
        airport=airport,
        target_date_local=target_date_local,
        issue_time_utc=issue_time_utc,
        result=prediction,
    )
    prediction_report_path = write_operational_prediction_report(prediction_payload, report_path)
    report_summary = _update_operational_reports(
        update_reports=update_reports,
        forecast_log_path=forecast_log_path,
        target_path=target_path,
        outcome_status_path=outcome_status_path,
        reports_dir=reports_dir,
    )
    return {
        "airport": airport,
        "target_date_local": target_date_local.isoformat(),
        "issue_time_utc": issue_time_utc.isoformat(),
        "forecast_id": prediction["forecast_id"],
        "model_version": prediction["metadata"]["model_version"],
        "forecast": _forecast_summary(prediction_payload),
        "forecast_quality": prediction["forecast_quality"],
        "forecast_acceptance": prediction["forecast_acceptance"],
        "accepted": bool(prediction["forecast_acceptance"].get("accepted", False)),
        "warnings": prediction["warnings"],
        "refresh_summary": prediction.get("refresh_summary"),
        "prediction_report_path": str(prediction_report_path),
        "report_summary": report_summary,
        "recommendation": _recommendation(prediction["forecast_acceptance"]),
    }


def _forecast_summary(payload: dict) -> dict:
    return {
        "expected_tmax_c": payload["expected_tmax_c"],
        "median_tmax_c": payload["median_tmax_c"],
        "most_likely_integer_c": payload["most_likely_integer_c"],
        "intervals": payload["intervals"],
        "probabilities_by_integer_c": payload["probabilities_by_integer_c"],
        "threshold_probabilities": payload["threshold_probabilities"],
        "data_freshness": payload.get("data_freshness", {}),
        "forecast_components": payload.get("forecast_components", {}),
    }


def _update_operational_reports(
    *,
    update_reports: bool,
    forecast_log_path: str | Path,
    target_path: str | Path,
    outcome_status_path: str | Path,
    reports_dir: str | Path,
) -> dict:
    status = build_forecast_outcome_status(
        forecast_log_path=forecast_log_path,
        target_path=target_path,
        output_path=outcome_status_path,
    )
    tables = build_operational_monitoring_tables(
        forecast_log_path=forecast_log_path,
        outcome_status_path=outcome_status_path,
        output_dir=reports_dir,
    )
    summary = {
        "forecast_outcome_status_rows": len(status),
        "operational_tables": {name: len(table) for name, table in tables.items()},
        "monitoring_report_updated": False,
        "first_analysis_updated": False,
    }
    if update_reports:
        write_monitoring_report()
        write_shadow_promotion_gate_report()
        write_first_analysis_report()
        summary["monitoring_report_updated"] = True
        summary["first_analysis_updated"] = True
    return summary


def _recommendation(acceptance: dict) -> str:
    if acceptance.get("accepted"):
        return "forecast_accepted_for_operational_use"
    reasons = acceptance.get("blocking_reasons", [])
    if reasons:
        return "forecast_rejected_review_acceptance_gate"
    return "forecast_not_accepted_review_quality_metadata"
