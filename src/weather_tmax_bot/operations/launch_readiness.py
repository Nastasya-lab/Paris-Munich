from __future__ import annotations

from pathlib import Path

from weather_tmax_bot.evaluation.monitoring import build_monitoring_summary
from weather_tmax_bot.models.registry_health import registry_health
from weather_tmax_bot.operations.pending_truth import pending_truth_status
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def assess_launch_readiness(root: str | Path = ".") -> dict:
    monitoring = build_monitoring_summary(root)
    registry = registry_health(Path(root) / "data/models", Path(root) / "data/models/quantile_mvp.joblib")
    freshness = evaluate_freshness_gate(root=root, fail_on_missing=False, fail_on_stale=True)
    pending = pending_truth_status()
    leakage_rows = monitoring.get("leakage_audit", [])
    leakage_passed = all(int(row.get("violations", 0)) == 0 for row in leakage_rows if row.get("check") != "rows")
    accepted_forecasts = _accepted_forecasts(monitoring.get("operational_forecast_inventory", []))
    blocking = []
    if not registry.get("passed"):
        blocking.append("registry_health_failed")
    if not freshness.get("passed"):
        blocking.append("freshness_gate_failed")
    if not leakage_passed:
        blocking.append("leakage_audit_failed")
    if accepted_forecasts < 1:
        blocking.append("no_accepted_operational_forecast")
    outcome_rows = int(monitoring.get("forecast_monitoring_rows", 0))
    return {
        "ready_for_forward_ops": not blocking,
        "ready_for_outcome_monitoring": outcome_rows > 0,
        "blocking_reasons": blocking,
        "accepted_operational_forecasts": accepted_forecasts,
        "forecast_log_rows": int(monitoring.get("forecast_log_rows", 0)),
        "forecast_monitoring_rows": outcome_rows,
        "pending_truth_rows": int(pending.get("pending_rows", 0)),
        "ready_truth_refresh_dates": pending.get("dates_to_refresh", []),
        "registry_health_passed": bool(registry.get("passed")),
        "freshness_gate_passed": bool(freshness.get("passed")),
        "leakage_audit_passed": leakage_passed,
        "next_action": _next_action(blocking, outcome_rows, pending),
    }


def _accepted_forecasts(inventory: list[dict]) -> int:
    total = 0.0
    for row in inventory:
        total += float(row.get("accepted_rate", 0.0)) * int(row.get("logged_forecasts", 0))
    return int(round(total))


def _next_action(blocking: list[str], outcome_rows: int, pending: dict) -> str:
    if blocking:
        return "resolve_forward_ops_blocking_reasons"
    if outcome_rows <= 0:
        if pending.get("dates_to_refresh"):
            return "run_pending_truth_cron_with_fetch"
        return "continue_forward_logging_until_dwd_truth_is_available"
    return "review_outcome_analysis_and_continue_monitoring"
