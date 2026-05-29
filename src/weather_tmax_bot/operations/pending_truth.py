from __future__ import annotations

from datetime import date

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report
from weather_tmax_bot.evaluation.monitoring import write_monitoring_report
from weather_tmax_bot.operations.truth_refresh import plan_pending_truth_refresh, refresh_pending_truth


def pending_truth_status(as_of_date: date | None = None, min_lag_days: int = 1) -> dict:
    plan = plan_pending_truth_refresh(as_of_date=as_of_date, min_lag_days=min_lag_days)
    return {
        "pending_rows": plan.get("pending_rows", 0),
        "ready_rows": plan.get("ready_rows", 0),
        "dates_to_refresh": plan.get("dates_to_refresh", []),
        "as_of_date": plan.get("as_of_date"),
        "cutoff_date": plan.get("cutoff_date"),
        "min_lag_days": plan.get("min_lag_days"),
        "outcome_status_counts": plan.get("outcome_status_counts", {}),
        "action_required": bool(plan.get("dates_to_refresh")),
        "recommendation": _status_recommendation(plan, fetch=False),
    }


def run_pending_truth_cron(
    fetch: bool = False,
    as_of_date: date | None = None,
    min_lag_days: int = 1,
    update_reports: bool = False,
) -> dict:
    before = pending_truth_status(as_of_date=as_of_date, min_lag_days=min_lag_days)
    refresh_summary = None
    ran_refresh = False
    if fetch and before["action_required"]:
        refresh_summary = refresh_pending_truth(fetch=True, as_of_date=as_of_date, min_lag_days=min_lag_days)
        ran_refresh = True
    after = pending_truth_status(as_of_date=as_of_date, min_lag_days=min_lag_days)
    reports_updated = False
    if update_reports:
        write_monitoring_report()
        write_first_analysis_report()
        reports_updated = True
    return {
        "status": after,
        "status_before": before,
        "refresh_summary": refresh_summary,
        "ran_refresh": ran_refresh,
        "reports_updated": reports_updated,
        "recommendation": _cron_recommendation(before, after, fetch=fetch, ran_refresh=ran_refresh),
    }


def _status_recommendation(plan: dict, fetch: bool) -> str:
    if plan.get("dates_to_refresh"):
        if fetch:
            return "ready_to_refresh"
        return "run_with_fetch_to_score_ready_forecasts"
    if plan.get("pending_rows", 0):
        return "wait_for_truth_lag_or_target_completion"
    return "no_pending_truth"


def _cron_recommendation(before: dict, after: dict, *, fetch: bool, ran_refresh: bool) -> str:
    if before["action_required"] and not fetch:
        return "ready_dates_found_run_again_with_fetch"
    if ran_refresh and after["action_required"]:
        return "refresh_attempted_but_some_ready_dates_remain"
    if ran_refresh:
        return "refresh_completed_review_outcome_analysis"
    if after["pending_rows"]:
        return "pending_forecasts_not_ready_for_truth_refresh"
    return "no_pending_truth"
