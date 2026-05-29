from __future__ import annotations

from weather_tmax_bot.evaluation.monitoring import build_monitoring_summary
from weather_tmax_bot.models.registry_health import registry_health
from weather_tmax_bot.operations.pending_truth import pending_truth_status
from weather_tmax_bot.temporal.freshness import assess_archive_freshness
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def format_status() -> str:
    health = registry_health()
    monitoring = build_monitoring_summary()
    active = monitoring.get("active_model", {})
    inventory = monitoring.get("operational_forecast_inventory", [])
    inventory_rows = sum(int(row.get("logged_forecasts", 0)) for row in inventory)
    accepted = sum(float(row.get("accepted_rate", 0.0)) * int(row.get("logged_forecasts", 0)) for row in inventory)
    rejected = sum(float(row.get("rejected_rate", 0.0)) * int(row.get("logged_forecasts", 0)) for row in inventory)
    unknown_acceptance = sum(float(row.get("unknown_acceptance_rate", 0.0)) * int(row.get("logged_forecasts", 0)) for row in inventory)
    freshness = assess_archive_freshness()
    freshness_gate = evaluate_freshness_gate(fail_on_missing=False, fail_on_stale=True)
    pending_truth = pending_truth_status()
    return "\n".join(
        [
            "Weather Tmax Bot status",
            "",
            f"Registry health: {'passed' if health['passed'] else 'failed'}",
            f"Active model: {active.get('model_version')}",
            f"Active calibrator: {active.get('calibrator_version')}",
            f"Training rows: {monitoring.get('training_rows')}",
            f"Daily target rows: {monitoring.get('daily_target_rows')}",
            f"Forecast log rows: {monitoring.get('forecast_log_rows')}",
            f"Forecast inventory rows: {inventory_rows}",
            f"Forecast accepted/rejected/unknown: {accepted:.0f}/{rejected:.0f}/{unknown_acceptance:.0f}",
            f"Forecast monitoring rows with outcomes: {monitoring.get('forecast_monitoring_rows')}",
            f"Pending truth rows: {pending_truth['pending_rows']}",
            f"Ready truth refresh dates: {len(pending_truth['dates_to_refresh'])}",
            f"NWP archive rows: {monitoring.get('nwp_archive_rows')}",
            f"AWC live METAR rows: {monitoring.get('awc_metar_live_rows')}",
            f"AWC live TAF rows: {monitoring.get('awc_taf_live_rows')}",
            "",
            "Data freshness:",
            f"METAR: {freshness['statuses']['metar']['state']}",
            f"TAF: {freshness['statuses']['taf']['state']}",
            f"NWP: {freshness['statuses']['nwp']['state']}",
            f"Freshness gate: {'passed' if freshness_gate['passed'] else 'failed'}",
        ]
    )
