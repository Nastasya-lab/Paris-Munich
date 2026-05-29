from __future__ import annotations

from datetime import date, datetime, timezone

from weather_tmax_bot.evaluation.first_analysis import build_first_analysis
from weather_tmax_bot.models.predict import predict_best_available
from weather_tmax_bot.models.registry_health import registry_health
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def run_operational_smoke(
    airport: str = "EDDM",
    target_date: date | None = None,
    issue_time_utc: datetime | None = None,
    fail_on_stale: bool = False,
) -> dict:
    issue = issue_time_utc or datetime.now(timezone.utc)
    target = target_date or issue.date()
    health = registry_health()
    freshness = evaluate_freshness_gate(issue_time_utc=issue, fail_on_missing=False, fail_on_stale=fail_on_stale)
    dist, meta = predict_best_available(airport, target, issue)
    probability_sum = float(dist.probabilities.sum())
    analysis = build_first_analysis()
    checks = {
        "registry_health_passed": bool(health["passed"]),
        "freshness_gate_passed": bool(freshness["passed"]),
        "probabilities_sum_to_one": abs(probability_sum - 1.0) < 1e-6,
        "prediction_has_bins": len(dist.bins_c) > 0,
        "production_predict_ready": bool(analysis.get("readiness", {}).get("production_predict_ready")),
        "leakage_audit_passed": bool(analysis.get("readiness", {}).get("leakage_audit_passed")),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "airport": airport,
        "target_date_local": target.isoformat(),
        "issue_time_utc": issue.isoformat(),
        "model_version": meta.get("model_version"),
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "probability_sum": probability_sum,
        "warnings": meta.get("warnings", []),
        "freshness": freshness,
        "first_analysis_readiness": analysis.get("readiness", {}),
    }
