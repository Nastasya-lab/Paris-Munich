from datetime import date, datetime, timezone

from weather_tmax_bot.evaluation.smoke import run_operational_smoke


def test_operational_smoke_runs_without_requiring_fresh_sources():
    result = run_operational_smoke(
        airport="EDDM",
        target_date=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        fail_on_stale=False,
    )

    assert result["checks"]["registry_health_passed"]
    assert result["checks"]["probabilities_sum_to_one"]
    assert result["model_version"]
