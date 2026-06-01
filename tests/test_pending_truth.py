from datetime import date

import weather_tmax_bot.operations.pending_truth as pending_truth_module
from weather_tmax_bot.operations.pending_truth import pending_truth_status, run_pending_truth_cron


def test_pending_truth_status_returns_action_shape():
    status = pending_truth_status(as_of_date=date(2026, 5, 29))

    assert "pending_rows" in status
    assert "dates_to_refresh" in status
    assert "action_required" in status
    assert "recommendation" in status
    assert "outcome_status_counts" in status


def test_pending_truth_cron_without_fetch_returns_status():
    result = run_pending_truth_cron(fetch=False, as_of_date=date(2026, 5, 29))

    assert "status" in result
    assert result["refresh_summary"] is None
    assert result["ran_refresh"] is False
    assert result["reports_updated"] is False
    assert "recommendation" in result


def test_pending_truth_cron_can_update_reports(monkeypatch):
    calls = []

    monkeypatch.setattr(pending_truth_module, "write_monitoring_report", lambda: calls.append("monitoring"))
    monkeypatch.setattr(pending_truth_module, "write_first_analysis_report", lambda: calls.append("first_analysis"))
    monkeypatch.setattr(pending_truth_module, "write_shadow_promotion_gate_report", lambda: calls.append("promotion_gate"))

    result = run_pending_truth_cron(fetch=False, as_of_date=date(2026, 5, 29), update_reports=True)

    assert result["reports_updated"] is True
    assert calls == ["promotion_gate", "monitoring", "first_analysis"]
