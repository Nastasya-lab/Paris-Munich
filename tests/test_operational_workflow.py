import json
from datetime import date, datetime, timezone

from weather_tmax_bot.operations.workflow import run_operational_cycle


def test_operational_cycle_writes_prediction_report_without_network(tmp_path):
    report_path = tmp_path / "prediction.json"
    summary = run_operational_cycle(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        auto_refresh=True,
        refresh_awc=False,
        refresh_nwp=False,
        log=False,
        update_reports=False,
        report_path=report_path,
        forecast_log_path=tmp_path / "forecast_log.jsonl",
        outcome_status_path=tmp_path / "forecast_outcome_status.parquet",
        reports_dir=tmp_path,
        mode="test",
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert summary["forecast_id"] is None
    assert summary["prediction_report_path"] == str(report_path)
    assert "forecast_acceptance" in payload
    assert "report_summary" in summary
    assert summary["report_summary"]["monitoring_report_updated"] is False


def test_operational_cycle_logs_and_refreshes_report_tables(tmp_path, monkeypatch):
    log_path = tmp_path / "forecast_log.jsonl"
    report_path = tmp_path / "prediction.json"
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(log_path))

    summary = run_operational_cycle(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        auto_refresh=False,
        log=True,
        update_reports=False,
        report_path=report_path,
        forecast_log_path=log_path,
        outcome_status_path=tmp_path / "forecast_outcome_status.parquet",
        reports_dir=tmp_path,
        mode="test",
    )

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert summary["forecast_id"] == record["forecast_id"]
    assert record["raw_input_metadata"]["forecast_acceptance"]["accepted"] == summary["forecast_acceptance"]["accepted"]
    assert summary["report_summary"]["forecast_outcome_status_rows"] >= 0
