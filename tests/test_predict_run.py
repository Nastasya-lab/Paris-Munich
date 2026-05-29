import json
from datetime import date, datetime, timezone

from weather_tmax_bot.operations.predict_run import run_prediction
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.run_report import operational_prediction_payload


def test_run_prediction_with_optional_refresh_without_network():
    result = run_prediction_with_optional_refresh(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        auto_refresh=True,
        refresh_awc=False,
        refresh_nwp=False,
        log=False,
        mode="test",
    )

    assert result["metadata"]["model_version"]
    assert result["refresh_summary"]["airport"] == "EDDM"
    assert abs(result["distribution"].probabilities.sum() - 1.0) < 1e-6
    assert result["forecast_quality"]["status"] in {"ok", "degraded", "invalid"}
    assert "accepted" in result["forecast_acceptance"]


def test_run_prediction_logs_acceptance_metadata(tmp_path, monkeypatch):
    log_path = tmp_path / "forecast_log.jsonl"
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(log_path))

    result = run_prediction(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        log=True,
        mode="test",
    )

    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["forecast_id"] == result["forecast_id"]
    assert "forecast_quality" in record["raw_input_metadata"]
    assert "forecast_acceptance" in record["raw_input_metadata"]
    assert record["raw_input_metadata"]["forecast_acceptance"]["accepted"] == result["forecast_acceptance"]["accepted"]


def test_operational_prediction_payload_contains_run_report_fields():
    result = run_prediction_with_optional_refresh(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        auto_refresh=True,
        refresh_awc=False,
        refresh_nwp=False,
        log=False,
        mode="test",
    )

    payload = operational_prediction_payload(
        airport="EDDM",
        target_date_local=date(2026, 5, 29),
        issue_time_utc=datetime(2026, 5, 28, 20, 30, tzinfo=timezone.utc),
        result=result,
    )

    assert payload["forecast_acceptance"] == result["forecast_acceptance"]
    assert payload["forecast_quality"] == result["forecast_quality"]
    assert payload["refresh_summary"]["airport"] == "EDDM"
