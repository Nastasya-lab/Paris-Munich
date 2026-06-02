import sys

from scripts.railway_entrypoint import build_api_job_command, resolve_job


def test_railway_entrypoint_routes_service_names_to_jobs():
    assert resolve_job("Munich") == "api"
    assert resolve_job("forecast-cron") == "forecast"
    assert resolve_job("metar-cron") == "metar-event"
    assert resolve_job("outcome-cron") == "outcome"
    assert resolve_job("scheduler-health-cron") == "health"
    assert resolve_job("whatever", explicit_job="forecast") == "forecast"


def test_railway_entrypoint_builds_metar_polling_command(monkeypatch):
    monkeypatch.setenv("METAR_POLL_TIMEOUT_SECONDS", "900")
    monkeypatch.setenv("METAR_POLL_INTERVAL_SECONDS", "30")

    command = build_api_job_command("metar-event")

    assert command[:3] == [sys.executable, "scripts/33_call_api_job.py", "metar-event"]
    assert "--poll-timeout-seconds" in command
    assert "900" in command
    assert "--poll-interval-seconds" in command
    assert "30" in command
