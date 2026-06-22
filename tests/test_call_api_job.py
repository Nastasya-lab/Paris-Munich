import importlib.util
from pathlib import Path


def _load_job_module():
    path = Path("scripts/33_call_api_job.py")
    spec = importlib.util.spec_from_file_location("call_api_job", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metar_event_polling_stops_when_new_metar_arrives(monkeypatch):
    module = _load_job_module()
    calls = []
    responses = [
        {"status": "no_new_metar", "latest_metar_time_utc": "2026-06-01T07:50:00Z"},
        {"status": "new_metar_forecast", "latest_metar_time_utc": "2026-06-01T08:20:00Z", "notification_sent": False},
    ]

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(*args, **kwargs):
        calls.append((args, kwargs))
        return DummyResponse(responses[len(calls) - 1])

    monkeypatch.setattr(module.requests, "post", fake_post)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    now = {"value": 0.0}

    def fake_monotonic():
        now["value"] += 1.0
        return now["value"]

    monkeypatch.setattr(module.time, "monotonic", fake_monotonic)

    result = module._run_metar_event_with_optional_polling(
        base="https://example.test",
        headers={},
        airport="EDDM",
        target="2026-06-01",
        issue_time="now",
        request_timeout=10,
        poll_timeout_seconds=60,
        poll_interval_seconds=30,
    )

    assert result["status"] == "new_metar_forecast"
    assert len(calls) == 2
    assert result["polling"]["attempt_count"] == 2


def test_compact_job_result_omits_large_forecast_payload():
    module = _load_job_module()

    compact = module._compact_job_result(
        "metar-event",
        {
            "status": "new_metar_forecast",
            "airport": "EDDM",
            "forecast_id": "f1",
            "latest_metar_time_utc": "2026-06-02T16:20:00Z",
            "notification_sent": True,
            "forecast": {"forecast_components": {"very": "large"}},
            "telegram_notification": {"response": {"very": "large"}},
        },
    )

    assert compact["forecast_id"] == "f1"
    assert compact["notification_sent"] is True
    assert "forecast" not in compact
    assert "telegram_notification" not in compact


def test_compact_daily_report_result_keeps_model_summary():
    module = _load_job_module()

    compact = module._compact_job_result(
        "daily-report",
        {
            "status": "ok",
            "airport": "EDDM",
            "target_date_local": "2026-06-02",
            "mode": "preliminary_metar",
            "best_variant": {"forecast_variant": "production_champion"},
            "worst_variant": {"forecast_variant": "production_champion"},
            "telegram_notification": {"sent": True, "response": {"large": "body"}},
        },
    )

    assert compact["best_variant"] == "production_champion"
    assert compact["worst_variant"] == "production_champion"
    assert compact["notification_sent"] is True
    assert "telegram_notification" not in compact


def test_forecast_job_attaches_daily_report_when_available(monkeypatch, capsys):
    module = _load_job_module()
    calls = []

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, params, headers, timeout):
        calls.append((url, params))
        if url.endswith("/operational-cycle"):
            return DummyResponse({"airport": "EDDM", "forecast_quality": {"status": "ok"}})
        return DummyResponse(
            {
                "status": "ok",
                "mode": "preliminary_metar",
                "best_variant": {"forecast_variant": "production_champion"},
                "worst_variant": {"forecast_variant": "production_champion"},
                "telegram_notification": {"sent": True},
            }
        )

    monkeypatch.setenv("MUNICH_API_BASE_URL", "https://example.test")
    monkeypatch.setattr(module.requests, "post", fake_post)

    module.main(
        job="forecast",
        base_url=None,
        airport="EDDM",
        target_date="2026-06-02",
        issue_time="now",
        timeout=10,
        poll_timeout_seconds=0,
        poll_interval_seconds=30,
    )

    out = capsys.readouterr().out
    assert len(calls) == 2
    assert calls[1][0].endswith("/daily-report")
    assert '"daily_report"' in out
    assert "production_champion" in out


def test_metar_event_job_does_not_attach_daily_report_by_default(monkeypatch, capsys):
    module = _load_job_module()
    calls = []

    class DummyResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_post(url, params, headers, timeout):
        calls.append((url, params))
        return DummyResponse(
            {
                "status": "new_metar_forecast",
                "airport": "EDDM",
                "latest_metar_time_utc": "2026-06-02T16:20:00Z",
                "notification_sent": False,
            }
        )

    monkeypatch.setenv("MUNICH_API_BASE_URL", "https://example.test")
    monkeypatch.delenv("WEATHER_TMAX_EMBED_DAILY_REPORT_ON_METAR", raising=False)
    monkeypatch.setattr(module.requests, "post", fake_post)

    module.main(
        job="metar-event",
        base_url=None,
        airport="EDDM",
        target_date="2026-06-02",
        issue_time="now",
        timeout=10,
        poll_timeout_seconds=0,
        poll_interval_seconds=30,
    )

    out = capsys.readouterr().out
    assert len(calls) == 1
    assert calls[0][0].endswith("/metar-event-cycle")
    assert '"daily_report"' not in out


def test_daily_report_not_ready_is_not_logged(monkeypatch):
    module = _load_job_module()
    payload = {"airport": "EDDM"}

    def fake_call(**kwargs):
        return {"status": "not_ready", "reason": "before_earliest_local_hour"}

    monkeypatch.setattr(module, "_call_preliminary_daily_report", fake_call)

    module._attach_daily_report_if_enabled(
        payload,
        base="https://example.test",
        headers={},
        airport="EDDM",
        target="2026-06-02",
        request_timeout=10,
    )

    assert "daily_report" not in payload


def test_legacy_metar_command_delegates_to_multi_airport_job(monkeypatch):
    module = _load_job_module()
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setenv("WEATHER_TMAX_JOB", "metar-event-all-once")
    monkeypatch.delenv("WEATHER_TMAX_MULTI_AIRPORT_CHILD", raising=False)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    delegated = module._delegate_multi_airport_job_if_requested(job="metar-event", issue_time="now")

    assert delegated is True
    assert calls == [([module.sys.executable, "scripts/55_multi_airport_job.py", "metar-event-all-once", "--issue-time", "now"], True)]


def test_child_metar_command_does_not_delegate(monkeypatch):
    module = _load_job_module()

    monkeypatch.setenv("WEATHER_TMAX_JOB", "metar-event-all-once")
    monkeypatch.setenv("WEATHER_TMAX_MULTI_AIRPORT_CHILD", "1")

    assert module._delegate_multi_airport_job_if_requested(job="metar-event", issue_time="now") is False
