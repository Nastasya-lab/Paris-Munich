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
