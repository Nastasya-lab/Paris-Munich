from weather_tmax_bot.notifications import telegram


def test_telegram_not_configured(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    result = telegram.notify_if_configured("hello")

    assert result["sent"] is False
    assert result["reason"] == "telegram_not_configured"


def test_send_telegram_message_uses_env(monkeypatch):
    calls = {}

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"ok": True}

    def fake_post(url, json, timeout):
        calls["url"] = url
        calls["json"] = json
        calls["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setattr(telegram.requests, "post", fake_post)

    result = telegram.send_telegram_message("hello")

    assert result["sent"] is True
    assert calls["url"].endswith("/bottoken/sendMessage")
    assert calls["json"]["chat_id"] == "123"
    assert calls["json"]["text"] == "hello"


def test_operational_cycle_message_contains_status():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": True,
            "airport": "EDDM",
            "target_date_local": "2026-05-29",
            "issue_time_utc": "2026-05-29T15:00:00Z",
            "model_version": "m1",
            "forecast_id": "f1",
            "forecast_quality": {"status": "ok"},
            "forecast_acceptance": {"blocking_reasons": [], "cautions": ["caution"]},
            "refresh_summary": {"freshness_gate": {"passed": True}},
            "recommendation": "ok",
        }
    )

    assert "ACCEPTED" in text
    assert "EDDM" in text
