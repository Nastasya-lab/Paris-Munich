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


def test_notify_uses_html_formatting(monkeypatch):
    calls = {}

    def fake_send(text, *, parse_mode=None, timeout=15):
        calls["text"] = text
        calls["parse_mode"] = parse_mode
        return {"sent": True}

    monkeypatch.setattr(telegram, "send_telegram_message", fake_send)

    telegram.notify_if_configured("<b>Прогноз</b>")

    assert calls["parse_mode"] == "HTML"


def test_operational_cycle_message_is_human_readable_russian():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": True,
            "airport": "EDDM",
            "target_date_local": "2026-05-29",
            "issue_time_utc": "2026-05-29T15:00:00Z",
            "model_version": "m1",
            "forecast_id": "f1",
            "forecast": {
                "expected_tmax_c": 24.3,
                "median_tmax_c": 24.0,
                "most_likely_integer_c": 24,
                "intervals": {"80": [22.0, 27.0]},
                "probabilities_by_integer_c": {"23": 0.12, "24": 0.42, "25": 0.005},
                "threshold_probabilities": {"ge_20": 0.98, "ge_25": 0.21, "ge_30": 0.01, "le_0": 0.0},
            },
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {
                "blocking_reasons": [],
                "cautions": ["calibration is preliminary"],
            },
            "refresh_summary": {
                "freshness_gate": {
                    "passed": True,
                    "freshness": {"statuses": {"metar": {"state": "fresh", "age_hours": 0.3}}},
                }
            },
            "recommendation": "ok",
        }
    )

    assert "Прогноз готов" in text
    assert "EDDM" in text
    assert "29.05.2026 17:00" in text
    assert "Ожидаемый максимум: <b>24.3 °C</b>" in text
    assert "Самая вероятная корзина: <b>24 °C</b>" in text
    assert "+23 °C: <b>12.0%</b>" in text
    assert "Не ниже +25 °C: 21.0%" in text
    assert "калибровка вероятностей пока предварительная" in text


def test_operational_cycle_message_escapes_dynamic_html():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": False,
            "airport": "<EDDM>",
            "forecast_quality": {"status": "degraded", "reasons": ["<unsafe>"]},
            "forecast_acceptance": {"cautions": []},
        }
    )

    assert "&lt;EDDM&gt;" in text
    assert "&lt;unsafe&gt;" in text
    assert "<unsafe>" not in text


def test_operational_cycle_message_includes_shadow_intraday_comparison():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": True,
            "airport": "EDDM",
            "forecast": {
                "expected_tmax_c": 29.1,
                "median_tmax_c": 29.0,
                "most_likely_integer_c": 29,
                "intervals": {"80": [29.0, 30.0]},
                "probabilities_by_integer_c": {"29": 0.8, "30": 0.2},
                "threshold_probabilities": {"ge_20": 1.0, "ge_25": 1.0, "ge_30": 0.2, "le_0": 0.0},
                "forecast_components": {
                    "shadow_mode": {
                        "intraday_update": {
                            "active": True,
                            "seasonal_profile": "warm",
                            "intraday_blend_weight": 0.95,
                            "late_drop_override_active": True,
                        },
                        "final_model": {
                            "expected_tmax_c": 29.0,
                            "most_likely_integer_c": 29,
                            "intervals": {"80": [29.0, 29.0]},
                            "probabilities_by_integer_c": {"29": 0.96, "30": 0.04},
                            "threshold_probabilities": {"ge_25": 1.0, "ge_30": 0.04},
                        },
                        "comparison_to_champion": {
                            "expected_tmax_delta_c": -0.1,
                            "ge_25_probability_delta": 0.0,
                            "ge_30_probability_delta": -0.16,
                        },
                    }
                },
            },
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {"cautions": []},
        }
    )

    assert "Теневой сценарий: seasonal intraday" in text
    assert "Ожидаемый максимум: <b>29.0 °C</b> (-0.1 °C к основному)" in text
    assert "Не ниже +30 °C: 4.0% (-16.0% к основному)" in text
    assert "Late-drop override: активен" in text


def test_metar_event_message_includes_integer_bin_probability_changes():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.4,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {"ge_20": 1.0, "ge_25": 0.02, "ge_30": 0.0},
            "probabilities_by_integer_c": {"21": 0.2, "22": 0.5, "23": 0.3},
            "forecast_components": {
                "intraday_update": {
                    "last_metar_temp_c": 20.0,
                    "observed_max_so_far_c": 20.0,
                    "drop_from_observed_max_c": 0.0,
                    "peak_passed_probability": 0.2,
                    "intraday_blend_weight": 0.55,
                }
            },
            "model_version": "m1",
            "forecast_id": "f1",
        },
        {
            "previous": {
                "expected_tmax_c": 22.3,
                "most_likely_integer_c": 22,
                "probabilities_by_integer_c": {"21": 0.3, "22": 0.4, "23": 0.3},
            },
            "current": {"most_likely_integer_c": 22},
            "deltas": {"expected_tmax_delta_c": 0.1, "ge_20_delta": 0.0, "ge_25_delta": 0.01, "ge_30_delta": 0.0},
        },
        ["routine_new_metar_update"],
    )

    assert "Распределение по градусам" in text
    assert "+21 °C: <b>20.0%</b> (-10.0 п.п.)" in text
    assert "+22 °C: <b>50.0%</b> (+10.0 п.п.)" in text
    assert "+23 °C: <b>30.0%</b> (+0.0 п.п.)" in text


def test_metar_event_message_says_when_distribution_is_unchanged():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.0,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {},
            "probabilities_by_integer_c": {"22": 1.0},
            "forecast_components": {"intraday_update": {}},
        },
        {
            "previous": {"most_likely_integer_c": 22, "probabilities_by_integer_c": {"22": 1.0}},
            "current": {"most_likely_integer_c": 22},
            "deltas": {},
        },
    )

    assert "распределение не изменилось" in text
    assert "+22 °C: <b>100.0%</b> (+0.0 п.п.)" in text


def test_metar_event_message_includes_shadow_distribution_without_deltas():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.4,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {},
            "probabilities_by_integer_c": {"22": 1.0},
            "forecast_components": {
                "intraday_update": {},
                "shadow_mode": {
                    "intraday_update": {"intraday_blend_weight": 0.1},
                    "final_model": {
                        "expected_tmax_c": 22.8,
                        "probabilities_by_integer_c": {"21": 0.1, "22": 0.5, "23": 0.4},
                        "threshold_probabilities": {"ge_30": 0.0},
                    },
                },
            },
        },
        {"previous": None, "current": {}, "deltas": {}},
    )

    assert "Shadow-\u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439" in text
    assert "\u0420\u0430\u0441\u043f\u0440\u0435\u0434\u0435\u043b\u0435\u043d\u0438\u0435: +21 \u00b0C 10.0%, +22 \u00b0C 50.0%, +23 \u00b0C 40.0%" in text
    assert "\u043f.\u043f." not in text.split("Shadow-\u0441\u0446\u0435\u043d\u0430\u0440\u0438\u0439", 1)[1]


def test_outcome_and_healthcheck_messages_are_russian():
    outcome = telegram.format_outcome_update_message(
        {"status": {"pending_rows": 2, "ready_rows": 0}, "ran_refresh": False}
    )
    health = telegram.format_healthcheck_message(
        {
            "ready_for_forward_ops": True,
            "ready_for_outcome_monitoring": True,
            "accepted_operational_forecasts": 1,
            "pending_truth_rows": 2,
            "next_action": "review_outcome_analysis_and_continue_monitoring",
        }
    )

    assert "Обновление фактических результатов" in outcome
    assert "Новых завершенных суток" in outcome
    assert "Система работает штатно" in health
    assert "продолжать накопление прогнозов" in health
