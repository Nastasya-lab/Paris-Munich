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

    telegram.notify_if_configured("<b>Forecast</b>")

    assert calls["parse_mode"] == "HTML"


def test_operational_cycle_message_is_human_readable():
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
            "forecast_acceptance": {"blocking_reasons": [], "cautions": ["calibration is preliminary"]},
            "refresh_summary": {
                "freshness_gate": {
                    "passed": True,
                    "freshness": {"statuses": {"metar": {"state": "fresh", "age_hours": 0.3}}},
                }
            },
            "recommendation": "ok",
        }
    )

    assert "EDDM" in text
    assert "29.05.2026 17:00" in text
    assert "24.3" in text
    assert "+23" in text
    assert "21.0%" in text
    assert "calibration" not in text


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


def test_operational_cycle_message_hides_challenger_and_keeps_growth_potential():
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
                "forecast_components": _phase_and_growth_components(),
            },
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {"cautions": []},
        }
    )

    assert "phase-arbitrated" not in text
    assert "safe-blend" not in text
    assert "+3" in text
    assert "ML shadow: remaining upside" not in text
    assert "Shadow-" not in text
    assert "blended shadow" not in text


def test_operational_cycle_message_includes_spatial_wind_advection_shadow():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": True,
            "airport": "EDDM",
            "forecast": {
                "expected_tmax_c": 29.1,
                "median_tmax_c": 29.0,
                "most_likely_integer_c": 29,
                "intervals": {"80": [28.0, 31.0]},
                "probabilities_by_integer_c": {"28": 0.2, "29": 0.5, "30": 0.3},
                "threshold_probabilities": {"ge_20": 1.0, "ge_25": 1.0, "ge_30": 0.3, "le_0": 0.0},
                "forecast_components": {
                    "spatial_wind_advection_candidate": {
                        "enabled": True,
                        "active": True,
                        "forecast": {
                            "expected_tmax_c": 29.4,
                            "most_likely_integer_c": 29,
                            "probabilities_by_integer_c": {"28": 0.1, "29": 0.6, "30": 0.3},
                            "threshold_probabilities": {"ge_25": 1.0, "ge_30": 0.3},
                        },
                        "comparison_to_champion": {"expected_tmax_delta_c": 0.3},
                        "wind_advection_features": {
                            "available_station_count": 4,
                            "mean_temp_trend_1h": 0.25,
                            "any_frontal_passage_signal": False,
                        },
                    }
                },
            },
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {"cautions": []},
        }
    )

    assert "EDDM spatial + wind/advection shadow" in text
    assert "Diagnostic only" in text
    assert "Expected METAR Tmax: <b>29.4 C</b>" in text
    assert "P(Tmax >= 30 C): 30.0%" in text


def test_operational_cycle_message_includes_unimodal_shadow():
    text = telegram.format_operational_cycle_message(
        {
            "accepted": True,
            "airport": "EDDM",
            "forecast": {
                "expected_tmax_c": 29.1,
                "median_tmax_c": 29.0,
                "most_likely_integer_c": 29,
                "intervals": {"80": [28.0, 31.0]},
                "probabilities_by_integer_c": {"28": 0.2, "29": 0.5, "30": 0.3},
                "threshold_probabilities": {"ge_20": 1.0, "ge_25": 1.0, "ge_30": 0.3, "le_0": 0.0},
                "forecast_components": {
                    "unimodal_shadow_candidate": {
                        "forecast": {
                            "expected_tmax_c": 29.2,
                            "most_likely_integer_c": 29,
                            "probabilities_by_integer_c": {"28": 0.2, "29": 0.5, "30": 0.3},
                            "threshold_probabilities": {"ge_25": 1.0, "ge_30": 0.3},
                        },
                        "comparison_to_champion": {"expected_tmax_delta_c": 0.1},
                        "metadata": {"shadow_unimodal_violation_count": 0},
                    }
                },
            },
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {"cautions": []},
        }
    )

    assert "EDDM unimodal PMF shadow" in text
    assert "Diagnostic only" in text
    assert "Expected METAR Tmax: <b>29.2 C</b>" in text
    assert "Shape violations: 0" in text


def test_metar_event_message_includes_current_integer_bin_probabilities_without_bin_deltas():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.4,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {"ge_20": 1.0, "ge_25": 0.02, "ge_30": 0.0},
            "probabilities_by_integer_c": {"21": 0.2, "22": 0.5, "23": 0.3},
            "forecast_components": {"intraday_update": {"last_metar_temp_c": 20.0, "observed_max_so_far_c": 20.0}},
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

    assert "+21" in text
    assert "20.0%" in text
    assert "+22" in text
    assert "50.0%" in text
    assert "+10.0 п.п." not in text
    assert "-10.0 п.п." not in text


def test_metar_event_message_shows_current_distribution_without_unchanged_notice():
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

    assert "+22" in text
    assert "100.0%" in text
    assert "+0.0 п.п." not in text


def test_metar_event_message_hides_challenger_and_keeps_growth_potential():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.4,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {},
            "probabilities_by_integer_c": {"22": 1.0},
            "forecast_components": {"intraday_update": {}, **_phase_and_growth_components()},
        },
        {"previous": None, "current": {}, "deltas": {}},
    )

    assert "phase-arbitrated" not in text
    assert "safe-blend" not in text
    assert "+1" in text
    assert "+2" in text
    assert "+3" in text
    assert "ML shadow: remaining upside" not in text
    assert "Shadow-" not in text
    assert "blended shadow" not in text


def test_metar_event_message_includes_unimodal_model_change():
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
                "unimodal_shadow_candidate": {
                    "forecast": {
                        "expected_tmax_c": 22.8,
                        "most_likely_integer_c": 23,
                        "probabilities_by_integer_c": {"22": 0.3, "23": 0.7},
                        "threshold_probabilities": {"ge_25": 0.0, "ge_30": 0.0},
                    },
                    "comparison_to_champion": {"expected_tmax_delta_c": 0.4},
                    "metadata": {"shadow_unimodal_violation_count": 0},
                },
            },
        },
        {
            "previous": {"most_likely_integer_c": 22, "probabilities_by_integer_c": {"22": 1.0}},
            "current": {"most_likely_integer_c": 22},
            "deltas": {},
            "variants": {
                "shadow_unimodal_pmf": {
                    "has_previous": True,
                    "current": {"expected_tmax_c": 22.8, "most_likely_integer_c": 23},
                    "previous": {"expected_tmax_c": 22.1, "most_likely_integer_c": 22},
                    "deltas": {"expected_tmax_delta_c": 0.7},
                }
            },
        },
    )

    assert "EDDM unimodal PMF shadow" in text
    assert "Изменение с прошлого METAR" in text
    assert "Expected Tmax: +0.7" in text
    assert "+22 °C -> +23 °C" in text


def test_metar_event_message_includes_source_compatibility_audit():
    text = telegram.format_metar_event_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-01",
            "issue_time_utc": "2026-06-01T10:55:00Z",
            "expected_tmax_c": 22.4,
            "most_likely_integer_c": 22,
            "threshold_probabilities": {},
            "probabilities_by_integer_c": {"22": 1.0},
            "forecast_components": {"intraday_update": {}},
            "source_compatibility": {
                "metar": {"status": "known_compatible", "runtime_source_id": "awc.metar.live.EDDM"},
                "taf": {"status": "exact_match", "runtime_source_id": "iem.taf.archive.EDDM"},
                "nwp": {"status": "missing", "runtime_source_id": None},
            },
        },
        {"previous": None, "current": {}, "deltas": {}},
    )

    assert "awc.metar.live.EDDM" in text
    assert "iem.taf.archive.EDDM" in text


def test_daily_model_report_message_includes_hourly_comparison():
    text = telegram.format_daily_model_report_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-02",
            "mode": "preliminary_metar",
            "actual_tmax_c": 25.0,
            "truth_source": "operational METAR max",
            "summary_by_variant": [
                {
                    "forecast_variant": "production_champion",
                    "mae_expected": 0.5,
                    "bias_expected": 0.4,
                    "mean_probability_actual_integer_bin": 0.5,
                    "coverage_ratio": 1.0,
                },
            ],
            "best_variant": {"forecast_variant": "production_champion"},
            "worst_variant": {"forecast_variant": "production_champion"},
            "hourly_comparison": [
                {
                    "local_hour": 18,
                    "best_variant": "production_champion",
                    "variants": [
                        {"forecast_variant": "production_champion", "mae_expected": 0.5},
                    ],
                }
            ],
        }
    )

    assert "EDDM" in text
    assert "production_champion" in text
    assert "shadow_phase_arbitrated" not in text
    assert "18:00" not in text


def test_outcome_and_healthcheck_messages_are_not_empty():
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

    assert "2" in outcome
    assert outcome
    assert health


def _phase_and_growth_components():
    return {
        "phase_arbitrated_shadow_mode": {
            "details": {
                "selected_variant": "shadow_safe_blend",
                "selection_reason": "midday_safe_blend_best_ablation",
            },
            "final_model": {
                "expected_tmax_c": 22.8,
                "probabilities_by_integer_c": {"22": 0.4, "23": 0.4, "24": 0.2},
            },
            "comparison_to_champion": {"expected_tmax_delta_c": 0.4},
        },
        "ml_shadow_mode": {
            "details": {
                "active": True,
                "probability_peak_already_passed": 0.4,
                "probability_upside_ge_1c": 0.5,
                "probability_upside_ge_2c": 0.2,
                "probability_upside_ge_3c": 0.1,
            }
        },
    }
