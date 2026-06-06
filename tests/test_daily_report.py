import json
from datetime import date

import pandas as pd

from weather_tmax_bot.notifications.telegram import format_daily_model_report_message
from weather_tmax_bot.operations.daily_report import build_daily_model_report, run_daily_model_report


def test_preliminary_daily_report_scores_variants_from_forecast_log(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    record = _forecast_record(
        forecast_id="f1",
        expected_probs={"25": 0.8, "26": 0.2},
        variants={
            "base_prior": {"distribution": {"probabilities_by_integer_c": {"26": 0.8, "27": 0.2}}},
            "shadow_phase_arbitrated": {"distribution": {"probabilities_by_integer_c": {"25": 0.95, "26": 0.05}}},
        },
        observed_max=25.0,
    )
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = build_daily_model_report(
        airport="EDDM",
        target_date_local=date(2026, 6, 2),
        mode="preliminary_metar",
        forecast_log_path=log_path,
    )

    assert report["status"] == "ok"
    assert report["actual_tmax_c"] == 25.0
    assert report["best_variant"]["forecast_variant"] == "shadow_phase_arbitrated"
    assert report["worst_variant"]["forecast_variant"] == "production_champion"
    assert report["hourly_comparison"][0]["best_variant"] == "shadow_phase_arbitrated"


def test_final_daily_report_scores_dwd_variant_monitoring(tmp_path):
    variant_path = tmp_path / "forecast_variant_monitoring.parquet"
    pd.DataFrame(
        [
            _scored_row("f1", "production_champion", 25.0, 25.3, 0.7),
            _scored_row("f1", "shadow_phase_arbitrated", 25.0, 25.1, 0.9),
            _scored_row("f1", "base_prior", 25.0, 27.0, 0.05),
        ]
    ).to_parquet(variant_path, index=False)

    report = build_daily_model_report(
        airport="EDDM",
        target_date_local=date(2026, 6, 2),
        mode="dwd_final",
        variant_monitoring_path=variant_path,
    )

    assert report["status"] == "ok"
    assert report["truth_source"] == "DWD 10-minute truth"
    assert report["best_variant"]["forecast_variant"] == "shadow_phase_arbitrated"


def test_final_daily_report_waits_for_dwd_scored_rows(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    log_path.write_text(json.dumps(_forecast_record("f1", {"25": 1.0}, {}, 25.0)) + "\n", encoding="utf-8")

    report = build_daily_model_report(
        airport="EDDM",
        target_date_local=date(2026, 6, 2),
        mode="dwd_final",
        forecast_log_path=log_path,
        variant_monitoring_path=tmp_path / "missing.parquet",
    )

    assert report["status"] == "no_data"
    assert report["mode"] == "dwd_final"
    assert report["reason"] == "variant_monitoring_missing"


def test_daily_report_notify_dedupes_sent_messages(tmp_path, monkeypatch):
    log_path = tmp_path / "forecast_log.jsonl"
    log_path.write_text(json.dumps(_forecast_record("f1", {"25": 1.0}, {}, 25.0)) + "\n", encoding="utf-8")
    sent = []

    def fake_notify(text):
        sent.append(text)
        return {"sent": True}

    monkeypatch.setattr("weather_tmax_bot.operations.daily_report.notify_if_configured", fake_notify)

    first = run_daily_model_report(
        target_date_local=date(2026, 6, 2),
        force=False,
        forecast_log_path=log_path,
        sent_registry_path=tmp_path / "sent.json",
        output_path=tmp_path / "report.json",
    )
    second = run_daily_model_report(
        target_date_local=date(2026, 6, 2),
        force=False,
        forecast_log_path=log_path,
        sent_registry_path=tmp_path / "sent.json",
        output_path=tmp_path / "report.json",
    )

    assert first["telegram_notification"]["sent"] is True
    assert second["telegram_notification"]["sent"] is False
    assert len(sent) == 1


def test_daily_report_message_is_human_readable():
    text = format_daily_model_report_message(
        {
            "airport": "EDDM",
            "target_date_local": "2026-06-02",
            "mode": "preliminary_metar",
            "actual_tmax_c": 25.0,
            "truth_source": "operational METAR max",
            "analysis": ["Лучше всего выглядел shadow_phase_arbitrated."],
            "summary_by_variant": [
                {
                    "forecast_variant": "shadow_phase_arbitrated",
                    "mae_expected": 0.2,
                    "bias_expected": 0.1,
                    "mean_probability_actual_integer_bin": 0.8,
                    "coverage_ratio": 1.0,
                }
            ],
            "best_variant": {"forecast_variant": "shadow_phase_arbitrated"},
            "worst_variant": {"forecast_variant": "production_champion"},
            "hourly_comparison": [
                {
                    "local_hour": 18,
                    "best_variant": "shadow_phase_arbitrated",
                    "variants": [
                        {"forecast_variant": "shadow_phase_arbitrated", "mae_expected": 0.2},
                        {"forecast_variant": "production_champion", "mae_expected": 0.5},
                    ],
                }
            ],
        }
    )

    assert "Вечерний предварительный разбор моделей" in text
    assert "Предварительный максимум по METAR" in text
    assert "shadow_phase_arbitrated" in text
    assert "production_champion" in text
    assert "18:00" in text
    assert "после прихода DWD truth" in text


def _forecast_record(forecast_id, expected_probs, variants, observed_max):
    return {
        "forecast_id": forecast_id,
        "airport": "EDDM",
        "issue_time_utc": "2026-06-02T16:00:00+00:00",
        "target_date_local": "2026-06-02",
        "model_version": "m1",
        "probability_distribution": expected_probs,
        "raw_input_metadata": {
            "forecast_variants": variants,
            "forecast_components": {"intraday_update": {"observed_max_so_far_c": observed_max}},
        },
    }


def _scored_row(forecast_id, variant, actual, expected, probability_actual):
    return {
        "forecast_id": forecast_id,
        "airport": "EDDM",
        "target_date_local": "2026-06-02",
        "issue_time_utc": "2026-06-02T16:00:00+00:00",
        "forecast_variant": variant,
        "actual_tmax_c": actual,
        "expected_tmax_c": expected,
        "error_expected_c": expected - actual,
        "nll": 0.1 if probability_actual > 0.5 else 3.0,
        "crps": abs(expected - actual) / 10,
        "probability_actual_integer_bin": probability_actual,
        "probability_above_actual_integer_bin": 0.1,
    }
