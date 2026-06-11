import json
from datetime import date, datetime, timezone

import numpy as np

from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.operations.predict_run import run_prediction
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.run_report import operational_prediction_payload
from weather_tmax_bot.models import predict as predict_module


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
    assert "forecast_variants" in record["raw_input_metadata"]
    assert "production_champion" in record["raw_input_metadata"]["forecast_variants"]
    assert set(record["raw_input_metadata"]["forecast_variants"]) == {
        "production_champion",
    }
    assert "component_variants" in record["raw_input_metadata"]
    assert "shadow_seasonal_intraday" in record["raw_input_metadata"]["component_variants"]
    assert "shadow_safe_blend" in record["raw_input_metadata"]["component_variants"]
    assert "growth_potential" in record["raw_input_metadata"]
    assert "ml_shadow_mode" in record["raw_input_metadata"]["forecast_components"]
    assert "blended_shadow_mode" in record["raw_input_metadata"]["forecast_components"]
    assert "phase_arbitrated_shadow_mode" not in record["raw_input_metadata"]["forecast_components"]
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


def test_intraday_ml_shadow_load_failure_degrades_without_crashing(tmp_path, monkeypatch):
    model_path = tmp_path / "intraday_ml.joblib"
    model_path.write_bytes(b"placeholder")

    def fail_load(path):
        raise ModuleNotFoundError("No module named '_loss'")

    monkeypatch.setattr(predict_module.joblib, "load", fail_load)

    dist, details = predict_module._predict_intraday_ml_shadow({}, model_path=model_path)

    assert dist is None
    assert details["active"] is False
    assert "intraday_ml_prediction_unavailable" in details["reason"]


def test_late_day_ml_component_is_promoted_to_production(monkeypatch):
    promoted = TmaxDistribution(np.array([17, 18]), np.array([0.1, 0.9]))

    def fake_ml_shadow(feature_row):
        return promoted, {
            "active": True,
            "calibration_status": "contextual_out_of_fold_survival_calibrated",
            "probability_peak_already_passed": 0.9,
            "probability_upside_ge_1c": 0.1,
            "probability_upside_ge_2c": 0.0,
            "probability_upside_ge_3c": 0.0,
        }

    monkeypatch.setattr(predict_module, "_predict_intraday_ml_shadow", fake_ml_shadow)

    result = run_prediction(
        airport="EDDM",
        target_date_local=date(2026, 6, 11),
        issue_time_utc=datetime(2026, 6, 11, 15, 30, tzinfo=timezone.utc),
        log=False,
        mode="test",
    )

    metadata = result["metadata"]["feature_snapshot"]
    promotion = metadata["forecast_components"]["late_day_promotion"]
    assert promotion["active"] is True
    assert promotion["selected_variant"] == "shadow_intraday_ml"
    assert result["distribution"].expected_tmax_c == promoted.expected_tmax_c
    assert set(metadata["forecast_variants"]) == {"production_champion"}
