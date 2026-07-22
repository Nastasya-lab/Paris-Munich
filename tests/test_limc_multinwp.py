from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from weather_tmax_bot.models.multinwp_tmax import MultiNwpMetarTmaxModel


def _load_script(filename: str, module_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_limc_forecast_modes_cover_full_day() -> None:
    module = _load_script("119_predict_limc_metar_tmax.py", "test_limc_predictor_modes")

    assert module.forecast_mode_for_hour(2) == (6, "early_nwp_residual")
    assert module.forecast_mode_for_hour(6) == (6, "trained_intraday")
    assert module.forecast_mode_for_hour(15) == (15, "trained_intraday")
    assert module.forecast_mode_for_hour(20) == (20, "trained_intraday")
    assert module.forecast_mode_for_hour(22) == (20, "late_clamped_intraday")


def test_limc_message_contains_trigger_and_all_nwp_sources() -> None:
    module = _load_script("119_predict_limc_metar_tmax.py", "test_limc_predictor_message")
    payload = {
        "target_date_local": "2026-07-22",
        "issue_time_local": "2026-07-22T12:00:00+02:00",
        "update_trigger": "new_metar",
        "forecast_mode": "trained_intraday",
        "model_version": "limc_test",
        "forecast": {
            "expected_tmax_c": 31.2,
            "most_likely_integer_c": 31,
            "intervals": {"80": [30.0, 32.0]},
            "probabilities_by_integer_c": {"30": 0.2, "31": 0.6, "32": 0.2},
        },
        "latest_metar_record": {
            "observation_time_utc": "2026-07-22T10:00:00Z",
            "temperature_c": 28.0,
            "current_max_c": 28.0,
            "raw_metar": "METAR LIMC TEST",
        },
        "nwp": {
            "individual_tmax_c": {"icon_d2": 31.0, "icon_eu": 32.0, "arpege": 31.5},
            "blend_tmax_c": 31.4,
            "spread_c": 1.0,
            "degraded": False,
            "available_models": ["icon_d2", "icon_eu", "arpege"],
        },
    }

    text = module.format_forecast_message(payload)

    assert "LIMC Milan Tmax forecast" in text
    assert "METAR LIMC TEST" in text
    assert "ICON-D2" in text
    assert "ICON-EU" in text
    assert "ARPEGE Europe" in text


def test_limc_telegram_uses_requested_chat_and_main_bot(monkeypatch) -> None:
    module = _load_script("120_limc_forecast_job.py", "test_limc_forecast_job")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN_LIMC", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "main-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_LFPB", "paris-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID_LIMC", raising=False)

    module._activate_telegram()

    assert module.os.environ["TELEGRAM_BOT_TOKEN"] == "main-token"
    assert module.os.environ["TELEGRAM_CHAT_ID"] == "-1004371899833"


def test_limc_forecast_job_runs_isolated_paper_trader(monkeypatch) -> None:
    module = _load_script("120_limc_forecast_job.py", "test_limc_forecast_job_trading")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return type("Completed", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module._run_polymarket_paper()

    assert calls[0][0][-3:] == ["--airport", "LIMC", "--notify"]
    assert calls[0][1]["check"] is False


def test_single_provider_wrapper_requires_selected_provider() -> None:
    model = MultiNwpMetarTmaxModel(
        ensemble=object(),
        nwp_weights={"icon_d2": 1.0, "icon_eu": 0.0, "arpege": 0.0},
        nwp_prefixes=["icon_d2", "icon_eu", "arpege"],
        model_version="test",
        minimum_nwp_models=1,
        residual_nwp_prefix="icon_d2",
    )

    assert model.source_status({"icon_d2_tmax_c": 30.0})["usable"] is True
    status = model.source_status({"icon_eu_tmax_c": 30.0, "arpege_tmax_c": 31.0})
    assert status["usable"] is False
    assert status["required_models"] == ["icon_d2"]


def test_early_residual_uses_icon_d2_not_diagnostic_blend() -> None:
    module = _load_script("119_predict_limc_metar_tmax.py", "test_limc_early_residual")
    model = type("Model", (), {"residual_nwp_prefix": "icon_d2"})()
    row = module.residual_feature_row(
        model,
        {
            "icon_d2_tmax_c": 30.0,
            "icon_d2_future_temp_max_c": 29.5,
        },
        {"model_tmax_c": 32.0, "model_future_temp_max_c": 31.5},
    )

    assert row["model_tmax_c"] == 30.0
    assert row["model_future_temp_max_c"] == 29.5
