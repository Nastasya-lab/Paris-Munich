from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.data.previous_runs import build_previous_day1_snapshots
from weather_tmax_bot.models.multinwp_tmax import blend_nwp_features


def _predictor_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "115_predict_lemd_metar_tmax.py"
    spec = importlib.util.spec_from_file_location("test_lemd_predictor", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _forecast_job_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "116_lemd_forecast_job.py"
    spec = importlib.util.spec_from_file_location("test_lemd_forecast_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blend_nwp_features_renormalizes_missing_provider() -> None:
    row = {
        "icon_eu_tmax_c": 30.0,
        "icon_eu_future_temp_max_c": 29.0,
        "ecmwf_tmax_c": 32.0,
        "ecmwf_future_temp_max_c": 31.0,
        "gfs_tmax_c": np.nan,
        "arpege_tmax_c": np.nan,
    }
    result = blend_nwp_features(
        row,
        {"icon_eu": 0.25, "ecmwf": 0.25, "gfs": 0.25, "arpege": 0.25},
        prefixes=["icon_eu", "ecmwf", "gfs", "arpege"],
    )

    assert result["nwp_available_model_count"] == 2
    assert result["model_tmax_c"] == 31.0
    assert result["nwp_tmax_spread_c"] == 2.0


def test_previous_day1_snapshot_is_known_by_local_day_start() -> None:
    times = pd.date_range("2026-07-10T22:00:00Z", periods=24, freq="h")
    hourly = pd.DataFrame(
        {
            "valid_time_utc": times,
            "model_name": "icon_eu",
            "temperature_2m": np.linspace(20, 35, len(times)),
        }
    )

    snapshots = build_previous_day1_snapshots(
        hourly,
        timezone_name="Europe/Madrid",
        issue_hours=[6, 12, 20],
        prefix="icon_eu",
    )

    assert len(snapshots) == 3
    assert snapshots["target_date_local"].eq("2026-07-11").all()
    assert pd.to_datetime(snapshots["nwp_knowledge_time_utc"], utc=True).max() <= pd.to_datetime(
        snapshots["issue_time_utc"], utc=True
    ).min()


def test_lemd_message_identifies_trigger_and_nwp_sources() -> None:
    module = _predictor_module()
    payload = {
        "target_date_local": "2026-07-11",
        "issue_time_local": "2026-07-11T12:00:00+02:00",
        "update_trigger": "new_metar",
        "model_version": "lemd_test",
        "forecast": {
            "expected_tmax_c": 36.2,
            "most_likely_integer_c": 36,
            "intervals": {"80": [35.0, 37.0]},
            "probabilities_by_integer_c": {"35": 0.2, "36": 0.6, "37": 0.2},
        },
        "latest_metar_record": {
            "observation_time_utc": "2026-07-11T10:00:00Z",
            "temperature_c": 31.0,
            "current_max_c": 31.0,
            "raw_metar": "METAR LEMD TEST",
        },
        "nwp": {
            "individual_tmax_c": {"icon_eu": 36.0, "ecmwf": 37.0, "gfs": 36.5, "arpege": 36.0},
            "blend_tmax_c": 36.4,
            "spread_c": 1.0,
            "degraded": False,
            "available_models": ["icon_eu", "ecmwf", "gfs", "arpege"],
        },
    }

    text = module.format_forecast_message(payload)

    assert "LEMD Madrid Tmax forecast" in text
    assert "Триггер: новый METAR" in text
    assert "ICON-EU" in text
    assert "ARPEGE Europe" in text


def test_lemd_supported_window_uses_trained_hours() -> None:
    module = _predictor_module()

    assert 6 in module.SUPPORTED_LOCAL_HOURS
    assert 20 in module.SUPPORTED_LOCAL_HOURS
    assert 21 not in module.SUPPORTED_LOCAL_HOURS


def test_lemd_telegram_prefers_main_bot_over_paris_bot(monkeypatch) -> None:
    module = _forecast_job_module()
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN_LEMD", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "main-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_LFPB", "paris-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID_LEMD", raising=False)

    module._activate_telegram()

    assert module.os.environ["TELEGRAM_BOT_TOKEN"] == "main-token"
    assert module.os.environ["TELEGRAM_CHAT_ID"] == "-1004409683948"
