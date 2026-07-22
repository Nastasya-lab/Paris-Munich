from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.models.multinwp_tmax import RCSS_NWP_MODELS
from weather_tmax_bot.models.regression_tmax import RegressionResidualTmaxModel


ROOT = Path(__file__).resolve().parents[1]


def _load_script(filename: str, module_name: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_rcss_nwp_contract_and_trained_artifact() -> None:
    assert RCSS_NWP_MODELS == {
        "jma_msm": "jma_msm",
        "jma_gsm": "jma_gsm",
        "icon_global": "icon_global",
        "ecmwf_ifs025": "ecmwf",
    }
    model = joblib.load(ROOT / "data/models/rcss_metar_tmax_multinwp_d1_v1.joblib")
    assert model.model_version == "rcss_metar_tmax_multinwp_d1_v1"
    assert model.residual_nwp_prefix == "jma_gsm"
    assert model.shape_strategy == "temperature_unimodal"
    assert model.source_status({"jma_gsm_tmax_c": 31.0})["usable"] is False
    assert model.source_status(
        {"jma_gsm_tmax_c": 31.0, "ecmwf_tmax_c": 31.0}
    )["usable"] is True
    assert model.source_status({"ecmwf_tmax_c": 31.0})["usable"] is False


def test_rcss_forecast_modes_cover_full_day() -> None:
    module = _load_script("123_predict_rcss_metar_tmax.py", "test_rcss_predictor_modes")

    assert module.forecast_mode_for_hour(2) == (6, "early_nwp_residual")
    assert module.forecast_mode_for_hour(6) == (6, "trained_intraday")
    assert module.forecast_mode_for_hour(15) == (15, "trained_intraday")
    assert module.forecast_mode_for_hour(20) == (20, "trained_intraday")
    assert module.forecast_mode_for_hour(22) == (20, "late_clamped_intraday")


def test_rcss_message_identifies_source_and_city() -> None:
    module = _load_script("123_predict_rcss_metar_tmax.py", "test_rcss_predictor_message")
    payload = {
        "target_date_local": "2026-07-22",
        "issue_time_local": "2026-07-22T15:00:00+08:00",
        "update_trigger": "new_metar",
        "forecast_mode": "trained_intraday",
        "model_version": "rcss_test",
        "forecast": {
            "expected_tmax_c": 35.2,
            "most_likely_integer_c": 35,
            "intervals": {"80": [34.0, 36.0]},
            "probabilities_by_integer_c": {"34": 0.2, "35": 0.6, "36": 0.2},
        },
        "latest_metar_record": {
            "observation_time_utc": "2026-07-22T07:00:00Z",
            "temperature_c": 33.0,
            "current_max_c": 33.0,
            "raw_metar": "METAR RCSS TEST",
        },
        "nwp": {
            "individual_tmax_c": {
                "jma_msm": 34.0,
                "jma_gsm": 35.0,
                "icon_global": 36.0,
                "ecmwf": 35.5,
            },
            "blend_tmax_c": 35.7,
            "spread_c": 2.0,
            "degraded": False,
            "available_models": ["jma_msm", "jma_gsm", "icon_global", "ecmwf"],
        },
    }

    text = module.format_forecast_message(payload)

    assert "RCSS Taipei Tmax forecast" in text
    assert "по Тайбэю" in text
    assert "Production anchor: JMA GSM" in text
    assert "JMA MSM" in text
    assert "ICON Global" in text
    assert "ECMWF IFS" in text


def test_rcss_telegram_uses_requested_chat_and_main_bot(monkeypatch) -> None:
    module = _load_script("124_rcss_forecast_job.py", "test_rcss_forecast_job")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN_RCSS", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "main-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_LFPB", "paris-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID_RCSS", raising=False)

    module._activate_telegram()

    assert module.os.environ["TELEGRAM_BOT_TOKEN"] == "main-token"
    assert module.os.environ["TELEGRAM_CHAT_ID"] == "-1004469237763"
    assert not hasattr(module, "_run_polymarket_paper")


def test_multi_airport_scheduler_includes_rcss_without_replacing_other_cities(
    monkeypatch,
    capsys,
) -> None:
    module = _load_script("55_multi_airport_job.py", "test_rcss_multi_airport_job")
    calls = []
    monkeypatch.setattr(
        module,
        "_run_step",
        lambda label, command: calls.append((label, command))
        or {"label": label, "returncode": 0},
    )

    monkeypatch.setattr(module.sys, "argv", ["55_multi_airport_job.py", "forecast-all"])
    module.main()
    forecast_labels = [label for label, _ in calls]
    assert forecast_labels == [
        "EDDM forecast",
        "LFPB forecast",
        "EHAM forecast",
        "LEMD forecast",
        "LIMC forecast",
        "RCSS forecast",
    ]

    calls.clear()
    monkeypatch.setattr(
        module.sys,
        "argv",
        ["55_multi_airport_job.py", "metar-event-all-once"],
    )
    module.main()
    assert [label for label, _ in calls][-1] == "RCSS METAR once"
    assert calls[-1][1][1] == "scripts/125_rcss_metar_event_job.py"
    capsys.readouterr()


def test_regression_residual_candidate_returns_normalized_distribution() -> None:
    rows = 180
    frame = pd.DataFrame(
        {
            "feature": np.linspace(-1.0, 1.0, rows),
            "model_tmax_c": np.full(rows, 30.0),
            "current_metar_max_c": np.linspace(25.0, 30.0, rows),
            "final_metar_tmax_c": 31.0 + np.sin(np.arange(rows) / 12.0),
            "local_issue_hour": np.tile(np.arange(6, 21), 12),
        }
    )
    model = RegressionResidualTmaxModel(feature_columns=["feature", "current_metar_max_c"])
    model.fit(frame.iloc[:120], frame.iloc[120:])
    distribution = model.predict_distribution(frame.iloc[-1].to_dict())

    assert np.isclose(distribution.probabilities.sum(), 1.0)
    assert distribution.bins_c.min() >= round(frame.iloc[-1]["current_metar_max_c"])
    assert set(model.component_predictions(frame.iloc[-1].to_dict())) == {
        "ridge",
        "hist_gradient_boosting",
        "extra_trees",
    }


def test_rcss_metadata_records_full_walk_forward_result() -> None:
    metadata = json.loads(
        (ROOT / "data/models/rcss_metar_tmax_multinwp_d1_v1.metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["airport"] == "RCSS"
    assert metadata["walk_forward_test_days"] == 270
    assert metadata["walk_forward_metrics"]["mae_expected"] < 0.5
    assert metadata["walk_forward_metrics"]["mean_shape_violations"] == 0.0
