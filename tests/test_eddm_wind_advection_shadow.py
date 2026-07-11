from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from weather_tmax_bot.models import predict
from weather_tmax_bot.models.distribution import TmaxDistribution


class _Model:
    model_version = "eddm_wind_test"

    def predict_distribution(self, feature_row):
        return TmaxDistribution([20, 21], [0.4, 0.6])


def test_eddm_wind_advection_shadow_is_active_before_noon(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "eddm_wind.joblib"
    model_path.touch()
    monkeypatch.setattr(predict, "EDDM_WIND_ADVECTION_MODEL_PATH", model_path)
    monkeypatch.setattr(predict.joblib, "load", lambda path: _Model())
    monkeypatch.setattr(predict, "_load_json", lambda path: {"model_version": "eddm_wind_test"})
    monkeypatch.setattr(
        predict,
        "build_wind_advection_features",
        lambda *args, **kwargs: {
            "adv_leakage_check_passed": True,
            "adv_available_station_count": 1,
        },
    )

    result = predict._predict_eddm_wind_advection_candidate(
        airport="EDDM",
        target_date=date(2026, 7, 11),
        issue_time_utc=datetime(2026, 7, 11, 7, 0, tzinfo=UTC),
        base_feature_row={"current_metar_max_c": 20.0},
        metar=pd.DataFrame(),
        spatial_metars={},
        champion=TmaxDistribution([20, 21], [0.5, 0.5]),
    )

    assert result["active"] is True
    assert result["active_local_hour_window"] == "all_day"
    assert result["reason"] == "active_all_day_spatial_wind_advection_shadow"
