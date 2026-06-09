from __future__ import annotations

import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_intraday_survival import apply_metar_intraday_survival_layer


def test_late_rain_and_low_future_max_moves_upside_mass_to_observed_max() -> None:
    base = TmaxDistribution([19, 20, 21, 22], [0.12, 0.44, 0.34, 0.10])
    history = _history()
    feature_row = {
        "target_date_local": "2026-06-09",
        "local_issue_hour": 18.0,
        "current_metar_max_c": 19.0,
        "latest_metar_temp_c": 16.0,
        "drop_from_current_max_c": 3.0,
        "temp_trend_1h": -1.0,
        "temp_trend_3h": -3.0,
        "has_rain_recent_metar": True,
        "nwp_future_minus_current_max_c": -1.3,
        "model_future_temp_max_c": 17.7,
    }

    result = apply_metar_intraday_survival_layer(base, feature_row, historical_dataset=history)

    assert result.active is True
    assert result.distribution.threshold_ge(20) < 0.15
    assert result.distribution.probabilities[result.distribution.bins_c == 19][0] > 0.85
    assert result.details["adjusted_probability_upside_ge_1c"] < result.details["original_probability_upside_ge_1c"]


def test_morning_heating_window_keeps_survival_layer_weak() -> None:
    base = TmaxDistribution([15, 16, 17, 18, 19, 20], [0.05, 0.05, 0.10, 0.20, 0.25, 0.35])
    history = _history()
    feature_row = {
        "target_date_local": "2026-06-09",
        "local_issue_hour": 9.0,
        "current_metar_max_c": 15.0,
        "latest_metar_temp_c": 15.0,
        "drop_from_current_max_c": 0.0,
        "temp_trend_1h": 1.0,
        "temp_trend_3h": 3.0,
        "has_rain_recent_metar": False,
        "nwp_future_minus_current_max_c": 4.0,
        "model_future_temp_max_c": 19.0,
    }

    result = apply_metar_intraday_survival_layer(base, feature_row, historical_dataset=history)

    assert result.active is True
    assert result.details["effective_strength"] <= 0.1
    assert result.distribution.threshold_ge(16) > 0.85


def _history() -> pd.DataFrame:
    rows = []
    for day in range(1, 41):
        for hour, upside in [
            (8, 4),
            (10, 3),
            (12, 2),
            (14, 1 if day % 3 else 2),
            (16, 1 if day % 4 == 0 else 0),
            (18, 1 if day % 20 == 0 else 0),
            (20, 0),
        ]:
            rows.append(
                {
                    "target_date_local": f"2025-06-{(day - 1) % 30 + 1:02d}",
                    "local_issue_hour": float(hour),
                    "remaining_upside_c": float(upside),
                }
            )
    return pd.DataFrame(rows)
