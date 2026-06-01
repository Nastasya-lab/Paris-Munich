from datetime import date

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.intraday_survival_prior import (
    adjust_upside_probability,
    build_daily_first_metar_max,
    build_seasonal_hourly_survival_table,
    lookup_survival_prior,
    season_for_month,
)
from weather_tmax_bot.models.distribution import TmaxDistribution


def test_season_for_month():
    assert season_for_month(1) == "winter_DJF"
    assert season_for_month(4) == "spring_MAM"
    assert season_for_month(7) == "summer_JJA"
    assert season_for_month(10) == "autumn_SON"


def test_build_survival_table_uses_first_hour_of_daily_maximum():
    metar = pd.DataFrame(
        {
            "observation_time_utc": [
                "2025-07-01T12:00:00Z",
                "2025-07-01T13:00:00Z",
                "2025-07-01T14:00:00Z",
                "2025-07-02T13:00:00Z",
                "2025-07-02T14:00:00Z",
                "2025-07-02T15:00:00Z",
            ],
            "temperature_c": [20, 22, 22, 20, 21, 23],
        }
    )

    daily = build_daily_first_metar_max(metar, min_obs_count=3)
    table = build_seasonal_hourly_survival_table(daily, train_before=date(2025, 8, 1))

    assert daily["first_max_hour_local"].tolist() == [15, 17]
    assert lookup_survival_prior(table, month=7, local_hour=14.9) == 1.0
    assert lookup_survival_prior(table, month=7, local_hour=15.1) == 0.5
    assert lookup_survival_prior(table, month=7, local_hour=17.1) == 0.0


def test_cap_blend_moves_excess_upside_mass_to_observed_maximum():
    dist = TmaxDistribution(np.array([22, 23, 24]), np.array([0.1, 0.4, 0.5]))

    adjusted = adjust_upside_probability(
        dist,
        observed_max_so_far_c=22.0,
        survival_prior=0.2,
        formula="cap_blend",
        strength=1.0,
    )

    assert np.isclose(adjusted.distribution.probabilities.sum(), 1.0)
    assert np.isclose(adjusted.original_upside_probability, 0.9)
    assert np.isclose(adjusted.adjusted_upside_probability, 0.2)
    assert np.allclose(adjusted.distribution.probabilities, [0.8, 0.0888888889, 0.1111111111])


def test_multiply_formula_can_apply_partial_strength():
    dist = TmaxDistribution(np.array([23, 24]), np.array([0.75, 0.25]))

    adjusted = adjust_upside_probability(
        dist,
        observed_max_so_far_c=23.0,
        survival_prior=0.04,
        formula="multiply",
        strength=0.5,
    )

    assert np.isclose(adjusted.adjusted_upside_probability, 0.05)
    assert np.allclose(adjusted.distribution.probabilities, [0.95, 0.05])
