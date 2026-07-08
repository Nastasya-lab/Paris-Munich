from __future__ import annotations

import pandas as pd

from weather_tmax_bot.models.distribution import unimodal_violation_count
from weather_tmax_bot.models.hf_icon_eu_shadow import (
    HfIconEuResidualPmfShadowModel,
    prepare_hf_icon_eu_training_frame,
)


def test_hf_icon_eu_shadow_model_predicts_unimodal_distribution_above_current_max():
    frame = pd.DataFrame(
        {
            "target_date_local": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "model_run_time_utc": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T06:00:00Z",
                "2024-01-04T06:00:00Z",
            ],
            "model_tmax_c": [10.0, 11.0, 12.0, 13.0],
        }
    )
    target = pd.DataFrame(
        {
            "target_date_local": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "metar_tmax_c": [11.0, 12.0, 13.0, 14.0],
        }
    )
    training = prepare_hf_icon_eu_training_frame(frame, target)

    model = HfIconEuResidualPmfShadowModel(minimum_run_hour_samples=1).fit(training)
    distribution = model.predict_distribution(
        {
            "model_tmax_c": 20.0,
            "model_run_hour": 0,
            "current_metar_max_c": 21.0,
        }
    )

    assert distribution.expected_tmax_c >= 21.0
    assert min(distribution.bins_c[distribution.probabilities > 0]) >= 21
    assert unimodal_violation_count(distribution) == 0
