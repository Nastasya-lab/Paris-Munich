from datetime import date

import pandas as pd

from weather_tmax_bot.models.nwp_residual_model import NWPResidualDistributionModel


def test_nwp_residual_model_predicts_distribution():
    dataset = pd.DataFrame(
        {
            "target_date_local": [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)],
            "tmax_c": [10.0, 11.0, 12.0, 13.0],
            "model_tmax_c": [9.0, 10.0, 10.0, 11.0],
            "month": [1, 1, 1, 1],
            "issue_hour_utc": [6, 6, 6, 6],
            "nwp_missing": [False, False, False, False],
        }
    )
    model = NWPResidualDistributionModel(min_group_rows=2).fit(dataset)

    dist = model.predict_distribution(
        pd.DataFrame([{"model_tmax_c": 20.0, "month": 1, "issue_hour_utc": 6, "nwp_missing": False}])
    )

    assert dist.expected_tmax_c > 20
    assert abs(dist.probabilities.sum() - 1.0) < 1e-6
