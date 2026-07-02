from __future__ import annotations

import joblib
import pandas as pd

from weather_tmax_bot.models.discrete_hazard_tmax import (
    DiscreteHazardCalibrator,
    DiscreteHazardUpsideModel,
    hazard_calibration_rows,
)


def test_discrete_hazard_model_predicts_distribution_and_roundtrips(tmp_path):
    rows = []
    for day in range(80):
        current = 10.0 + (day % 3)
        upside = float(day % 5)
        rows.append(
            {
                "target_date_local": f"2026-04-{(day % 28) + 1:02d}",
                "issue_time_utc": pd.Timestamp("2026-04-01T10:00:00Z") + pd.Timedelta(days=day),
                "current_metar_max_c": current,
                "latest_metar_temp_c": current - 0.5,
                "remaining_upside_c": upside,
                "final_metar_tmax_c": current + upside,
                "local_issue_hour": 12,
                "leakage_check_passed": True,
            }
        )
    frame = pd.DataFrame(rows)
    model = DiscreteHazardUpsideModel(
        min_rows=40,
        min_at_risk_rows=10,
        max_iter=10,
        feature_columns=["local_issue_hour", "current_metar_max_c", "latest_metar_temp_c"],
    ).fit(frame)
    calibrator = DiscreteHazardCalibrator(max_upside_c=model.max_upside_c, min_rows_per_threshold=10).fit(
        hazard_calibration_rows(model, frame)
    )
    model.calibrator = calibrator

    dist = model.predict_distribution(frame.iloc[0])
    assert dist.probabilities.sum() == 1.0
    assert dist.bins_c.min() >= int(round(frame.iloc[0]["current_metar_max_c"]))

    path = tmp_path / "hazard.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    loaded_dist = loaded.predict_distribution(frame.iloc[1])
    assert loaded_dist.probabilities.sum() == 1.0
