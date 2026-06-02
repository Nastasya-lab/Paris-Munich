import pandas as pd

from weather_tmax_bot.models.intraday_ml import (
    IntradayMLSurvivalCalibrator,
    IntradayMLUpsideModel,
    infer_intraday_ml_context,
    prepare_intraday_ml_dataset,
)


def test_prepare_intraday_ml_dataset_builds_remaining_upside_labels():
    dataset = pd.DataFrame(
        {
            "target_date_local": ["2026-06-01", "2026-06-02"],
            "issue_time_utc": ["2026-06-01T12:00:00Z", "2026-06-02T12:00:00Z"],
            "tmax_c": [25.0, 20.0],
            "observed_max_so_far_from_metar": [22.0, 20.0],
            "last_metar_temp_c": [21.0, 19.0],
            "leakage_check_passed": [True, True],
        }
    )

    prepared = prepare_intraday_ml_dataset(dataset)

    assert prepared["remaining_upside_c"].tolist() == [3.0, 0.0]
    assert prepared["peak_already_passed"].tolist() == [False, True]
    assert prepared["upside_ge_3c"].tolist() == [True, False]


def test_intraday_ml_predicts_monotonic_survival_and_distribution():
    rows = []
    for idx in range(80):
        rows.append(
            {
                "target_date_local": f"2026-01-{idx % 28 + 1:02d}",
                "issue_time_utc": f"2026-01-{idx % 28 + 1:02d}T12:00:00Z",
                "issue_hour_utc": 12,
                "month": 1,
                "doy_sin": 0.0,
                "doy_cos": 1.0,
                "tmax_c": 20.0 + idx % 5,
                "observed_max_so_far_from_metar": 20.0,
                "last_metar_temp_c": 19.0,
                "leakage_check_passed": True,
                "nwp_missing": True,
                "taf_missing": True,
            }
        )
    model = IntradayMLUpsideModel(max_upside_c=5, min_rows=40).fit(pd.DataFrame(rows))

    dist, details = model.predict_distribution(rows[0])

    survival = list(details["upside_survival_probabilities"].values())
    assert all(left >= right for left, right in zip(survival, survival[1:]))
    assert abs(dist.probabilities.sum() - 1.0) < 1e-6
    assert dist.bins_c.min() == 20
    assert details["active"] is True


def test_intraday_ml_survival_calibrator_preserves_ordinal_monotonicity():
    calibration_rows = pd.DataFrame(
        {
            "issue_hour_utc": [12, 12, 12, 12],
            "remaining_upside_c": [0.0, 1.0, 2.0, 3.0],
            "raw_probability_upside_ge_1c": [0.1, 0.3, 0.7, 0.9],
            "actual_upside_ge_1c": [0.0, 0.0, 1.0, 1.0],
            "raw_probability_upside_ge_2c": [0.05, 0.2, 0.5, 0.8],
            "actual_upside_ge_2c": [0.0, 0.0, 0.0, 1.0],
            "raw_probability_upside_ge_3c": [0.01, 0.1, 0.4, 0.7],
            "actual_upside_ge_3c": [0.0, 0.0, 0.0, 1.0],
        }
    )
    calibrator = IntradayMLSurvivalCalibrator(max_upside_c=3, min_rows_per_threshold=4).fit(calibration_rows)

    calibrated = calibrator.transform({1: 0.8, 2: 0.75, 3: 0.7})

    values = list(calibrated.values())
    assert calibrator.fitted is True
    assert all(0.0 <= value <= 1.0 for value in values)
    assert all(left >= right for left, right in zip(values, values[1:]))


def test_intraday_ml_reports_when_oof_calibration_is_applied():
    rows = []
    for idx in range(80):
        rows.append(
            {
                "target_date_local": f"2026-01-{idx % 28 + 1:02d}",
                "issue_time_utc": f"2026-01-{idx % 28 + 1:02d}T12:00:00Z",
                "issue_hour_utc": 12,
                "month": 1,
                "doy_sin": 0.0,
                "doy_cos": 1.0,
                "tmax_c": 20.0 + idx % 5,
                "observed_max_so_far_from_metar": 20.0,
                "last_metar_temp_c": 19.0,
                "leakage_check_passed": True,
            }
        )
    model = IntradayMLUpsideModel(max_upside_c=3, min_rows=40).fit(pd.DataFrame(rows))
    model.calibrator = IntradayMLSurvivalCalibrator(max_upside_c=3, min_rows_per_threshold=4).fit(
        pd.DataFrame(
            {
                "issue_hour_utc": [12, 12, 12, 12],
                "remaining_upside_c": [0.0, 1.0, 2.0, 3.0],
                "raw_probability_upside_ge_1c": [0.1, 0.3, 0.7, 0.9],
                "actual_upside_ge_1c": [0.0, 0.0, 1.0, 1.0],
                "raw_probability_upside_ge_2c": [0.05, 0.2, 0.5, 0.8],
                "actual_upside_ge_2c": [0.0, 0.0, 0.0, 1.0],
                "raw_probability_upside_ge_3c": [0.01, 0.1, 0.4, 0.7],
                "actual_upside_ge_3c": [0.0, 0.0, 0.0, 1.0],
            }
        )
    )

    _, details = model.predict_distribution(rows[0])

    assert details["calibration_status"] == "contextual_out_of_fold_survival_calibrated"
    assert "raw_probability_upside_ge_1c" in details


def test_intraday_ml_context_separates_morning_rain_from_late_sharp_drop():
    morning = infer_intraday_ml_context(
        {
            "issue_hour_utc": 6,
            "month": 6,
            "last_metar_temp_c": 15.0,
            "observed_max_so_far_from_metar": 18.0,
            "has_precip_recent": True,
            "model_precip_sum": 0.0,
        }
    )
    evening = infer_intraday_ml_context(
        {
            "issue_hour_utc": 15,
            "month": 6,
            "last_metar_temp_c": 18.0,
            "observed_max_so_far_from_metar": 23.0,
            "has_precip_recent": True,
            "model_precip_sum": 2.0,
        }
    )

    assert morning["phase"] == "morning"
    assert morning["weather_regime"] == "adverse"
    assert evening["phase"] == "evening"
    assert evening["weather_regime"] == "sharp_drop"
