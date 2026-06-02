import pandas as pd

from weather_tmax_bot.models.intraday_ml import IntradayMLUpsideModel, prepare_intraday_ml_dataset


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
