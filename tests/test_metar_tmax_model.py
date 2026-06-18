from __future__ import annotations

import numpy as np
import pandas as pd

from weather_tmax_bot.models.metar_tmax_model import (
    MetarTmaxHybridModel,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    mix_distributions,
    prepare_metar_tmax_dataset,
    survival_to_probabilities,
)
from weather_tmax_bot.models.distribution import TmaxDistribution


def _synthetic_dataset(rows: int = 160) -> pd.DataFrame:
    records = []
    for i in range(rows):
        issue_hour = [6, 8, 10, 12, 14, 16, 18, 20][i % 8]
        current_max = 10.0 + (i % 12) * 0.7 + max(issue_hour - 12, 0) * 0.15
        expected_upside = max(0.0, 4.5 - issue_hour * 0.18 + ((i % 5) - 2) * 0.25)
        final_tmax = current_max + expected_upside
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 8)
        records.append(
            {
                "airport_icao": "LFPB",
                "target_date_local": date.date().isoformat(),
                "issue_time_utc": (date + pd.Timedelta(hours=issue_hour - 1)).isoformat() + "Z",
                "local_issue_hour": issue_hour,
                "final_metar_tmax_c": final_tmax,
                "current_metar_max_c": current_max,
                "latest_metar_temp_c": current_max - max(0, issue_hour - 15) * 0.2,
                "drop_from_current_max_c": max(0.0, issue_hour - 15) * 0.2,
                "remaining_upside_c": final_tmax - current_max,
                "metar_count_so_far": max(1, issue_hour * 2),
                "metar_count_last_1h": 2,
                "metar_count_last_3h": 6,
                "temp_trend_1h": 0.2 if issue_hour < 14 else -0.1,
                "temp_trend_3h": 0.8 if issue_hour < 14 else -0.4,
                "temp_trend_6h": 1.2 if issue_hour < 14 else -0.7,
                "has_rain_recent_metar": i % 11 == 0,
                "has_thunder_recent_metar": False,
                "is_cavok_latest": i % 3 == 0,
                "rain_mm_last_30m": 0.0,
                "rain_mm_last_1h": 0.0,
                "rain_mm_last_3h": 0.0,
                "rain_mm_since_midnight": 0.0,
                "rain_max_6min_last_3h": 0.0,
                "leakage_check_passed": True,
            }
        )
    return pd.DataFrame(records)


def test_prepare_metar_tmax_dataset_adds_calendar_features_and_filters_leakage():
    dataset = _synthetic_dataset(16)
    dataset.loc[0, "leakage_check_passed"] = False

    prepared = prepare_metar_tmax_dataset(dataset)

    assert len(prepared) == 15
    assert {"month", "doy_sin", "doy_cos"}.issubset(prepared.columns)
    assert prepared["remaining_upside_c"].min() >= 0


def test_survival_to_probabilities_is_valid_distribution():
    probs = survival_to_probabilities({1: 0.7, 2: 0.4, 3: 0.1}, max_upside_c=3)

    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)
    assert np.allclose(probs, [0.3, 0.3, 0.3, 0.1])


def test_metar_tmax_model_predicts_monotone_no_below_current_max_distribution():
    model = MetarTmaxUpsideModel(max_upside_c=6, min_rows=40).fit(_synthetic_dataset())
    row = _synthetic_dataset(8).iloc[-1].to_dict()
    row["current_metar_max_c"] = 22.0

    survival = model.predict_upside_survival(row)
    dist = model.predict_distribution(row)

    assert all(survival[k] >= survival[k + 1] for k in range(1, model.max_upside_c))
    assert np.isclose(dist.probabilities.sum(), 1.0)
    assert dist.probabilities[dist.bins_c < 22].sum() == 0
    assert dist.bins_c.min() == 22


def test_metar_tmax_calibrator_keeps_distribution_valid():
    model = MetarTmaxUpsideModel(max_upside_c=6, min_rows=40).fit(_synthetic_dataset())
    prepared = prepare_metar_tmax_dataset(_synthetic_dataset())
    raw = model.predict_upside_survival_frame(prepared)
    calibration_rows = []
    for index, row in prepared.iterrows():
        out = {
            "local_issue_hour": int(row["local_issue_hour"]),
            "season": "winter",
            "remaining_upside_c": float(row["remaining_upside_c"]),
        }
        for threshold in range(1, model.max_upside_c + 1):
            out[f"raw_probability_upside_ge_{threshold}c"] = float(raw.loc[index, f"probability_upside_ge_{threshold}c"])
            out[f"actual_upside_ge_{threshold}c"] = float(row["remaining_upside_c"] >= threshold)
        calibration_rows.append(out)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=6, min_rows_per_threshold=20).fit(pd.DataFrame(calibration_rows))
    model.calibrator = calibrator
    row = prepared.iloc[-1].to_dict()

    dist = model.predict_distribution(row)

    assert calibrator.fitted
    assert np.isclose(dist.probabilities.sum(), 1.0)
    assert dist.bins_c.min() == int(round(row["current_metar_max_c"]))


def test_mix_distributions_keeps_probability_mass():
    base = TmaxDistribution(np.array([20, 21]), np.array([0.8, 0.2]))
    prior = TmaxDistribution(np.array([21, 22]), np.array([0.5, 0.5]))

    mixed = mix_distributions(base, prior, prior_weight=0.4)

    assert np.isclose(mixed.probabilities.sum(), 1.0)
    assert mixed.bins_c.tolist() == [20, 21, 22]
    assert np.allclose(mixed.probabilities, [0.48, 0.32, 0.20])


def test_hybrid_model_predicts_distribution():
    base_model = MetarTmaxUpsideModel(max_upside_c=6, min_rows=40).fit(_synthetic_dataset())
    row = _synthetic_dataset(8).iloc[-1].to_dict()
    row["current_metar_max_c"] = 22.0
    row["season"] = "winter"
    hybrid = MetarTmaxHybridModel(
        base_model=base_model,
        phase_priors={"20|winter": np.array([0.0, 1.0, 2.0]), "20|all": np.array([0.0])},
        global_prior=np.array([0.0]),
        blend_weight=0.5,
    )

    dist = hybrid.predict_distribution(row)

    assert np.isclose(dist.probabilities.sum(), 1.0)
    assert dist.bins_c.min() >= 22
