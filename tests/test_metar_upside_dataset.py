from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from weather_tmax_bot.features.metar_upside_dataset import build_current_metar_upside_features, build_metar_remaining_upside_dataset


def test_metar_remaining_upside_dataset_uses_only_asof_metars():
    metar = pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-01T06:00:00Z",
                "knowledge_time_utc": "2026-06-01T06:05:00Z",
                "temperature_c": 18.0,
                "raw_metar": "METAR LFPB 010600Z 9999 18/12 Q1015",
                "cavok": False,
            },
            {
                "observation_time_utc": "2026-06-01T09:50:00Z",
                "knowledge_time_utc": "2026-06-01T09:55:00Z",
                "temperature_c": 23.0,
                "raw_metar": "METAR LFPB 010950Z 9999 23/12 Q1015",
                "cavok": True,
            },
            {
                "observation_time_utc": "2026-06-01T15:50:00Z",
                "knowledge_time_utc": "2026-06-01T15:55:00Z",
                "temperature_c": 25.0,
                "raw_metar": "METAR LFPB 011550Z 9999 25/12 Q1015",
                "cavok": True,
            },
        ]
    )
    target = pd.DataFrame(
        [{"target_date_local": "2026-06-01", "metar_tmax_c": 25.0, "quality_flags": "ok"}]
    )

    dataset = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
        local_issue_hours=[12, 18],
    )

    midday = dataset[dataset["local_issue_hour"] == 12].iloc[0]
    evening = dataset[dataset["local_issue_hour"] == 18].iloc[0]
    assert midday["current_metar_max_c"] == 23.0
    assert midday["remaining_upside_c"] == 2.0
    assert midday["upside_ge_2c"] == True
    assert evening["current_metar_max_c"] == 25.0
    assert evening["remaining_upside_c"] == 0.0
    assert evening["upside_ge_1c"] == False
    assert dataset["leakage_check_passed"].all()


def test_current_metar_upside_features_do_not_use_future_metar():
    metar = pd.DataFrame(
        [
            {
                "observation_time_utc": "2024-06-01T07:50:00Z",
                "knowledge_time_utc": "2024-06-01T07:55:00Z",
                "temperature_c": 18.0,
                "raw_metar": "METAR LFPB 010750Z 18/10 Q1015",
                "cavok": False,
            },
            {
                "observation_time_utc": "2024-06-01T10:50:00Z",
                "knowledge_time_utc": "2024-06-01T10:55:00Z",
                "temperature_c": 24.0,
                "raw_metar": "METAR LFPB 011050Z 24/10 Q1015",
                "cavok": True,
            },
        ]
    )

    row = build_current_metar_upside_features(
        metar,
        airport_icao="LFPB",
        target_date_local=date(2024, 6, 1),
        issue_time_utc=pd.Timestamp("2024-06-01T10:00:00Z"),
        timezone_name="Europe/Paris",
    )

    assert row["current_metar_max_c"] == 18.0
    assert row["latest_metar_temp_c"] == 18.0
    assert row["leakage_check_passed"]


def test_metar_remaining_upside_dataset_adds_rain_features():
    metar = pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-01T09:50:00Z",
                "knowledge_time_utc": "2026-06-01T09:55:00Z",
                "temperature_c": 23.0,
                "raw_metar": "METAR LFPB 010950Z RA 23/12 Q1015",
            }
        ]
    )
    target = pd.DataFrame(
        [{"target_date_local": "2026-06-01", "metar_tmax_c": 24.0, "quality_flags": "ok"}]
    )
    rain = pd.DataFrame(
        [
            {"observation_time_utc": "2026-06-01T09:54:00Z", "rr_mm": 0.2},
            {"observation_time_utc": "2026-06-01T10:00:00Z", "rr_mm": 0.4},
        ]
    )

    dataset = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
        local_issue_hours=[12],
        rain_6min=rain,
    )

    row = dataset.iloc[0]
    assert row["rain_6min_missing"] == False
    assert row["rain_mm_last_30m"] == pytest.approx(0.6)
    assert row["has_rain_recent_metar"] == True
