from __future__ import annotations

import pandas as pd

from weather_tmax_bot.features.wind_advection import build_wind_advection_features


def test_wind_advection_detects_cold_front_signal_without_leakage() -> None:
    metar = pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-01T10:00:00Z",
                "knowledge_time_utc": "2026-06-01T10:05:00Z",
                "temperature_c": 24.0,
                "dewpoint_c": 15.0,
                "qnh_hpa": 1012.0,
                "wind_direction_deg": 210.0,
                "wind_speed_kt": 8.0,
            },
            {
                "observation_time_utc": "2026-06-01T11:00:00Z",
                "knowledge_time_utc": "2026-06-01T11:05:00Z",
                "temperature_c": 21.0,
                "dewpoint_c": 11.0,
                "qnh_hpa": 1014.0,
                "wind_direction_deg": 330.0,
                "wind_speed_kt": 14.0,
            },
            {
                "observation_time_utc": "2026-06-01T12:30:00Z",
                "knowledge_time_utc": "2026-06-01T12:35:00Z",
                "temperature_c": 25.0,
                "dewpoint_c": 17.0,
                "qnh_hpa": 1010.0,
                "wind_direction_deg": 180.0,
                "wind_speed_kt": 6.0,
            },
        ]
    )

    features = build_wind_advection_features(
        {"LFPB": metar},
        target_date_local=pd.Timestamp("2026-06-01").date(),
        issue_time_utc=pd.Timestamp("2026-06-01T11:10:00Z"),
        timezone_name="Europe/Paris",
        stations=["LFPB"],
    )

    assert features["adv_lfpb_available"] is True
    assert features["adv_lfpb_north_sector_latest"] is True
    assert features["adv_lfpb_cold_advection_signal"] is True
    assert features["adv_lfpb_frontal_passage_signal"] is True
    assert features["adv_available_station_count"] == 1
    assert features["adv_leakage_check_passed"] is True
    assert features["adv_max_feature_knowledge_time_utc"] == "2026-06-01T11:05:00+00:00"
