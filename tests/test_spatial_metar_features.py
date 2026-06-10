from datetime import date

import pandas as pd

from weather_tmax_bot.features.spatial_metar import build_spatial_metar_features


def test_spatial_metar_features_use_only_asof_neighbor_reports() -> None:
    base = {
        "latest_metar_temp_c": 20.0,
        "current_metar_max_c": 21.0,
        "max_feature_knowledge_time_utc": "2026-06-01T10:00:00Z",
    }
    lfpg = pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-01T09:30:00Z",
                "knowledge_time_utc": "2026-06-01T09:35:00Z",
                "temperature_c": 22.0,
                "dewpoint_c": 10.0,
                "raw_metar": "LFPG 010930Z CAVOK 22/10 Q1015",
                "cavok": True,
            },
            {
                "observation_time_utc": "2026-06-01T10:30:00Z",
                "knowledge_time_utc": "2026-06-01T10:35:00Z",
                "temperature_c": 25.0,
                "dewpoint_c": 10.0,
                "raw_metar": "LFPG 011030Z CAVOK 25/10 Q1015",
                "cavok": True,
            },
        ]
    )

    features = build_spatial_metar_features(
        base,
        {"LFPG": lfpg, "LFPO": pd.DataFrame()},
        target_date_local=date(2026, 6, 1),
        issue_time_utc=pd.Timestamp("2026-06-01T10:00:00Z"),
        timezone_name="Europe/Paris",
    )

    assert features["spatial_available_station_count"] == 1
    assert features["spatial_lfpg_latest_temp_c"] == 22.0
    assert features["spatial_latest_temp_max_c"] == 22.0
    assert features["spatial_any_neighbor_above_lfpb_latest"] is True
    assert features["spatial_leakage_check_passed"] is True
    assert features["spatial_max_feature_knowledge_time_utc"] == "2026-06-01T10:00:00+00:00"
