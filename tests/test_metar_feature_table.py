import pandas as pd

from weather_tmax_bot.features.metar_feature_table import build_metar_feature_table


def test_metar_feature_table_is_as_of_safe():
    metar = pd.DataFrame(
        {
            "observation_time_utc": ["2026-07-15T05:50:00Z", "2026-07-15T06:20:00Z"],
            "knowledge_time_utc": ["2026-07-15T05:55:00Z", "2026-07-15T06:25:00Z"],
            "temperature_c": [18.0, 30.0],
            "dewpoint_c": [12.0, 14.0],
            "qnh_hpa": [1018.0, 1017.0],
            "wind_direction_deg": [180.0, 180.0],
            "wind_speed_kt": [5.0, 6.0],
            "raw_metar": [
                "EDDM 150550Z 18005KT CAVOK 18/12 Q1018",
                "EDDM 150620Z 18006KT CAVOK 30/14 Q1017",
            ],
            "cavok": [True, True],
        }
    )
    features = build_metar_feature_table(metar, ["2026-07-15"])
    row_06 = features[features["issue_hour_utc"] == 6].iloc[0]
    assert row_06["last_metar_temp_c"] == 18.0
    assert row_06["observed_max_so_far_from_metar"] == 18.0
    assert pd.Timestamp(row_06["max_metar_knowledge_time_utc"]) <= pd.Timestamp("2026-07-15T06:00:00Z")
