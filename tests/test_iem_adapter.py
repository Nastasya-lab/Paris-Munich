from datetime import date, datetime, timezone

import pandas as pd

from weather_tmax_bot.data.iem import _empty_taf_frame
from weather_tmax_bot.data.metar import parse_metar
from weather_tmax_bot.features.metar_features import build_metar_features


def test_parse_metar_weather_codes_are_string():
    parsed = parse_metar("EDDM 151220Z AUTO 31004KT 6000 +TSRA FEW034 19/15 Q1019")
    assert parsed["temperature_c"] == 19
    assert "+TSRA" in parsed["weather_codes"]


def test_metar_features_only_use_passed_slice():
    df = pd.DataFrame(
        {
            "observation_time_utc": [
                datetime(2026, 7, 15, 5, tzinfo=timezone.utc),
                datetime(2026, 7, 15, 7, tzinfo=timezone.utc),
            ],
            "knowledge_time_utc": [
                datetime(2026, 7, 15, 5, 5, tzinfo=timezone.utc),
                datetime(2026, 7, 15, 7, 5, tzinfo=timezone.utc),
            ],
            "temperature_c": [18.0, 30.0],
            "dewpoint_c": [12.0, 15.0],
            "qnh_hpa": [1018.0, 1017.0],
            "source_id": ["iem.metar.archive.EDDM", "iem.metar.archive.EDDM"],
            "raw_metar": ["EDDM 150500Z 18005KT CAVOK 18/12 Q1018", "EDDM 150700Z 18005KT CAVOK 30/15 Q1017"],
        }
    )
    features = build_metar_features(df, datetime(2026, 7, 15, 6, tzinfo=timezone.utc))
    assert features["last_metar_temp_c"] == 18.0
    assert features["observed_max_so_far_from_metar"] == 18.0
    assert features["latest_metar_source_id"] == "iem.metar.archive.EDDM"


def test_metar_context_does_not_truncate_tomorrow_by_previous_day_max():
    df = pd.DataFrame(
        {
            "observation_time_utc": [datetime(2026, 7, 14, 20, tzinfo=timezone.utc)],
            "knowledge_time_utc": [datetime(2026, 7, 14, 20, 5, tzinfo=timezone.utc)],
            "temperature_c": [31.0],
            "dewpoint_c": [12.0],
            "qnh_hpa": [1018.0],
            "source_id": ["awc.metar.live.EDDM"],
            "raw_metar": ["EDDM 142000Z 18005KT CAVOK 31/12 Q1018"],
        }
    )
    features = build_metar_features(
        df,
        datetime(2026, 7, 14, 21, tzinfo=timezone.utc),
        target_date_local=date(2026, 7, 15),
    )
    assert features["last_metar_temp_c"] == 31.0
    assert pd.isna(features["observed_max_so_far_from_metar"])


def test_empty_taf_frame_has_required_columns():
    assert "knowledge_time_utc" in _empty_taf_frame().columns
