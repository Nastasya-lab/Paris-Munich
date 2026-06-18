from __future__ import annotations

import pandas as pd

from weather_tmax_bot.features.build_metar_target import build_daily_metar_tmax


def test_build_daily_metar_tmax_uses_local_day_and_integer_metar_target():
    metar = pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-01T21:50:00Z",
                "temperature_c": 18.0,
                "raw_metar": "METAR LFPB 012150Z 00000KT 9999 18/12 Q1015",
            },
            {
                "observation_time_utc": "2026-06-01T22:20:00Z",
                "temperature_c": 19.0,
                "raw_metar": "METAR LFPB 012220Z 00000KT 9999 19/12 Q1015",
            },
            {
                "observation_time_utc": "2026-06-02T12:20:00Z",
                "temperature_c": 24.0,
                "raw_metar": "SPECI LFPB 021220Z 00000KT 9999 24/12 Q1015",
            },
        ]
    )

    target = build_daily_metar_tmax(
        metar,
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
        source_id="iem.metar.archive.LFPB",
        expected_reports_per_day=48,
    )

    first = target[target["target_date_local"] == "2026-06-01"].iloc[0]
    second = target[target["target_date_local"] == "2026-06-02"].iloc[0]
    assert first["metar_tmax_c"] == 18.0
    assert second["metar_tmax_c"] == 24.0
    assert second["has_speci"] == True
    assert second["quality_flags"] == "low_coverage"


def test_build_daily_metar_tmax_empty_frame_has_expected_columns():
    target = build_daily_metar_tmax(
        pd.DataFrame(),
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
        source_id="iem.metar.archive.LFPB",
    )

    assert target.empty
    assert "metar_tmax_c" in target.columns
