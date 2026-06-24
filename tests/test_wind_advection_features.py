from datetime import date

import pandas as pd

from weather_tmax_bot.features.wind_advection import build_wind_advection_features, wind_advection_feature_columns


def test_wind_advection_features_support_eddm_target_station() -> None:
    eddm = pd.DataFrame(
        [
            _metar("2026-06-01T09:00:00Z", 20.0, 10.0, 1012.0, 180, 6),
            _metar("2026-06-01T10:00:00Z", 21.0, 10.5, 1011.5, 190, 8),
        ]
    )
    edmo = pd.DataFrame(
        [
            _metar("2026-06-01T09:00:00Z", 22.0, 11.0, 1012.0, 200, 7),
            _metar("2026-06-01T10:00:00Z", 24.0, 12.0, 1011.0, 210, 10),
            _metar("2026-06-01T10:30:00Z", 30.0, 12.0, 1010.0, 220, 10),
        ]
    )

    features = build_wind_advection_features(
        {"EDDM": eddm, "EDMO": edmo},
        target_date_local=date(2026, 6, 1),
        issue_time_utc=pd.Timestamp("2026-06-01T10:00:00Z"),
        timezone_name="Europe/Berlin",
        stations=["EDDM", "EDMO"],
        target_station="EDDM",
    )

    assert "adv_neighbor_mean_minus_eddm_temp_trend_1h" in wind_advection_feature_columns(
        ["EDDM", "EDMO"], target_station="EDDM"
    )
    assert "adv_neighbor_mean_minus_lfpb_temp_trend_1h" in wind_advection_feature_columns()
    assert features["adv_available_station_count"] == 2
    assert features["adv_neighbor_mean_minus_eddm_temp_trend_1h"] == 1.0
    assert features["adv_edmo_minus_eddm_temp_trend_1h"] == 1.0
    assert features["adv_leakage_check_passed"] is True
    assert features["adv_max_feature_knowledge_time_utc"] == "2026-06-01T10:00:00+00:00"


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
        target_date_local=date(2026, 6, 1),
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


def _metar(time_utc: str, temperature: float, dewpoint: float, qnh: float, wind_dir: int, wind_speed: int) -> dict:
    return {
        "observation_time_utc": time_utc,
        "knowledge_time_utc": time_utc,
        "temperature_c": temperature,
        "dewpoint_c": dewpoint,
        "qnh_hpa": qnh,
        "wind_direction_deg": wind_dir,
        "wind_speed_kt": wind_speed,
    }
