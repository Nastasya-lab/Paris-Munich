from weather_tmax_bot.models.weather_regime import detect_weather_regime


def test_weather_regime_detects_frontal_rain_after_drop() -> None:
    prediction = detect_weather_regime(
        {
            "has_rain_recent_metar": True,
            "rain_started_after_current_max": True,
            "drop_from_current_max_c": 4.0,
            "temp_trend_3h": -3.0,
            "cloud_cover_proxy_latest": 0.9,
            "model_future_cloud_cover_mean": 0.8,
            "model_future_precip_sum": 2.0,
            "pressure_tendency_3h": -1.4,
        }
    )

    assert prediction.label == "frontal_rain"
    assert prediction.scores["frontal_rain"] > prediction.scores["clear_heating"]


def test_weather_regime_detects_clear_heating() -> None:
    prediction = detect_weather_regime(
        {
            "is_cavok_latest": True,
            "cloud_cover_proxy_latest": 0.1,
            "model_future_cloud_cover_mean": 0.1,
            "model_future_shortwave_radiation_sum": 1400.0,
            "temp_trend_1h": 1.0,
            "temp_slope_since_sunrise": 0.7,
            "dewpoint_depression_latest": 8.0,
            "model_future_precip_sum": 0.0,
        }
    )

    assert prediction.label == "clear_heating"
    assert prediction.scores["clear_heating"] > prediction.scores["cloud_limited"]
