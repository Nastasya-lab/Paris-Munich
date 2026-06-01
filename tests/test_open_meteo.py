from datetime import date, datetime, timezone

import pandas as pd

from weather_tmax_bot.data import open_meteo
from weather_tmax_bot.data.open_meteo import _value_at_local_hour


def test_value_at_local_hour_uses_airport_timezone():
    day = pd.DataFrame(
        {
            "valid_time_utc": pd.to_datetime(["2026-07-15T06:00:00Z", "2026-07-15T12:00:00Z"], utc=True),
            "temperature_2m": [18.0, 25.0],
        }
    )
    assert _value_at_local_hour(day, "temperature_2m", date(2026, 7, 15), "Europe/Berlin", 8) == 18.0


def test_single_run_extract_preserves_issued_run_identity(monkeypatch):
    class DummyResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            times = pd.date_range("2026-07-15T00:00:00Z", periods=48, freq="h")
            return {
                "hourly": {
                    "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                    "temperature_2m": list(range(48)),
                    "dew_point_2m": [10.0] * 48,
                    "relative_humidity_2m": [55.0] * 48,
                    "cloud_cover": [20.0] * 48,
                    "precipitation": [0.0] * 48,
                    "shortwave_radiation": [100.0] * 48,
                    "wind_speed_10m": [5.0] * 48,
                    "wind_gusts_10m": [8.0] * 48,
                    "surface_pressure": [980.0] * 48,
                }
            }

    calls = {}

    def fake_get(url, params, timeout):
        calls["url"] = url
        calls["params"] = params
        return DummyResponse()

    monkeypatch.setattr(open_meteo.requests, "get", fake_get)
    run = datetime(2026, 7, 15, 0, tzinfo=timezone.utc)
    rows = open_meteo.fetch_open_meteo_single_run_extract(
        airport_icao="EDDM",
        latitude=48.3538,
        longitude=11.7861,
        run_time_utc=run,
        target_dates_local=[date(2026, 7, 15)],
        timezone_name="Europe/Berlin",
        availability_latency_hours=3.0,
    )

    assert calls["url"] == open_meteo.OPEN_METEO_SINGLE_RUN_URL
    assert calls["params"]["run"] == "2026-07-15T00:00"
    assert rows.iloc[0]["source_id"] == "open_meteo.single_run.icon_d2"
    assert rows.iloc[0]["model_run_time_utc"] == run
    assert rows.iloc[0]["model_availability_time_utc"] == datetime(2026, 7, 15, 3, tzinfo=timezone.utc)
    assert rows.iloc[0]["model_future_temp_max_c"] == 21
    assert rows.iloc[0]["model_future_precip_sum"] == 0.0
    assert rows.iloc[0]["quality_flag"] == "historical_forecast_as_issued_single_run"
