import pandas as pd

from weather_tmax_bot.features.build_target import build_daily_tmax


def test_daily_tmax_uses_local_day():
    obs = pd.DataFrame(
        {
            "station_id": ["01262", "01262"],
            "observation_time_utc": ["2026-07-14T22:10:00Z", "2026-07-15T12:00:00Z"],
            "temperature_c": [20.0, 25.0],
            "source_id": ["dwd.10min.air_temperature.01262"] * 2,
            "source_version": ["test"] * 2,
        }
    )
    out = build_daily_tmax(obs)
    row = out[out["target_date_local"] == "2026-07-15"].iloc[0]
    assert row["tmax_c"] == 25.0
