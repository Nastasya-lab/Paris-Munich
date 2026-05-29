from datetime import date

import pandas as pd

from weather_tmax_bot.data.open_meteo import _value_at_local_hour


def test_value_at_local_hour_uses_airport_timezone():
    day = pd.DataFrame(
        {
            "valid_time_utc": pd.to_datetime(["2026-07-15T06:00:00Z", "2026-07-15T12:00:00Z"], utc=True),
            "temperature_2m": [18.0, 25.0],
        }
    )
    assert _value_at_local_hour(day, "temperature_2m", date(2026, 7, 15), "Europe/Berlin", 8) == 18.0
