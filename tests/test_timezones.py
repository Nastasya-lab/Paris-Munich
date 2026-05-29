from datetime import date

from weather_tmax_bot.utils.time import local_day_bounds_utc


def test_local_day_bounds_use_berlin_dst():
    start, end = local_day_bounds_utc(date(2026, 7, 15), "Europe/Berlin")
    assert start.hour == 22
    assert end.hour == 21
