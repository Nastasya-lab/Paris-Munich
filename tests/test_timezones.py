from datetime import date
from datetime import datetime
from datetime import timezone

from weather_tmax_bot.features.issue_time_features import build_issue_time_features
from weather_tmax_bot.utils.time import local_day_bounds_utc


def test_local_day_bounds_use_berlin_dst():
    start, end = local_day_bounds_utc(date(2026, 7, 15), "Europe/Berlin")
    assert start.hour == 22
    assert end.hour == 21


def test_issue_schedule_accepts_top_of_hour_and_icon_availability_slots():
    top = build_issue_time_features(datetime(2026, 5, 31, 15, 0, tzinfo=timezone.utc), date(2026, 5, 31))
    shifted = build_issue_time_features(datetime(2026, 5, 31, 16, 40, tzinfo=timezone.utc), date(2026, 5, 31))
    near_shifted = build_issue_time_features(datetime(2026, 5, 31, 16, 36, tzinfo=timezone.utc), date(2026, 5, 31))

    assert top["issue_schedule_offset_minutes"] == 0
    assert shifted["issue_schedule_offset_minutes"] == 0
    assert near_shifted["issue_schedule_offset_minutes"] == 4
