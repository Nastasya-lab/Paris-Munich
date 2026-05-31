from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from weather_tmax_bot.utils.time import local_day_bounds_utc


ISSUE_HOURS_UTC = [0, 3, 6, 9, 12, 15, 18]
OPERATIONAL_ISSUE_SCHEDULE_MINUTES_UTC = [
    *(hour * 60 for hour in ISSUE_HOURS_UTC),
    # Open-Meteo regional-model metadata advises waiting after model availability.
    # These slots are intended for ICON-D2 availability-aware production forecasts.
    *(((hour * 60 + 100) % 1440) for hour in ISSUE_HOURS_UTC),
]


def build_issue_time_features(
    issue_time_utc: datetime,
    target_date_local: date,
    issue_hours_utc: list[int] | None = None,
    timezone_name: str = "Europe/Berlin",
) -> dict:
    issue_hours = issue_hours_utc or ISSUE_HOURS_UTC
    issue = pd.Timestamp(issue_time_utc)
    if issue.tzinfo is None:
        issue = issue.tz_localize("UTC")
    else:
        issue = issue.tz_convert("UTC")
    minute_of_day = issue.hour * 60 + issue.minute
    scheduled_minutes = (
        OPERATIONAL_ISSUE_SCHEDULE_MINUTES_UTC
        if issue_hours_utc is None
        else [hour * 60 for hour in issue_hours]
    )
    nearest_offset = min(abs(minute_of_day - value) for value in scheduled_minutes)
    # Wrap around midnight, e.g. 23:50 is 10 minutes from 00 UTC.
    nearest_offset = min(nearest_offset, min(1440 - abs(minute_of_day - value) for value in scheduled_minutes))
    _, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    lead_to_local_day_end_hours = (pd.Timestamp(day_end) - issue).total_seconds() / 3600
    return {
        "issue_minute_utc": int(issue.minute),
        "issue_hour_continuous_utc": float(issue.hour + issue.minute / 60),
        "issue_minutes_since_midnight_utc": int(minute_of_day),
        "issue_schedule_offset_minutes": int(nearest_offset),
        "issue_off_schedule": bool(nearest_offset > 0),
        "lead_to_local_day_end_hours": float(lead_to_local_day_end_hours),
    }
