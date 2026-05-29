from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(timezone.utc)


def parse_issue_time(value: str | None) -> datetime:
    if value in (None, "now"):
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ensure_utc(parsed)


def local_day_bounds_utc(local_date: date, tz_name: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    start_local = datetime.combine(local_date, time.min, tzinfo=tz)
    end_local = datetime.combine(local_date, time.max, tzinfo=tz)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def to_local_date(dt_utc: datetime, tz_name: str) -> date:
    return ensure_utc(dt_utc).astimezone(ZoneInfo(tz_name)).date()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
