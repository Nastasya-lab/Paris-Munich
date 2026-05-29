from __future__ import annotations

from datetime import datetime, timedelta, timezone


def metar_knowledge_time(observation_time_utc: datetime, latency_minutes: int = 5) -> datetime:
    return observation_time_utc.astimezone(timezone.utc) + timedelta(minutes=latency_minutes)


def taf_knowledge_time(issue_time_utc: datetime, latency_minutes: int = 5) -> datetime:
    return issue_time_utc.astimezone(timezone.utc) + timedelta(minutes=latency_minutes)


def nwp_knowledge_time(model_availability_time_utc: datetime) -> datetime:
    return model_availability_time_utc.astimezone(timezone.utc)


def final_observation_knowledge_time(event_time_utc: datetime, release_lag_days: int = 2) -> datetime:
    return event_time_utc.astimezone(timezone.utc) + timedelta(days=release_lag_days)
