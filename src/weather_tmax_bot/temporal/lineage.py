from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class TemporalRecord(BaseModel):
    event_time_utc: datetime | None = None
    observation_time_utc: datetime | None = None
    issue_time_utc: datetime | None = None
    valid_from_utc: datetime | None = None
    valid_to_utc: datetime | None = None
    model_run_time_utc: datetime | None = None
    model_availability_time_utc: datetime | None = None
    ingest_time_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    knowledge_time_utc: datetime
    source_id: str
    source_name: str
    source_version: str = "unknown"
    source_url_or_reference: str | None = None
    raw_record_hash: str | None = None
    parser_version: str = "mvp.v1"
    quality_flag: str = "unchecked"
    payload: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        for field_name in (
            "event_time_utc",
            "observation_time_utc",
            "issue_time_utc",
            "valid_from_utc",
            "valid_to_utc",
            "model_run_time_utc",
            "model_availability_time_utc",
            "ingest_time_utc",
            "knowledge_time_utc",
        ):
            value = getattr(self, field_name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{field_name} must be timezone-aware")
