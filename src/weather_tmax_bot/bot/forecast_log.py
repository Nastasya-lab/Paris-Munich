from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

from weather_tmax_bot.data.storage import append_jsonl
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.utils.hashing import stable_hash


def log_forecast(
    airport: str,
    issue_time_utc: datetime,
    target_date_local: date,
    distribution: TmaxDistribution,
    feature_snapshot: dict | None = None,
    model_version: str = "unknown",
    path: str | Path = "data/logs/forecast_log.jsonl",
) -> str:
    path = os.getenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(path))
    feature_snapshot = feature_snapshot or {}
    payload = distribution.to_payload()
    record = {
        "forecast_id": str(uuid4()),
        "airport": airport,
        "issue_time_utc": issue_time_utc.astimezone(timezone.utc).isoformat(),
        "target_date_local": target_date_local.isoformat(),
        "model_version": model_version,
        "feature_set_version": "mvp.v1",
        "source_registry_version": "2026-05-28.v1",
        "feature_snapshot_hash": stable_hash(feature_snapshot),
        "data_sources_used": feature_snapshot.get("data_sources_used", []),
        "max_feature_knowledge_time_utc": feature_snapshot.get("max_feature_knowledge_time_utc"),
        "probability_distribution": payload["probabilities_by_integer_c"],
        "expected_tmax_c": payload["expected_tmax_c"],
        "median_tmax_c": payload["median_tmax_c"],
        "most_likely_integer_c": payload["most_likely_integer_c"],
        "raw_input_metadata": feature_snapshot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(json.dumps(record, sort_keys=True, default=str), path)
    return record["forecast_id"]
