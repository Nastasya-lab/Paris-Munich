from __future__ import annotations

from datetime import datetime
from pathlib import Path

from weather_tmax_bot.temporal.freshness import assess_archive_freshness


def evaluate_freshness_gate(
    root: str | Path = ".",
    issue_time_utc: datetime | None = None,
    fail_on_missing: bool = True,
    fail_on_stale: bool = True,
) -> dict:
    freshness = assess_archive_freshness(root=root, issue_time_utc=issue_time_utc)
    failures = []
    for source, status in freshness["statuses"].items():
        if fail_on_missing and status["state"] == "missing":
            failures.append({"source": source, "state": status["state"], "warning": status.get("warning")})
        if fail_on_stale and status["state"] == "stale":
            failures.append({"source": source, "state": status["state"], "warning": status.get("warning")})
        if status["state"] == "future_timestamp":
            failures.append({"source": source, "state": status["state"], "warning": status.get("warning")})
    return {"passed": not failures, "failures": failures, "freshness": freshness}
