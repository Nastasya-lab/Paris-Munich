from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def main(
    root: str = typer.Option("."),
    issue_time: str | None = typer.Option(None),
    fail_on_missing: bool = typer.Option(True),
    fail_on_stale: bool = typer.Option(True),
    report_path: str = typer.Option("data/reports/data_freshness_health.json"),
):
    issue = _parse_issue_time(issue_time)
    result = evaluate_freshness_gate(root=root, issue_time_utc=issue, fail_on_missing=fail_on_missing, fail_on_stale=fail_on_stale)
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if not result["passed"]:
        raise typer.Exit(code=1)


def _parse_issue_time(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


if __name__ == "__main__":
    typer.run(main)
