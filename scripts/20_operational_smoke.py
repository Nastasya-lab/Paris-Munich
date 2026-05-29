from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import typer

from weather_tmax_bot.evaluation.smoke import run_operational_smoke


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    issue_time: str | None = typer.Option(None),
    fail_on_stale: bool = typer.Option(False),
    report_path: str = typer.Option("data/reports/operational_smoke.json"),
):
    target = None if target_date is None else date.fromisoformat(target_date)
    issue = _parse_issue_time(issue_time)
    result = run_operational_smoke(airport=airport, target_date=target, issue_time_utc=issue, fail_on_stale=fail_on_stale)
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
