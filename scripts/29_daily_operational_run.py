from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import typer

from weather_tmax_bot.operations.workflow import run_operational_cycle
from weather_tmax_bot.utils.time import parse_issue_time


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    issue_time: str = typer.Option("now"),
    require_ok: bool = typer.Option(True),
    auto_refresh: bool = typer.Option(True),
    refresh_awc: bool = typer.Option(True),
    refresh_nwp: bool = typer.Option(True),
    update_reports: bool = typer.Option(True),
    report_path: str = typer.Option("data/reports/daily_operational_run.json"),
):
    issue = parse_issue_time(issue_time)
    target = datetime.now(ZoneInfo("Europe/Berlin")).date() if target_date is None else datetime.fromisoformat(target_date).date()
    summary = run_operational_cycle(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=True,
        update_reports=update_reports,
        mode="railway_daily_operational_run",
    )
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    if require_ok and not summary["accepted"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
