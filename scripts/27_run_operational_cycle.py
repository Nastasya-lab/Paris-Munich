from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from weather_tmax_bot.operations.workflow import run_operational_cycle
from weather_tmax_bot.utils.time import parse_issue_time


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str = typer.Option(...),
    issue_time: str = typer.Option("now"),
    auto_refresh: bool = typer.Option(True),
    refresh_awc: bool = typer.Option(True),
    refresh_nwp: bool = typer.Option(True),
    log: bool = typer.Option(True),
    update_reports: bool = typer.Option(True),
    require_ok: bool = typer.Option(False),
    prediction_report_path: str = typer.Option("data/reports/latest_operational_prediction.json"),
    cycle_report_path: str = typer.Option("data/reports/latest_operational_cycle.json"),
):
    issue = parse_issue_time(issue_time)
    target = date.fromisoformat(target_date)
    summary = run_operational_cycle(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=log,
        update_reports=update_reports,
        report_path=prediction_report_path,
        allow_issue_time_advance=issue_time in (None, "now"),
    )
    output = Path(cycle_report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {prediction_report_path}")
    print(f"Wrote {output}")
    if require_ok and not summary["accepted"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
