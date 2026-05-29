from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report
from weather_tmax_bot.evaluation.monitoring import write_monitoring_report
from weather_tmax_bot.evaluation.operational_monitoring import build_operational_monitoring_tables
from weather_tmax_bot.operations.refresh import refresh_operational_data


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    skip_awc: bool = typer.Option(False),
    skip_nwp: bool = typer.Option(False),
    fail_on_stale: bool = typer.Option(False),
    report_path: str = typer.Option("data/reports/prepare_operational_run.json"),
):
    target = date.fromisoformat(target_date) if target_date else date.today()
    summary = refresh_operational_data(
        airport=airport,
        target_date_local=target,
        refresh_awc=not skip_awc,
        refresh_nwp=not skip_nwp,
    )
    build_operational_monitoring_tables()
    write_monitoring_report()
    write_first_analysis_report()
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    if fail_on_stale and not summary["freshness_gate"]["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
