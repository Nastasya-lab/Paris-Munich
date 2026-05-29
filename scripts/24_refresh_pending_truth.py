from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report
from weather_tmax_bot.evaluation.monitoring import write_monitoring_report
from weather_tmax_bot.operations.truth_refresh import refresh_pending_truth


def main(
    airport: str = typer.Option("EDDM"),
    station_id: str = typer.Option("01262"),
    fetch: bool = typer.Option(False),
    as_of_date: str | None = typer.Option(None),
    min_lag_days: int = typer.Option(1),
    report_path: str = typer.Option("data/reports/pending_truth_refresh.json"),
):
    as_of = None if as_of_date is None else date.fromisoformat(as_of_date)
    summary = refresh_pending_truth(
        airport=airport,
        station_id=station_id,
        fetch=fetch,
        as_of_date=as_of,
        min_lag_days=min_lag_days,
    )
    write_monitoring_report()
    write_first_analysis_report()
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    typer.run(main)
