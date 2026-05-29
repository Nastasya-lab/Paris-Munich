from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from weather_tmax_bot.operations.pending_truth import run_pending_truth_cron


def main(
    fetch: bool = typer.Option(False),
    as_of_date: str | None = typer.Option(None),
    min_lag_days: int = typer.Option(1),
    update_reports: bool = typer.Option(True),
    fail_if_ready: bool = typer.Option(False),
    report_path: str = typer.Option("data/reports/pending_truth_cron.json"),
):
    as_of = None if as_of_date is None else date.fromisoformat(as_of_date)
    result = run_pending_truth_cron(
        fetch=fetch,
        as_of_date=as_of,
        min_lag_days=min_lag_days,
        update_reports=update_reports,
    )
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    if fail_if_ready and result["status"]["action_required"] and not fetch:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
