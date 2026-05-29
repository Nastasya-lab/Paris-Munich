from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from weather_tmax_bot.notifications.telegram import format_outcome_update_message, notify_if_configured
from weather_tmax_bot.operations.pending_truth import run_pending_truth_cron


def main(
    fetch: bool = typer.Option(True),
    as_of_date: str | None = typer.Option(None),
    min_lag_days: int = typer.Option(1),
    update_reports: bool = typer.Option(True),
    notify: bool = typer.Option(True),
    report_path: str = typer.Option("data/reports/daily_outcome_update.json"),
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
    if notify:
        result["telegram_notification"] = notify_if_configured(format_outcome_update_message(result))
    output.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    typer.run(main)
