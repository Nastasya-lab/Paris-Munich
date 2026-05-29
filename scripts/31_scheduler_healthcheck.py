from __future__ import annotations

import json
from pathlib import Path

import typer

from weather_tmax_bot.notifications.telegram import format_healthcheck_message, notify_if_configured
from weather_tmax_bot.operations.launch_readiness import assess_launch_readiness


def main(
    require_forward_ready: bool = typer.Option(True),
    require_outcome_ready: bool = typer.Option(False),
    notify_on_success: bool = typer.Option(False),
    notify_on_failure: bool = typer.Option(True),
    report_path: str = typer.Option("data/reports/scheduler_healthcheck.json"),
):
    readiness = assess_launch_readiness()
    should_notify = (readiness["ready_for_forward_ops"] and notify_on_success) or (not readiness["ready_for_forward_ops"] and notify_on_failure)
    if should_notify:
        readiness["telegram_notification"] = notify_if_configured(format_healthcheck_message(readiness))
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(readiness, indent=2, default=str), encoding="utf-8")
    print(json.dumps(readiness, indent=2, default=str))
    if require_forward_ready and not readiness["ready_for_forward_ops"]:
        raise typer.Exit(code=1)
    if require_outcome_ready and not readiness["ready_for_outcome_monitoring"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
