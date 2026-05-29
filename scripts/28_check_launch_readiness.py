from __future__ import annotations

import json
from pathlib import Path

import typer

from weather_tmax_bot.operations.launch_readiness import assess_launch_readiness


def main(
    report_path: str = typer.Option("data/reports/launch_readiness.json"),
    require_forward_ready: bool = typer.Option(False),
    require_outcome_ready: bool = typer.Option(False),
):
    readiness = assess_launch_readiness()
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
