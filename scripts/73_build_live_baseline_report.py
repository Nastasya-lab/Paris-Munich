from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from weather_tmax_bot.evaluation.live_baseline import (
    build_live_baseline_report,
    monitoring_from_telegram_exports,
    write_live_baseline_report,
)


def main(
    monitoring_path: str = typer.Option("data/reports/forecast_monitoring.parquet"),
    output_dir: str = typer.Option("data/reports"),
    start_hour: float = typer.Option(10.0),
    end_hour: float = typer.Option(17.0),
    munich_telegram_json: str | None = typer.Option(None),
    paris_telegram_json: str | None = typer.Option(None),
):
    telegram_paths = {}
    if munich_telegram_json:
        telegram_paths["EDDM"] = munich_telegram_json
    if paris_telegram_json:
        telegram_paths["LFPB"] = paris_telegram_json
    if telegram_paths:
        monitoring = monitoring_from_telegram_exports(telegram_paths)
    else:
        path = Path(monitoring_path)
        monitoring = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    report = build_live_baseline_report(monitoring, start_hour=start_hour, end_hour=end_hour)
    paths = write_live_baseline_report(report, output_dir=output_dir)
    print(f"evaluated_rows: {report.metadata['evaluated_rows']}")
    for name, output_path in paths.items():
        print(f"{name}: {output_path}")


if __name__ == "__main__":
    typer.run(main)
