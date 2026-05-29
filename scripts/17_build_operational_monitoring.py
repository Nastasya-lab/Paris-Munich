from __future__ import annotations

import typer

from weather_tmax_bot.evaluation.operational_monitoring import build_operational_monitoring_tables


def main(
    monitoring_path: str = typer.Option("data/reports/forecast_monitoring.parquet"),
    forecast_log_path: str = typer.Option("data/logs/forecast_log.jsonl"),
    output_dir: str = typer.Option("data/reports"),
):
    tables = build_operational_monitoring_tables(
        monitoring_path=monitoring_path,
        forecast_log_path=forecast_log_path,
        output_dir=output_dir,
    )
    for name, table in tables.items():
        print(f"{name}: {len(table)} rows")


if __name__ == "__main__":
    typer.run(main)
