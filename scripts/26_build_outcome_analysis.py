from __future__ import annotations

import typer

from weather_tmax_bot.evaluation.outcome_analysis import build_outcome_analysis


def main(
    monitoring_path: str = typer.Option("data/reports/forecast_monitoring.parquet"),
    json_path: str = typer.Option("data/reports/outcome_analysis.json"),
    markdown_path: str = typer.Option("docs/outcome_analysis.md"),
):
    analysis = build_outcome_analysis(
        monitoring_path=monitoring_path,
        output_json_path=json_path,
        output_markdown_path=markdown_path,
    )
    print(f"Outcome analysis status: {analysis['status']}, rows: {analysis['rows']}")


if __name__ == "__main__":
    typer.run(main)
