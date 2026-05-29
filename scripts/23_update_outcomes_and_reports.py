from __future__ import annotations

import json
from pathlib import Path

import typer

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report
from weather_tmax_bot.evaluation.monitoring import write_monitoring_report
from weather_tmax_bot.evaluation.outcome_analysis import build_outcome_analysis
from weather_tmax_bot.evaluation.operational_monitoring import build_operational_monitoring_tables
from weather_tmax_bot.evaluation.outcomes import build_forecast_outcome_status, update_forecast_outcomes


def main(report_path: str = typer.Option("data/reports/outcome_update_summary.json")):
    monitoring = update_forecast_outcomes()
    status = build_forecast_outcome_status()
    tables = build_operational_monitoring_tables()
    outcome_analysis = build_outcome_analysis()
    write_monitoring_report()
    write_first_analysis_report()
    summary = {
        "forecast_monitoring_rows": len(monitoring),
        "forecast_outcome_status_rows": len(status),
        "operational_tables": {name: len(table) for name, table in tables.items()},
        "outcome_analysis_status": outcome_analysis["status"],
        "outcome_analysis_rows": outcome_analysis["rows"],
    }
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    typer.run(main)
