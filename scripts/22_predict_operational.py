from __future__ import annotations

from datetime import date

import typer

from weather_tmax_bot.bot.formatter import format_prediction
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.run_report import operational_prediction_payload, write_operational_prediction_report
from weather_tmax_bot.utils.time import parse_issue_time


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str = typer.Option(...),
    issue_time: str = typer.Option("now"),
    auto_refresh: bool = typer.Option(True),
    refresh_awc: bool = typer.Option(True),
    refresh_nwp: bool = typer.Option(True),
    log: bool = typer.Option(True),
    require_ok: bool = typer.Option(False),
    report_path: str = typer.Option("data/reports/latest_operational_prediction.json"),
):
    issue = parse_issue_time(issue_time)
    target = date.fromisoformat(target_date)
    result = run_prediction_with_optional_refresh(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=log,
        mode="operational_script",
    )
    payload = operational_prediction_payload(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        result=result,
    )
    output = write_operational_prediction_report(payload, report_path)
    print(
        format_prediction(
            airport=airport,
            target_date=target,
            issue_time_utc=issue,
            dist=result["distribution"],
            model_version=result["metadata"]["model_version"],
            data_lineage=result["data_lineage"],
            forecast_quality=result["forecast_quality"],
            forecast_acceptance=result["forecast_acceptance"],
            warnings=result["warnings"],
        )
    )
    print(f"\nWrote {output}")
    if require_ok and not result["forecast_acceptance"]["accepted"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
