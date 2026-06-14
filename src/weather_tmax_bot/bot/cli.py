from __future__ import annotations

from datetime import date

import typer

from weather_tmax_bot.bot.formatter import format_prediction
from weather_tmax_bot.bot.status import format_status
from weather_tmax_bot.evaluation.first_analysis import format_first_analysis_markdown, build_first_analysis
from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.utils.time import parse_issue_time

app = typer.Typer()


@app.callback()
def main():
    """Weather Tmax Bot command line interface."""


@app.command()
def predict(
    airport: str = typer.Option("EDDM"),
    target_date: str = typer.Option(...),
    issue_time: str = typer.Option("now"),
    log: bool = typer.Option(True),
    auto_refresh: bool = typer.Option(False),
    refresh_awc: bool = typer.Option(True),
    refresh_nwp: bool = typer.Option(True),
    require_ok: bool = typer.Option(False),
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
        mode="cli",
        allow_issue_time_advance=issue_time in (None, "now"),
    )
    display_issue = result.get("issue_time_utc", issue)
    refresh_summary = result.get("refresh_summary")
    if refresh_summary:
        typer.echo(f"Refresh summary: {refresh_summary}")
    typer.echo(
        format_prediction(
            airport=airport,
            target_date=target,
            issue_time_utc=display_issue,
            dist=result["distribution"],
            model_version=result["metadata"]["model_version"],
            data_lineage=result["data_lineage"],
            forecast_quality=result["forecast_quality"],
            forecast_acceptance=result["forecast_acceptance"],
            forecast_components=result["feature_snapshot"].get("forecast_components", {}),
            warnings=result["warnings"],
        )
    )
    if require_ok and not result["forecast_acceptance"]["accepted"]:
        raise typer.Exit(code=1)


@app.command()
def status():
    """Print operational health and monitoring status."""
    typer.echo(format_status())


@app.command()
def analyze():
    """Print the first-analysis readiness summary."""
    typer.echo(format_first_analysis_markdown(build_first_analysis()))


if __name__ == "__main__":
    app()
