from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

import typer

from weather_tmax_bot.operations.predict_run import run_prediction_with_optional_refresh
from weather_tmax_bot.operations.run_report import operational_prediction_payload
from weather_tmax_bot.utils.time import parse_issue_time


def main(
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    issue_time: str = typer.Option("now"),
    auto_refresh: bool = typer.Option(False),
    refresh_awc: bool = typer.Option(True),
    refresh_nwp: bool = typer.Option(True),
    output: Path = typer.Option(Path("data/reports/latest_intraday_analysis.json")),
):
    issue = parse_issue_time(issue_time)
    target = date.fromisoformat(target_date) if target_date else issue.astimezone(ZoneInfo("Europe/Berlin")).date()
    result = run_prediction_with_optional_refresh(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        auto_refresh=auto_refresh,
        refresh_awc=refresh_awc,
        refresh_nwp=refresh_nwp,
        log=False,
        mode="intraday_analysis",
    )
    payload = operational_prediction_payload(
        airport=airport,
        target_date_local=target,
        issue_time_utc=issue,
        result=result,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(_summary(payload), indent=2, default=str))
    print(f"Wrote {output}")


def _summary(payload: dict) -> dict:
    components = payload.get("forecast_components", {})
    intraday = components.get("intraday_update", {})
    base = components.get("base_model", {})
    return {
        "airport": payload.get("airport"),
        "target_date_local": payload.get("target_date_local"),
        "issue_time_utc": payload.get("issue_time_utc"),
        "model_version": payload.get("model_version"),
        "base_expected_tmax_c": base.get("expected_tmax_c"),
        "final_expected_tmax_c": payload.get("expected_tmax_c"),
        "base_most_likely_integer_c": base.get("most_likely_integer_c"),
        "final_most_likely_integer_c": payload.get("most_likely_integer_c"),
        "intraday_active": intraday.get("active"),
        "intraday_reason": intraday.get("reason"),
        "peak_passed_probability": intraday.get("peak_passed_probability"),
        "observed_max_so_far_c": intraday.get("observed_max_so_far_c"),
        "last_metar_temp_c": intraday.get("last_metar_temp_c"),
        "drop_from_observed_max_c": intraday.get("drop_from_observed_max_c"),
        "intraday_blend_weight": intraday.get("intraday_blend_weight"),
        "final_probabilities_by_integer_c": payload.get("probabilities_by_integer_c"),
        "final_threshold_probabilities": payload.get("threshold_probabilities"),
        "forecast_quality": payload.get("forecast_quality"),
        "forecast_acceptance": payload.get("forecast_acceptance"),
    }


if __name__ == "__main__":
    typer.run(main)
