from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_tmax_bot.evaluation.monitoring import build_monitoring_summary

MIN_PRELIMINARY_OUTCOMES = 1
MIN_USEFUL_OUTCOMES = 10
MIN_ROBUST_OUTCOMES = 30


def build_first_analysis(root: str | Path = ".") -> dict:
    root = Path(root)
    monitoring = build_monitoring_summary(root)
    calibration = _read_table(root / "data/reports/calibration_comparison.parquet")
    rolling = _read_table(root / "data/reports/rolling_quantile_summary.parquet")
    holdout = _read_table(root / "data/reports/quantile_holdout_backtest.parquet")
    selected_calibration = _selected_calibration(calibration)
    readiness = _readiness(monitoring, selected_calibration, rolling)
    return {
        "active_model": monitoring.get("active_model", {}),
        "data_volume": {
            "daily_target_rows": monitoring.get("daily_target_rows", 0),
            "training_rows": monitoring.get("training_rows", 0),
            "forecast_log_rows": monitoring.get("forecast_log_rows", 0),
            "forecast_monitoring_rows": monitoring.get("forecast_monitoring_rows", 0),
            "forecast_outcome_status_rows": monitoring.get("forecast_outcome_status_rows", 0),
        },
        "readiness": readiness,
        "selected_calibration": selected_calibration,
        "rolling_summary": rolling,
        "holdout_rows": len(holdout),
        "freshness": monitoring.get("archive_freshness", {}),
        "freshness_gate": monitoring.get("freshness_gate", {}),
        "registry_health": monitoring.get("registry_health", {}),
        "operational_inventory": monitoring.get("operational_forecast_inventory", []),
        "pending_forecasts": monitoring.get("operational_pending_forecasts", []),
        "operational_acceptance": monitoring.get("operational_acceptance", []),
        "outcome_analysis": monitoring.get("outcome_analysis", {}),
        "next_actions": _next_actions(monitoring, readiness),
    }


def write_first_analysis_report(
    json_path: str | Path = "data/reports/first_analysis.json",
    markdown_path: str | Path = "docs/first_analysis.md",
) -> tuple[Path, Path]:
    analysis = build_first_analysis()
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    markdown_output.write_text(format_first_analysis_markdown(analysis), encoding="utf-8")
    return json_output, markdown_output


def format_first_analysis_markdown(analysis: dict) -> str:
    selected = analysis.get("selected_calibration") or {}
    readiness = analysis.get("readiness", {})
    volume = analysis.get("data_volume", {})
    lines = [
        "# First analysis",
        "",
        "This report summarizes whether the current MVP is ready for historical analysis and operational outcome analysis.",
        "",
        "## Current state",
        "",
        f"Active model: `{analysis.get('active_model', {}).get('model_version')}`",
        f"Registry health passed: `{analysis.get('registry_health', {}).get('passed')}`",
        f"Training rows: `{volume.get('training_rows')}`",
        f"Daily target rows: `{volume.get('daily_target_rows')}`",
        f"Logged forecasts: `{volume.get('forecast_log_rows')}`",
        f"Forecasts with outcomes: `{volume.get('forecast_monitoring_rows')}`",
        f"Forecast outcome status rows: `{volume.get('forecast_outcome_status_rows')}`",
        f"Operational outcome stage: `{readiness.get('operational_outcome_stage')}`",
        "",
        "## Readiness",
        "",
    ]
    lines.extend(f"- `{key}`: `{value}`" for key, value in readiness.items())
    lines.extend(
        [
            "",
            "## Selected calibration",
            "",
            f"Variant: `{selected.get('forecast_variant')}`",
            f"Rows: `{selected.get('rows')}`",
            f"NLL: `{selected.get('mean_nll')}`",
            f"CRPS: `{selected.get('mean_crps')}`",
            f"Coverage 50/80/90: `{selected.get('coverage_50')}` / `{selected.get('coverage_80')}` / `{selected.get('coverage_90')}`",
            "",
            "## Operational outcomes",
            "",
            f"Outcome rows: `{readiness.get('operational_outcome_rows')}`",
            f"Minimum for first analysis: `{MIN_PRELIMINARY_OUTCOMES}`",
            f"Minimum for useful sample: `{MIN_USEFUL_OUTCOMES}`",
            f"Minimum for robust monitoring: `{MIN_ROBUST_OUTCOMES}`",
            "",
            "### Acceptance breakdown",
            "",
        ]
    )
    lines.extend(_table_lines(analysis.get("operational_acceptance", [])))
    lines.extend(
        [
            "",
            "## Next actions",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in analysis.get("next_actions", []))
    lines.append("")
    return "\n".join(lines)


def _read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records"))


def _selected_calibration(rows: list[dict]) -> dict:
    for row in rows:
        if row.get("selected_for_production"):
            return row
    return rows[0] if rows else {}


def _readiness(monitoring: dict, selected_calibration: dict, rolling: list[dict]) -> dict:
    leakage_rows = monitoring.get("leakage_audit", [])
    leakage_passed = all(int(row.get("violations", 0)) == 0 for row in leakage_rows if row.get("check") != "rows")
    outcome_rows = int(monitoring.get("forecast_monitoring_rows", 0))
    return {
        "historical_backtest_ready": bool(selected_calibration and rolling),
        "calibration_ready": bool(selected_calibration),
        "production_predict_ready": bool(monitoring.get("registry_health", {}).get("passed")),
        "leakage_audit_passed": leakage_passed,
        "operational_outcome_analysis_ready": outcome_rows >= MIN_PRELIMINARY_OUTCOMES,
        "operational_outcome_useful_sample": outcome_rows >= MIN_USEFUL_OUTCOMES,
        "operational_outcome_robust_sample": outcome_rows >= MIN_ROBUST_OUTCOMES,
        "operational_outcome_rows": outcome_rows,
        "operational_outcome_stage": _operational_outcome_stage(outcome_rows),
        "freshness_gate_passed": bool(monitoring.get("freshness_gate", {}).get("passed")),
    }


def _next_actions(monitoring: dict, readiness: dict) -> list[str]:
    actions = []
    if not readiness["operational_outcome_analysis_ready"]:
        pending = monitoring.get("operational_pending_forecasts", [])
        if pending:
            actions.append("Some forecasts are logged but still pending DWD truth; run outcome update after their target dates are available.")
        else:
            actions.append("Log forecasts until at least one target date has completed, then update DWD truth and run outcome monitoring.")
    elif not readiness["operational_outcome_useful_sample"]:
        actions.append(
            f"Operational outcome analysis has started, but only {readiness['operational_outcome_rows']} scored forecast(s) are available; treat metrics as smoke-test evidence until at least {MIN_USEFUL_OUTCOMES} outcomes exist."
        )
    elif not readiness["operational_outcome_robust_sample"]:
        actions.append(
            f"Operational metrics are becoming useful, but reliability/calibration conclusions remain preliminary until at least {MIN_ROBUST_OUTCOMES} outcomes exist."
        )
    if not readiness["freshness_gate_passed"]:
        actions.append("Refresh stale operational archives, especially METAR/TAF/NWP sources flagged by the freshness gate.")
    if readiness["historical_backtest_ready"]:
        actions.append("Review `docs/backtest_results.md`, PIT/reliability plots, and rolling summaries for the first model-quality analysis.")
    if int(monitoring.get("nwp_archive_rows", 0)) < 30:
        actions.append("Continue accumulating forecast-as-issued NWP before drawing strong NWP backtest conclusions.")
    return actions or ["No immediate blockers detected."]


def _operational_outcome_stage(outcome_rows: int) -> str:
    if outcome_rows <= 0:
        return "pending"
    if outcome_rows < MIN_USEFUL_OUTCOMES:
        return "first_outcome"
    if outcome_rows < MIN_ROBUST_OUTCOMES:
        return "useful_sample"
    return "robust_sample"


def _table_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows."]
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return lines
