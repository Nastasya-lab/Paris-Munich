from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_tmax_bot.bot.lineage import model_info
from weather_tmax_bot.models.registry_health import registry_health
from weather_tmax_bot.temporal.freshness import assess_archive_freshness
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def build_monitoring_summary(root: str | Path = ".") -> dict:
    root = Path(root)
    summary = {
        "model_artifacts": sorted(p.name for p in (root / "data/models").glob("*")),
        "forecast_log_rows": _jsonl_count(root / "data/logs/forecast_log.jsonl"),
        "forecast_monitoring_rows": _parquet_rows(root / "data/reports/forecast_monitoring.parquet"),
        "forecast_variant_monitoring_rows": _parquet_rows(root / "data/reports/forecast_variant_monitoring.parquet"),
        "forecast_outcome_status_rows": _parquet_rows(root / "data/reports/forecast_outcome_status.parquet"),
        "nwp_archive_rows": _parquet_rows(root / "data/forecasts/open_meteo_archive.parquet"),
        "awc_metar_live_rows": _parquet_rows(root / "data/forecasts/awc_metar_live_EDDM.parquet"),
        "awc_taf_live_rows": _parquet_rows(root / "data/forecasts/awc_taf_live_EDDM.parquet"),
        "training_rows": _parquet_rows(root / "data/processed/training_dataset.parquet"),
        "daily_target_rows": _parquet_rows(root / "data/processed/daily_target.parquet"),
        "active_model": model_info(root / "data/models")["active_model"],
        "registry_health": registry_health(root / "data/models", root / "data/models/quantile_mvp.joblib"),
        "archive_freshness": assess_archive_freshness(root),
        "freshness_gate": evaluate_freshness_gate(root, fail_on_missing=False, fail_on_stale=True),
        "latest_retraining_report": _read_json(root / "data/reports/retraining_report.json"),
        "outcome_analysis": _read_json(root / "data/reports/outcome_analysis.json"),
        "shadow_promotion_gate": _read_json(root / "data/reports/shadow_promotion_gate.json"),
        "safe_blend_promotion_gate": _read_json(root / "data/reports/safe_blend_promotion_gate.json"),
        "leakage_audit": _read_leakage(root / "data/reports/leakage_audit.parquet"),
        "calibration_comparison": _read_table(root / "data/reports/calibration_comparison.parquet"),
        "rolling_summary": _read_table(root / "data/reports/rolling_quantile_summary.parquet"),
        "operational_by_model": _read_table(root / "data/reports/operational_by_model.parquet"),
        "operational_source_mismatch": _read_table(root / "data/reports/operational_source_mismatch.parquet"),
        "operational_availability": _read_table(root / "data/reports/operational_availability.parquet"),
        "operational_acceptance": _read_table(root / "data/reports/operational_acceptance.parquet"),
        "operational_forecast_inventory": _read_table(root / "data/reports/operational_forecast_inventory.parquet"),
        "operational_pending_forecasts": _read_table(root / "data/reports/operational_pending_forecasts.parquet"),
    }
    return summary


def write_monitoring_report(path: str | Path = "docs/monitoring_report.md") -> Path:
    summary = build_monitoring_summary()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_format_markdown(summary), encoding="utf-8")
    return output


def _format_markdown(summary: dict) -> str:
    lines = [
        "# Monitoring report",
        "",
        f"Daily target rows: `{summary['daily_target_rows']}`",
        f"Training rows: `{summary['training_rows']}`",
        f"NWP archive rows: `{summary['nwp_archive_rows']}`",
        f"AWC live METAR rows: `{summary['awc_metar_live_rows']}`",
        f"AWC live TAF rows: `{summary['awc_taf_live_rows']}`",
        f"Forecast log rows: `{summary['forecast_log_rows']}`",
        f"Forecast monitoring rows with outcomes: `{summary['forecast_monitoring_rows']}`",
        f"Forecast variant monitoring rows: `{summary['forecast_variant_monitoring_rows']}`",
        f"Forecast outcome status rows: `{summary['forecast_outcome_status_rows']}`",
        "",
        "## Active model",
        "",
        f"Model version: `{summary['active_model'].get('model_version')}`",
        f"Calibrator version: `{summary['active_model'].get('calibrator_version')}`",
        f"Model path: `{summary['active_model'].get('model_path')}`",
        f"Calibrator path: `{summary['active_model'].get('calibrator_path')}`",
        f"Model exists: `{summary['active_model'].get('model_exists')}`",
        f"Calibrator exists: `{summary['active_model'].get('calibrator_exists')}`",
        "",
        "## Registry health",
        "",
        f"Passed: `{summary['registry_health'].get('passed')}`",
        "",
        "## Archive freshness",
        "",
        f"METAR: `{summary['archive_freshness']['statuses']['metar']['state']}`",
        f"TAF: `{summary['archive_freshness']['statuses']['taf']['state']}`",
        f"NWP: `{summary['archive_freshness']['statuses']['nwp']['state']}`",
        f"Freshness gate passed: `{summary['freshness_gate'].get('passed')}`",
        "",
        "## Latest retraining",
        "",
        f"Model version: `{summary['latest_retraining_report'].get('model_version')}`",
        f"Promoted: `{summary['latest_retraining_report'].get('promoted')}`",
        f"Rows: `{summary['latest_retraining_report'].get('rows')}`",
        "",
        "## Outcome analysis",
        "",
        f"Status: `{summary['outcome_analysis'].get('status')}`",
        f"Rows: `{summary['outcome_analysis'].get('rows')}`",
        "",
        "## Shadow promotion gate",
        "",
        f"Status: `{summary['shadow_promotion_gate'].get('status')}`",
        f"Shadow version: `{summary['shadow_promotion_gate'].get('shadow_version')}`",
        f"Recommendation: `{summary['shadow_promotion_gate'].get('recommendation')}`",
        "",
        "## Safe blended shadow promotion gate",
        "",
        f"Status: `{summary['safe_blend_promotion_gate'].get('status')}`",
        f"Shadow version: `{summary['safe_blend_promotion_gate'].get('shadow_version')}`",
        f"Recommendation: `{summary['safe_blend_promotion_gate'].get('recommendation')}`",
        "",
        "## Model artifacts",
        "",
    ]
    lines.extend(f"- `{name}`" for name in summary["model_artifacts"])
    lines.extend(["", "## Leakage audit", ""])
    for row in summary["leakage_audit"]:
        lines.append(f"- `{row['check']}`: `{row['violations']}`")
    lines.extend(["", "## Calibration comparison", ""])
    lines.extend(_table_lines(summary["calibration_comparison"]))
    lines.extend(["", "## Rolling summary", ""])
    lines.extend(_table_lines(summary["rolling_summary"]))
    lines.extend(["", "## Operational by model", ""])
    lines.extend(_table_lines(summary["operational_by_model"]))
    lines.extend(["", "## Operational source mismatch", ""])
    lines.extend(_table_lines(summary["operational_source_mismatch"]))
    lines.extend(["", "## Operational availability", ""])
    lines.extend(_table_lines(summary["operational_availability"]))
    lines.extend(["", "## Operational acceptance", ""])
    lines.extend(_table_lines(summary["operational_acceptance"]))
    lines.extend(["", "## Operational forecast inventory", ""])
    lines.extend(_table_lines(summary["operational_forecast_inventory"]))
    lines.extend(["", "## Operational pending forecasts", ""])
    lines.extend(_table_lines(summary["operational_pending_forecasts"]))
    lines.append("")
    return "\n".join(lines)


def _table_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows."]
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return lines


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_parquet(path))


def _read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(pd.read_parquet(path).to_json(orient="records"))


def _read_leakage(path: Path) -> list[dict]:
    return _read_table(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
