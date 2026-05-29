from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_outcome_analysis(
    monitoring_path: str | Path = "data/reports/forecast_monitoring.parquet",
    output_json_path: str | Path | None = "data/reports/outcome_analysis.json",
    output_markdown_path: str | Path | None = "docs/outcome_analysis.md",
) -> dict:
    path = Path(monitoring_path)
    if not path.exists():
        analysis = _empty_analysis("forecast_monitoring.parquet not found")
    else:
        monitoring = pd.read_parquet(path)
        analysis = _analysis_from_monitoring(monitoring)
    if output_json_path is not None:
        output = Path(output_json_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(analysis, indent=2, default=str), encoding="utf-8")
    if output_markdown_path is not None:
        output = Path(output_markdown_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(format_outcome_analysis_markdown(analysis), encoding="utf-8")
    return analysis


def format_outcome_analysis_markdown(analysis: dict) -> str:
    lines = [
        "# Outcome analysis",
        "",
        f"Status: `{analysis['status']}`",
        f"Rows: `{analysis['rows']}`",
        "",
        "## Overall",
        "",
    ]
    overall = analysis.get("overall", {})
    if overall:
        lines.extend(f"- `{key}`: `{value}`" for key, value in overall.items())
    else:
        lines.append("No scored forecasts yet.")
    lines.extend(["", "## By model", ""])
    lines.extend(_table_lines(analysis.get("by_model", [])))
    lines.extend(["", "## By forecast quality", ""])
    lines.extend(_table_lines(analysis.get("by_quality", [])))
    lines.extend(["", "## By forecast acceptance", ""])
    lines.extend(_table_lines(analysis.get("by_acceptance", [])))
    lines.extend(["", "## By source mismatch", ""])
    lines.extend(_table_lines(analysis.get("by_source_mismatch", [])))
    lines.extend(["", "## By METAR source compatibility", ""])
    lines.extend(_table_lines(analysis.get("by_metar_compatibility", [])))
    lines.extend(["", "## By TAF source compatibility", ""])
    lines.extend(_table_lines(analysis.get("by_taf_compatibility", [])))
    lines.extend(["", "## Worst forecasts by CRPS", ""])
    lines.extend(_table_lines(analysis.get("worst_by_crps", [])))
    lines.append("")
    return "\n".join(lines)


def _analysis_from_monitoring(monitoring: pd.DataFrame) -> dict:
    if monitoring.empty:
        return _empty_analysis("no scored forecasts yet")
    df = monitoring.copy()
    df["abs_error_expected_c"] = df["error_expected_c"].abs()
    if "forecast_quality_status" not in df.columns:
        df["forecast_quality_status"] = "unknown"
    if "forecast_accepted" not in df.columns:
        df["forecast_accepted"] = "unknown"
    else:
        df["forecast_accepted"] = df["forecast_accepted"].map(_acceptance_label)
    if "metar_source_mismatch" not in df.columns:
        df["metar_source_mismatch"] = False
    if "taf_source_mismatch" not in df.columns:
        df["taf_source_mismatch"] = False
    if "metar_source_compatibility_status" not in df.columns:
        df["metar_source_compatibility_status"] = "unknown"
    if "taf_source_compatibility_status" not in df.columns:
        df["taf_source_compatibility_status"] = "unknown"
    df["any_source_mismatch"] = df["metar_source_mismatch"].fillna(False) | df["taf_source_mismatch"].fillna(False)
    return {
        "status": "ready",
        "rows": len(df),
        "overall": _metric_summary(df),
        "by_model": _group_summary(df, ["model_version"]),
        "by_quality": _group_summary(df, ["forecast_quality_status"]),
        "by_acceptance": _group_summary(df, ["forecast_accepted"]),
        "by_source_mismatch": _group_summary(df, ["any_source_mismatch"]),
        "by_metar_compatibility": _group_summary(df, ["metar_source_compatibility_status"]),
        "by_taf_compatibility": _group_summary(df, ["taf_source_compatibility_status"]),
        "worst_by_crps": _records(
            df.sort_values("crps", ascending=False).head(10)[
                [
                    "forecast_id",
                    "model_version",
                    "target_date_local",
                    "actual_tmax_c",
                    "expected_tmax_c",
                    "error_expected_c",
                    "nll",
                    "crps",
                    "forecast_quality_status",
                ]
            ]
        ),
        "best_by_crps": _records(df.sort_values("crps", ascending=True).head(10)),
    }


def _empty_analysis(reason: str) -> dict:
    return {
        "status": "pending",
        "reason": reason,
        "rows": 0,
        "overall": {},
        "by_model": [],
        "by_quality": [],
        "by_acceptance": [],
        "by_source_mismatch": [],
        "by_metar_compatibility": [],
        "by_taf_compatibility": [],
        "worst_by_crps": [],
        "best_by_crps": [],
    }


def _metric_summary(df: pd.DataFrame) -> dict:
    return {
        "forecasts": int(len(df)),
        "mae_expected": float(df["abs_error_expected_c"].mean()),
        "bias_expected": float(df["error_expected_c"].mean()),
        "mean_nll": float(df["nll"].mean()),
        "mean_crps": float(df["crps"].mean()),
        "brier_ge_20": float(df["brier_ge_20"].mean()),
        "brier_ge_25": float(df["brier_ge_25"].mean()),
        "brier_ge_30": float(df["brier_ge_30"].mean()),
    }


def _group_summary(df: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            forecasts=("forecast_id", "count"),
            mae_expected=("abs_error_expected_c", "mean"),
            bias_expected=("error_expected_c", "mean"),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
    )
    return _records(grouped)


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _acceptance_label(value) -> str:
    if pd.isna(value):
        return "unknown"
    return "accepted" if bool(value) else "rejected"


def _table_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["No rows."]
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
    return lines
