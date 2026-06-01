from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_tmax_bot.evaluation.promotion_gate import evaluate_shadow_promotion_gate


def build_outcome_analysis(
    monitoring_path: str | Path = "data/reports/forecast_monitoring.parquet",
    output_json_path: str | Path | None = "data/reports/outcome_analysis.json",
    output_markdown_path: str | Path | None = "docs/outcome_analysis.md",
    variant_monitoring_path: str | Path = "data/reports/forecast_variant_monitoring.parquet",
) -> dict:
    path = Path(monitoring_path)
    if not path.exists():
        analysis = _empty_analysis("forecast_monitoring.parquet not found")
    else:
        monitoring = pd.read_parquet(path)
        analysis = _analysis_from_monitoring(monitoring)
    variant_path = Path(variant_monitoring_path)
    if variant_path.exists():
        variants = pd.read_parquet(variant_path)
        analysis["by_forecast_variant"] = _variant_analysis(variants)
        analysis["champion_vs_shadow"] = _champion_shadow_pair_analysis(variants)
        analysis["by_variant_phase"] = _variant_context_analysis(variants, ["forecast_variant", "forecast_phase"])
        analysis["by_variant_scenario"] = _variant_context_analysis(variants, ["forecast_variant", "scenario_tracking"])
        analysis["by_variant_local_hour"] = _variant_local_hour_analysis(variants)
        analysis["shadow_promotion_gate"] = evaluate_shadow_promotion_gate(variants)
    else:
        analysis["by_forecast_variant"] = []
        analysis["champion_vs_shadow"] = {}
        analysis["by_variant_phase"] = []
        analysis["by_variant_scenario"] = []
        analysis["by_variant_local_hour"] = []
        analysis["shadow_promotion_gate"] = evaluate_shadow_promotion_gate(pd.DataFrame())
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
    lines.extend(["", "## Champion vs shadow", ""])
    lines.extend(_table_lines(analysis.get("by_forecast_variant", [])))
    pair = analysis.get("champion_vs_shadow", {}) or {}
    if pair:
        lines.extend(["", "## Champion vs shadow paired score", ""])
        lines.extend(f"- `{key}`: `{value}`" for key, value in pair.items())
    gate = analysis.get("shadow_promotion_gate", {}) or {}
    lines.extend(["", "## Shadow promotion gate", ""])
    lines.extend(
        [
            f"- `status`: `{gate.get('status')}`",
            f"- `shadow_version`: `{gate.get('shadow_version')}`",
            f"- `recommendation`: `{gate.get('recommendation')}`",
        ]
    )
    if gate.get("checks"):
        lines.extend(f"- `{key}`: `{value}`" for key, value in gate.get("checks", {}).items())
    lines.extend(["", "## By variant phase", ""])
    lines.extend(_table_lines(analysis.get("by_variant_phase", [])))
    lines.extend(["", "## By variant scenario", ""])
    lines.extend(_table_lines(analysis.get("by_variant_scenario", [])))
    lines.extend(["", "## By variant local hour", ""])
    lines.extend(_table_lines(analysis.get("by_variant_local_hour", [])))
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
        "by_forecast_variant": [],
        "champion_vs_shadow": {},
        "by_variant_phase": [],
        "by_variant_scenario": [],
        "by_variant_local_hour": [],
        "shadow_promotion_gate": evaluate_shadow_promotion_gate(pd.DataFrame()),
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
        "by_forecast_variant": [],
        "champion_vs_shadow": {},
        "by_variant_phase": [],
        "by_variant_scenario": [],
        "by_variant_local_hour": [],
        "shadow_promotion_gate": evaluate_shadow_promotion_gate(pd.DataFrame()),
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


def _variant_analysis(variants: pd.DataFrame) -> list[dict]:
    if variants.empty or "forecast_variant" not in variants.columns:
        return []
    df = variants.copy()
    _ensure_variant_optional_columns(df)
    df["abs_error_expected_c"] = df["error_expected_c"].abs()
    grouped = (
        df.groupby("forecast_variant", dropna=False)
        .agg(
            scored_forecasts=("forecast_id", "count"),
            mae_expected=("abs_error_expected_c", "mean"),
            bias_expected=("error_expected_c", "mean"),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
            brier_ge_20=("brier_ge_20", "mean"),
            brier_ge_25=("brier_ge_25", "mean"),
            brier_ge_30=("brier_ge_30", "mean"),
            mean_probability_actual_integer_bin=("probability_actual_integer_bin", "mean"),
            mean_probability_above_actual_integer_bin=("probability_above_actual_integer_bin", "mean"),
            coverage_80=("coverage_80", "mean"),
        )
        .reset_index()
    )
    return _records(grouped)


def _variant_context_analysis(variants: pd.DataFrame, group_cols: list[str]) -> list[dict]:
    if variants.empty or not set(group_cols).issubset(variants.columns):
        return []
    df = variants.copy()
    _ensure_variant_optional_columns(df)
    df["abs_error_expected_c"] = df["error_expected_c"].abs()
    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            scored_forecasts=("forecast_id", "count"),
            mae_expected=("abs_error_expected_c", "mean"),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
            mean_false_upside_probability=("probability_above_actual_integer_bin", "mean"),
            coverage_80=("coverage_80", "mean"),
        )
        .reset_index()
    )
    return _records(grouped)


def _variant_local_hour_analysis(variants: pd.DataFrame) -> list[dict]:
    if variants.empty or "local_issue_hour" not in variants.columns:
        return []
    df = variants.copy()
    df["local_hour_floor"] = pd.to_numeric(df["local_issue_hour"], errors="coerce").fillna(-1).astype(int)
    return _variant_context_analysis(df, ["forecast_variant", "local_hour_floor"])


def _ensure_variant_optional_columns(df: pd.DataFrame) -> None:
    defaults = {
        "probability_above_actual_integer_bin": 0.0,
        "coverage_80": False,
        "forecast_phase": "unknown",
        "scenario_tracking": "unknown",
        "local_issue_hour": -1.0,
    }
    for column, default in defaults.items():
        if column not in df.columns:
            df[column] = default


def _champion_shadow_pair_analysis(variants: pd.DataFrame) -> dict:
    variants = variants.copy()
    if "probability_above_actual_integer_bin" not in variants.columns:
        variants["probability_above_actual_integer_bin"] = 0.0
    required = {"forecast_id", "forecast_variant", "error_expected_c", "nll", "crps", "probability_actual_integer_bin"}
    if variants.empty or not required.issubset(variants.columns):
        return {}
    df = variants[variants["forecast_variant"].isin(["production_champion", "shadow_seasonal_intraday"])].copy()
    if df.empty:
        return {}
    df["abs_error_expected_c"] = df["error_expected_c"].abs()
    pivot = df.pivot_table(
        index="forecast_id",
        columns="forecast_variant",
        values=["abs_error_expected_c", "nll", "crps", "probability_actual_integer_bin", "probability_above_actual_integer_bin"],
        aggfunc="first",
    )
    if ("abs_error_expected_c", "production_champion") not in pivot or ("abs_error_expected_c", "shadow_seasonal_intraday") not in pivot:
        return {}
    paired = pivot.dropna()
    if paired.empty:
        return {}
    champion_abs = paired[("abs_error_expected_c", "production_champion")]
    shadow_abs = paired[("abs_error_expected_c", "shadow_seasonal_intraday")]
    champion_nll = paired[("nll", "production_champion")]
    shadow_nll = paired[("nll", "shadow_seasonal_intraday")]
    champion_crps = paired[("crps", "production_champion")]
    shadow_crps = paired[("crps", "shadow_seasonal_intraday")]
    champion_prob = paired[("probability_actual_integer_bin", "production_champion")]
    shadow_prob = paired[("probability_actual_integer_bin", "shadow_seasonal_intraday")]
    champion_upside = paired[("probability_above_actual_integer_bin", "production_champion")]
    shadow_upside = paired[("probability_above_actual_integer_bin", "shadow_seasonal_intraday")]
    return {
        "paired_forecasts": int(len(paired)),
        "shadow_mae_win_rate": float((shadow_abs < champion_abs).mean()),
        "shadow_nll_win_rate": float((shadow_nll < champion_nll).mean()),
        "shadow_crps_win_rate": float((shadow_crps < champion_crps).mean()),
        "shadow_actual_bin_probability_win_rate": float((shadow_prob > champion_prob).mean()),
        "mean_shadow_minus_champion_abs_error_c": float((shadow_abs - champion_abs).mean()),
        "mean_shadow_minus_champion_nll": float((shadow_nll - champion_nll).mean()),
        "mean_shadow_minus_champion_crps": float((shadow_crps - champion_crps).mean()),
        "mean_shadow_minus_champion_actual_bin_probability": float((shadow_prob - champion_prob).mean()),
        "mean_shadow_minus_champion_false_upside_probability": float((shadow_upside - champion_upside).mean()),
    }


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
