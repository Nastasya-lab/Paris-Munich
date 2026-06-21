from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import typer


BASE_VARIANT = "base_metar_intraday"
ENHANCED_VARIANT = "enhanced_metar_intraday"


def main(
    summary_path: str = typer.Option("data/reports/lfpb_intraday_enhanced_feature_comparison_summary.csv"),
    by_hour_path: str = typer.Option("data/reports/lfpb_intraday_enhanced_feature_comparison_by_hour.csv"),
    output_json: str = typer.Option("data/reports/lfpb_enhanced_intraday_promotion_gate.json"),
    output_md: str = typer.Option("data/reports/lfpb_enhanced_intraday_promotion_gate.md"),
):
    report = evaluate_gate(pd.read_csv(summary_path), pd.read_csv(by_hour_path))
    Path(output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(output_md).write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def evaluate_gate(summary: pd.DataFrame, by_hour: pd.DataFrame) -> dict:
    base = _variant_row(summary, BASE_VARIANT)
    enhanced = _variant_row(summary, ENHANCED_VARIANT)
    deltas = _deltas(base, enhanced)
    hourly = _hourly_deltas(by_hour)
    criteria = {
        "mae_improves_at_least_0_03c": deltas["mae_expected"] <= -0.03,
        "rmse_improves": deltas["rmse_expected"] < 0,
        "nll_not_worse_more_than_0_02": deltas["mean_nll"] <= 0.02,
        "crps_improves": deltas["mean_crps"] < 0,
        "coverage_not_worse_more_than_2pp": deltas["coverage_80"] >= -0.02,
        "bias_abs_not_worse_more_than_0_05c": abs(float(enhanced["bias_expected"])) <= abs(float(base["bias_expected"])) + 0.05,
        "hourly_mae_regressions_limited": hourly["bad_mae_regression_hours"] <= 1,
        "morning_10_mae_improves": hourly["hour_10_mae_delta"] < 0,
        "midday_12_mae_not_worse": hourly["hour_12_mae_delta"] <= 0.03,
    }
    passed = all(criteria.values())
    if passed:
        decision = "promote_to_live_shadow"
        reason = "Enhanced intraday features improve point and probabilistic metrics with limited hourly regression."
    elif criteria["mae_improves_at_least_0_03c"] and criteria["crps_improves"]:
        decision = "keep_as_shadow_candidate"
        reason = "Enhanced features are promising, but at least one promotion gate is not clean enough."
    else:
        decision = "do_not_promote"
        reason = "Enhanced features do not clear the promotion gate."
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "candidate": ENHANCED_VARIANT,
        "baseline": BASE_VARIANT,
        "decision": decision,
        "reason": reason,
        "criteria": criteria,
        "deltas_enhanced_minus_base": deltas,
        "hourly_diagnostics": hourly,
        "summary": {
            "baseline": _clean_row(base),
            "candidate": _clean_row(enhanced),
        },
    }


def _variant_row(frame: pd.DataFrame, variant: str) -> pd.Series:
    rows = frame[frame["model_variant"] == variant]
    if rows.empty:
        raise ValueError(f"missing variant in summary: {variant}")
    return rows.iloc[0]


def _deltas(base: pd.Series, enhanced: pd.Series) -> dict:
    metrics = [
        "mae_expected",
        "rmse_expected",
        "bias_expected",
        "mean_nll",
        "mean_crps",
        "brier_upside_ge_1c",
        "brier_upside_ge_2c",
        "brier_upside_ge_3c",
        "coverage_80",
    ]
    return {metric: float(enhanced[metric]) - float(base[metric]) for metric in metrics}


def _hourly_deltas(by_hour: pd.DataFrame) -> dict:
    base = by_hour[by_hour["model_variant"] == BASE_VARIANT].set_index("local_issue_hour")
    enhanced = by_hour[by_hour["model_variant"] == ENHANCED_VARIANT].set_index("local_issue_hour")
    common_hours = sorted(set(base.index).intersection(set(enhanced.index)))
    rows = []
    bad_mae_regression_hours = 0
    for hour in common_hours:
        mae_delta = float(enhanced.loc[hour, "mae_expected"]) - float(base.loc[hour, "mae_expected"])
        nll_delta = float(enhanced.loc[hour, "mean_nll"]) - float(base.loc[hour, "mean_nll"])
        rows.append({"local_issue_hour": int(hour), "mae_delta": mae_delta, "nll_delta": nll_delta})
        if mae_delta > 0.03:
            bad_mae_regression_hours += 1
    by_hour_deltas = {str(row["local_issue_hour"]): row for row in rows}
    return {
        "bad_mae_regression_hours": int(bad_mae_regression_hours),
        "hour_10_mae_delta": _hour_metric(rows, 10, "mae_delta"),
        "hour_12_mae_delta": _hour_metric(rows, 12, "mae_delta"),
        "deltas_by_hour": by_hour_deltas,
    }


def _hour_metric(rows: list[dict], hour: int, metric: str) -> float | None:
    for row in rows:
        if row["local_issue_hour"] == hour:
            return float(row[metric])
    return None


def _clean_row(row: pd.Series) -> dict:
    return {key: _json_value(value) for key, value in row.to_dict().items()}


def _json_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _markdown(report: dict) -> str:
    lines = [
        "# LFPB Enhanced Intraday Promotion Gate",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Reason: {report['reason']}",
        f"- Candidate: `{report['candidate']}`",
        f"- Baseline: `{report['baseline']}`",
        "",
        "## Criteria",
        "",
    ]
    for key, value in report["criteria"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Deltas Enhanced Minus Base", ""])
    for key, value in report["deltas_enhanced_minus_base"].items():
        lines.append(f"- `{key}`: `{value:.6f}`")
    lines.extend(["", "## Hourly Diagnostics", ""])
    lines.append(f"- `bad_mae_regression_hours`: `{report['hourly_diagnostics']['bad_mae_regression_hours']}`")
    lines.append(f"- `hour_10_mae_delta`: `{report['hourly_diagnostics']['hour_10_mae_delta']}`")
    lines.append(f"- `hour_12_mae_delta`: `{report['hourly_diagnostics']['hour_12_mae_delta']}`")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    typer.run(main)
