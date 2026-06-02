from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CHAMPION_VARIANT = "production_champion"
SHADOW_VARIANT = "shadow_seasonal_intraday"
PHASE_AWARE_SHADOW_VERSION = "phase_aware_intraday_challenger_v3"


def evaluate_shadow_promotion_gate(
    variants: pd.DataFrame,
    *,
    shadow_variant: str = SHADOW_VARIANT,
    shadow_version: str = PHASE_AWARE_SHADOW_VERSION,
    min_paired_forecasts: int = 30,
    min_distinct_target_days: int = 14,
    min_morning_pairs: int = 5,
    min_late_pairs: int = 5,
) -> dict:
    """Evaluate whether the shadow challenger is ready for manual promotion review.

    This function is deliberately conservative. It never promotes a model by
    itself; it only reports whether the evidence is strong enough to start a
    manual promotion review.
    """
    if variants.empty:
        return _pending("forecast_variant_monitoring is empty", shadow_variant=shadow_variant, shadow_version=shadow_version)
    required = {
        "forecast_id",
        "forecast_variant",
        "target_date_local",
        "error_expected_c",
        "nll",
        "crps",
        "probability_above_actual_integer_bin",
        "coverage_80",
    }
    missing = sorted(required.difference(variants.columns))
    if missing:
        return _pending(f"missing columns: {', '.join(missing)}", shadow_variant=shadow_variant, shadow_version=shadow_version)

    df = variants.copy()
    champion = df[df["forecast_variant"] == CHAMPION_VARIANT].copy()
    shadow = df[df["forecast_variant"] == shadow_variant].copy()
    if "variant_version" in shadow.columns:
        versioned = shadow[shadow["variant_version"] == shadow_version].copy()
        if not versioned.empty:
            shadow = versioned
    if champion.empty or shadow.empty:
        return _pending("paired champion/shadow rows are not available", shadow_variant=shadow_variant, shadow_version=shadow_version)

    pairs = champion.merge(
        shadow,
        on="forecast_id",
        suffixes=("_champion", "_shadow"),
    )
    if pairs.empty:
        return _pending("no paired champion/shadow forecasts", shadow_variant=shadow_variant, shadow_version=shadow_version)

    pairs["abs_error_champion"] = pairs["error_expected_c_champion"].abs()
    pairs["abs_error_shadow"] = pairs["error_expected_c_shadow"].abs()
    pairs["crps_delta_shadow_minus_champion"] = pairs["crps_shadow"] - pairs["crps_champion"]
    pairs["nll_delta_shadow_minus_champion"] = pairs["nll_shadow"] - pairs["nll_champion"]
    pairs["mae_delta_shadow_minus_champion"] = pairs["abs_error_shadow"] - pairs["abs_error_champion"]
    pairs["false_upside_delta_shadow_minus_champion"] = (
        pairs["probability_above_actual_integer_bin_shadow"] - pairs["probability_above_actual_integer_bin_champion"]
    )

    morning = pairs[_local_hour(pairs) < 11.0]
    late = pairs[_local_hour(pairs) >= 16.0]
    distinct_days = int(pairs["target_date_local_champion"].astype(str).nunique())
    metrics = {
        "paired_forecasts": int(len(pairs)),
        "distinct_target_days": distinct_days,
        "morning_pairs": int(len(morning)),
        "late_pairs": int(len(late)),
        "mean_shadow_minus_champion_crps": _mean(pairs, "crps_delta_shadow_minus_champion"),
        "mean_shadow_minus_champion_nll": _mean(pairs, "nll_delta_shadow_minus_champion"),
        "mean_shadow_minus_champion_mae_c": _mean(pairs, "mae_delta_shadow_minus_champion"),
        "shadow_crps_win_rate": float((pairs["crps_shadow"] < pairs["crps_champion"]).mean()),
        "shadow_nll_win_rate": float((pairs["nll_shadow"] < pairs["nll_champion"]).mean()),
        "shadow_mae_win_rate": float((pairs["abs_error_shadow"] < pairs["abs_error_champion"]).mean()),
        "mean_shadow_minus_champion_false_upside_probability": _mean(pairs, "false_upside_delta_shadow_minus_champion"),
        "late_mean_shadow_minus_champion_false_upside_probability": _mean(late, "false_upside_delta_shadow_minus_champion"),
        "morning_mean_shadow_minus_champion_crps": _mean(morning, "crps_delta_shadow_minus_champion"),
        "champion_coverage_80": float(pairs["coverage_80_champion"].mean()),
        "shadow_coverage_80": float(pairs["coverage_80_shadow"].mean()),
    }
    checks = {
        "enough_paired_forecasts": int(len(pairs)) >= min_paired_forecasts,
        "enough_distinct_target_days": distinct_days >= min_distinct_target_days,
        "enough_morning_pairs": int(len(morning)) >= min_morning_pairs,
        "enough_late_pairs": int(len(late)) >= min_late_pairs,
        "crps_not_worse": metrics["mean_shadow_minus_champion_crps"] <= 0.0,
        "nll_not_worse": metrics["mean_shadow_minus_champion_nll"] <= 0.0,
        "mae_not_materially_worse": metrics["mean_shadow_minus_champion_mae_c"] <= 0.10,
        "late_false_upside_not_worse": (metrics["late_mean_shadow_minus_champion_false_upside_probability"] or 0.0) <= 0.0,
        "morning_not_materially_worse": (metrics["morning_mean_shadow_minus_champion_crps"] or 0.0) <= 0.05,
        "coverage_80_not_materially_worse": abs(metrics["shadow_coverage_80"] - 0.80)
        <= abs(metrics["champion_coverage_80"] - 0.80) + 0.05,
    }
    sample_ready = all(
        checks[key]
        for key in (
            "enough_paired_forecasts",
            "enough_distinct_target_days",
            "enough_morning_pairs",
            "enough_late_pairs",
        )
    )
    quality_ready = all(
        checks[key]
        for key in (
            "crps_not_worse",
            "nll_not_worse",
            "mae_not_materially_worse",
            "late_false_upside_not_worse",
            "morning_not_materially_worse",
            "coverage_80_not_materially_worse",
        )
    )
    if sample_ready and quality_ready:
        status = "eligible_for_manual_promotion_review"
        recommendation = "manual_review_required_before_promotion"
    elif sample_ready:
        status = "do_not_promote_quality_gate_failed"
        recommendation = "keep_shadow_and_inspect_failed_quality_checks"
    else:
        status = "continue_shadow_monitoring"
        recommendation = "collect_more_independent_outcomes"
    return {
        "status": status,
        "shadow_variant": shadow_variant,
        "shadow_version": shadow_version,
        "metrics": metrics,
        "checks": checks,
        "recommendation": recommendation,
        "notes": [
            "This gate is advisory and does not switch production automatically.",
            "Promotion requires manual review of paired forecasts, late-day behavior, morning behavior, and calibration.",
        ],
    }


def write_shadow_promotion_gate_report(
    variant_monitoring_path: str | Path = "data/reports/forecast_variant_monitoring.parquet",
    json_path: str | Path = "data/reports/shadow_promotion_gate.json",
    markdown_path: str | Path = "docs/shadow_promotion_gate.md",
) -> dict:
    path = Path(variant_monitoring_path)
    variants = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    gate = evaluate_shadow_promotion_gate(variants)
    json_output = Path(json_path)
    markdown_output = Path(markdown_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    markdown_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
    markdown_output.write_text(format_shadow_promotion_gate_markdown(gate), encoding="utf-8")
    return gate


def format_shadow_promotion_gate_markdown(gate: dict) -> str:
    lines = [
        "# Shadow promotion gate",
        "",
        f"Status: `{gate.get('status')}`",
        f"Shadow version: `{gate.get('shadow_version')}`",
        f"Recommendation: `{gate.get('recommendation')}`",
        "",
        "## Metrics",
        "",
    ]
    metrics = gate.get("metrics") or {}
    lines.extend(f"- `{key}`: `{value}`" for key, value in metrics.items())
    lines.extend(["", "## Checks", ""])
    checks = gate.get("checks") or {}
    lines.extend(f"- `{key}`: `{value}`" for key, value in checks.items())
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in gate.get("notes", []))
    lines.append("")
    return "\n".join(lines)


def _pending(reason: str, *, shadow_variant: str = SHADOW_VARIANT, shadow_version: str = PHASE_AWARE_SHADOW_VERSION) -> dict:
    return {
        "status": "pending",
        "reason": reason,
        "shadow_variant": shadow_variant,
        "shadow_version": shadow_version,
        "metrics": {},
        "checks": {},
        "recommendation": "collect_more_independent_outcomes",
        "notes": ["No production change is allowed while the promotion gate is pending."],
    }


def _local_hour(df: pd.DataFrame) -> pd.Series:
    if "local_issue_hour_shadow" in df.columns:
        return pd.to_numeric(df["local_issue_hour_shadow"], errors="coerce").fillna(0.0)
    if "issue_time_utc_shadow" in df.columns:
        timestamps = pd.to_datetime(df["issue_time_utc_shadow"], utc=True, errors="coerce")
    else:
        timestamps = pd.to_datetime(df["issue_time_utc_champion"], utc=True, errors="coerce")
    local = timestamps.dt.tz_convert("Europe/Berlin")
    return local.dt.hour + local.dt.minute / 60


def _mean(df: pd.DataFrame, column: str):
    if df.empty or column not in df.columns:
        return None
    return float(pd.to_numeric(df[column], errors="coerce").mean())
