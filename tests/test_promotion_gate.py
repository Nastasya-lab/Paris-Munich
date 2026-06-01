import pandas as pd

from weather_tmax_bot.evaluation.promotion_gate import (
    PHASE_AWARE_SHADOW_VERSION,
    evaluate_shadow_promotion_gate,
    format_shadow_promotion_gate_markdown,
)


def test_shadow_promotion_gate_waits_for_enough_independent_outcomes():
    variants = pd.DataFrame(
        [
            _row("f1", "production_champion", 10.0, 2.0, 1.0, 0.4, 0.7, 8.0),
            _row("f1", "shadow_seasonal_intraday", 10.0, 1.5, 0.8, 0.2, 0.8, 8.0),
        ]
    )

    gate = evaluate_shadow_promotion_gate(variants)

    assert gate["status"] == "continue_shadow_monitoring"
    assert gate["checks"]["enough_paired_forecasts"] is False


def test_shadow_promotion_gate_allows_manual_review_when_quality_and_sample_pass():
    rows = []
    for idx in range(30):
        hour = 8.0 if idx < 6 else 17.0 if idx < 12 else 13.0
        forecast_id = f"f{idx}"
        rows.append(_row(forecast_id, "production_champion", 20.0, 2.0, 1.0, 0.30, 0.70, hour, day=idx))
        rows.append(_row(forecast_id, "shadow_seasonal_intraday", 20.0, 1.7, 0.8, 0.20, 0.82, hour, day=idx))
    gate = evaluate_shadow_promotion_gate(pd.DataFrame(rows))

    assert gate["status"] == "eligible_for_manual_promotion_review"
    assert gate["checks"]["late_false_upside_not_worse"] is True
    assert gate["metrics"]["paired_forecasts"] == 30


def test_shadow_promotion_gate_blocks_quality_failure_after_sample_passes():
    rows = []
    for idx in range(30):
        hour = 8.0 if idx < 6 else 17.0 if idx < 12 else 13.0
        forecast_id = f"f{idx}"
        rows.append(_row(forecast_id, "production_champion", 20.0, 2.0, 1.0, 0.20, 0.75, hour, day=idx))
        rows.append(_row(forecast_id, "shadow_seasonal_intraday", 20.0, 2.6, 1.4, 0.35, 0.65, hour, day=idx))
    gate = evaluate_shadow_promotion_gate(pd.DataFrame(rows))

    assert gate["status"] == "do_not_promote_quality_gate_failed"
    assert gate["checks"]["crps_not_worse"] is False
    assert gate["checks"]["late_false_upside_not_worse"] is False


def test_shadow_promotion_gate_markdown():
    text = format_shadow_promotion_gate_markdown(
        {
            "status": "continue_shadow_monitoring",
            "shadow_version": PHASE_AWARE_SHADOW_VERSION,
            "recommendation": "collect_more_independent_outcomes",
            "metrics": {"paired_forecasts": 1},
            "checks": {"enough_paired_forecasts": False},
            "notes": ["note"],
        }
    )

    assert "Shadow promotion gate" in text
    assert PHASE_AWARE_SHADOW_VERSION in text


def _row(
    forecast_id: str,
    variant: str,
    actual: float,
    nll: float,
    crps: float,
    false_upside: float,
    coverage_80: float,
    local_hour: float,
    *,
    day: int = 1,
) -> dict:
    return {
        "forecast_id": forecast_id,
        "forecast_variant": variant,
        "variant_version": PHASE_AWARE_SHADOW_VERSION if variant == "shadow_seasonal_intraday" else "production_dynamic_v1",
        "target_date_local": f"2026-07-{day % 28 + 1:02d}",
        "error_expected_c": -0.5 if variant == "shadow_seasonal_intraday" else 0.8,
        "nll": nll,
        "crps": crps,
        "probability_actual_integer_bin": 0.5,
        "probability_above_actual_integer_bin": false_upside,
        "coverage_80": coverage_80,
        "local_issue_hour": local_hour,
        "actual_tmax_c": actual,
    }
