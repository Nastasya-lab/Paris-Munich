from __future__ import annotations

from datetime import date, datetime

from weather_tmax_bot.models.distribution import TmaxDistribution


def format_prediction(
    airport: str,
    target_date: date,
    issue_time_utc: datetime,
    dist: TmaxDistribution,
    model_version: str = "climatology_mvp",
    data_lineage: dict | None = None,
    forecast_quality: dict | None = None,
    forecast_acceptance: dict | None = None,
    forecast_components: dict | None = None,
    warnings: list[str] | None = None,
) -> str:
    payload = dist.to_payload()
    lines = [
        f"Airport: {airport}",
        f"Target date: {target_date.isoformat()} Europe/Berlin",
        f"Issue time: {issue_time_utc:%Y-%m-%d %H:%M UTC}",
        f"Model version: {model_version}",
        f"Forecast status: {(forecast_quality or {}).get('status', 'unknown')}",
        f"Acceptance: {'accepted' if (forecast_acceptance or {}).get('accepted') else 'rejected'}",
        "Data mode: as-of knowledge view",
        "Truth source for training: DWD",
        "Feature sources: climatology baseline; METAR/TAF/NWP if available",
        "",
        f"Expected Tmax: {payload['expected_tmax_c']:.1f}C",
        f"Median Tmax: {payload['median_tmax_c']:.1f}C",
        f"Most likely bin: {payload['most_likely_integer_c']}C",
        f"80% interval: {payload['intervals']['80'][0]:.1f}C - {payload['intervals']['80'][1]:.1f}C",
        "",
        "Probabilities:",
    ]
    for k, v in payload["probabilities_by_integer_c"].items():
        if v >= 0.001:
            lines.append(f"{k}C: {100 * v:.1f}%")
    lines += [
        "",
        "Thresholds:",
        f"P(Tmax >= 20C): {100 * payload['threshold_probabilities']['ge_20']:.1f}%",
        f"P(Tmax >= 25C): {100 * payload['threshold_probabilities']['ge_25']:.1f}%",
        f"P(Tmax >= 30C): {100 * payload['threshold_probabilities']['ge_30']:.1f}%",
        f"P(Tmax <= 0C): {100 * payload['threshold_probabilities']['le_0']:.1f}%",
    ]
    intraday = (forecast_components or {}).get("intraday_update") or {}
    base = (forecast_components or {}).get("base_model") or {}
    if intraday:
        lines += [
            "",
            "Model signals:",
            f"- base expected Tmax: {_fmt_component_expected(base)}",
            f"- intraday active: {intraday.get('active')}",
        ]
        if intraday.get("active"):
            lines.extend(
                [
                    f"- peak already passed probability: {100 * float(intraday.get('peak_passed_probability', 0.0)):.1f}%",
                    f"- observed max so far: {float(intraday.get('observed_max_so_far_c')):.1f}C",
                    f"- drop from observed max: {float(intraday.get('drop_from_observed_max_c')):.1f}C",
                    f"- intraday blend weight: {100 * float(intraday.get('intraday_blend_weight', 0.0)):.1f}%",
                ]
            )
        elif intraday.get("reason"):
            lines.append(f"- intraday reason: {intraday.get('reason')}")
    shadow = (forecast_components or {}).get("shadow_mode") or {}
    shadow_intraday = shadow.get("intraday_update") or {}
    shadow_final = shadow.get("final_model") or {}
    shadow_comparison = shadow.get("comparison_to_champion") or {}
    if shadow:
        lines += [
            "",
            "Shadow scenario: seasonal intraday challenger",
            "- shadow only: does not affect the operational forecast",
            f"- intraday active: {shadow_intraday.get('active')}",
        ]
        if shadow_intraday.get("active"):
            lines.extend(
                [
                    f"- seasonal profile: {shadow_intraday.get('seasonal_profile')}",
                    f"- intraday blend weight: {100 * float(shadow_intraday.get('intraday_blend_weight', 0.0)):.1f}%",
                    f"- expected Tmax: {_fmt_component_expected(shadow_final)} ({_fmt_signed_c(shadow_comparison.get('expected_tmax_delta_c'))} vs champion)",
                    f"- most likely bin: {shadow_final.get('most_likely_integer_c')}C",
                    f"- P(Tmax >= 30C): {100 * float((shadow_final.get('threshold_probabilities') or {}).get('ge_30', 0.0)):.1f}%",
                    f"- late-drop override active: {shadow_intraday.get('late_drop_override_active', False)}",
                ]
            )
        elif shadow_intraday.get("reason"):
            lines.append(f"- intraday reason: {shadow_intraday.get('reason')}")
    if data_lineage:
        lines.append("")
        lines.append("Data used:")
        lines.append(f"- latest METAR: {data_lineage.get('metar_latest_time_utc')}")
        lines.append(f"- latest TAF: {data_lineage.get('taf_issue_time_utc')}")
        lines.append(f"- NWP models used: {data_lineage.get('nwp_runs')}")
        lines.append(f"- max feature knowledge time: {data_lineage.get('max_feature_knowledge_time_utc')}")
    if forecast_quality:
        lines.append("")
        lines.append("Quality:")
        for reason in forecast_quality.get("reasons", []):
            lines.append(f"- {reason}")
        for caution in forecast_quality.get("cautions", []):
            lines.append(f"- caution: {caution}")
        recommendation = forecast_quality.get("recommendation")
        if recommendation:
            lines.append(f"Recommendation: {recommendation}")
    if forecast_acceptance and not forecast_acceptance.get("accepted"):
        lines.append("")
        lines.append("Acceptance gate:")
        for reason in forecast_acceptance.get("blocking_reasons", []):
            lines.append(f"- {reason}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines)


def _fmt_component_expected(component: dict) -> str:
    value = component.get("expected_tmax_c")
    return "not available" if value is None else f"{float(value):.1f}C"


def _fmt_signed_c(value) -> str:
    return "not available" if value is None else f"{float(value):+.1f}C"
