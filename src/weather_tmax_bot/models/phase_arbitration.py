from __future__ import annotations

from dataclasses import dataclass

from weather_tmax_bot.models.distribution import TmaxDistribution


@dataclass(frozen=True)
class PhaseArbitrationResult:
    distribution: TmaxDistribution
    details: dict


def build_phase_arbitrated_candidate(
    *,
    champion: TmaxDistribution,
    safe_blend: TmaxDistribution | None,
    seasonal_shadow: TmaxDistribution | None,
    ml_shadow: TmaxDistribution | None,
    local_hour: float,
) -> PhaseArbitrationResult:
    """Select a conservative shadow distribution using observed phase performance.

    This is intentionally shadow-only. It encodes the current operational
    ablation finding: champion is safest before midday, safe blend is strongest
    through the main heating window, and late-day intraday models are best once
    the daily peak is usually constrained by observations.
    """
    if local_hour < 12.0:
        selected_name = "production_champion"
        selected = champion
        reason = "before_12_keep_stable_champion"
    elif local_hour < 16.0:
        selected_name, selected, reason = _first_available(
            ("shadow_safe_blend", safe_blend, "midday_safe_blend_best_ablation"),
            ("shadow_seasonal_intraday", seasonal_shadow, "midday_safe_blend_missing_use_seasonal"),
            ("production_champion", champion, "midday_shadow_missing_use_champion"),
        )
    elif local_hour < 20.0:
        selected_name, selected, reason = _first_available(
            ("shadow_intraday_ml", ml_shadow, "late_day_ml_best_ablation"),
            ("shadow_seasonal_intraday", seasonal_shadow, "late_day_ml_missing_use_seasonal"),
            ("shadow_safe_blend", safe_blend, "late_day_seasonal_missing_use_safe_blend"),
            ("production_champion", champion, "late_day_shadow_missing_use_champion"),
        )
    else:
        selected_name, selected, reason = _first_available(
            ("shadow_seasonal_intraday", seasonal_shadow, "evening_seasonal_best_ablation"),
            ("shadow_safe_blend", safe_blend, "evening_seasonal_missing_use_safe_blend"),
            ("shadow_intraday_ml", ml_shadow, "evening_safe_blend_missing_use_ml"),
            ("production_champion", champion, "evening_shadow_missing_use_champion"),
        )
    return PhaseArbitrationResult(
        selected,
        {
            "active": selected_name != "production_champion",
            "variant_version": "phase_arbitrated_shadow_v1",
            "status": "shadow_only_does_not_affect_operational_forecast",
            "selected_variant": selected_name,
            "selection_reason": reason,
            "local_issue_hour": float(local_hour),
            "expected_tmax_c": selected.expected_tmax_c,
            "champion_expected_tmax_c": champion.expected_tmax_c,
        },
    )


def _first_available(*items: tuple[str, TmaxDistribution | None, str]) -> tuple[str, TmaxDistribution, str]:
    for name, distribution, reason in items:
        if distribution is not None:
            return name, distribution, reason
    raise ValueError("at least one candidate distribution must be provided")
