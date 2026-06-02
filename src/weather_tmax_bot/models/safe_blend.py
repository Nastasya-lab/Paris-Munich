from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from weather_tmax_bot.models.distribution import TmaxDistribution


@dataclass
class SafeBlendResult:
    distribution: TmaxDistribution
    details: dict


def build_blended_shadow_candidate(
    champion: TmaxDistribution,
    phase_shadow: TmaxDistribution,
    *,
    phase_details: dict,
    ml_shadow_details: dict | None = None,
    model_disagreement: dict | None = None,
    source_compatibility: dict | None = None,
    freshness: dict | None = None,
) -> SafeBlendResult:
    """Build a conservative smooth shadow candidate without mixing raw ML bins."""
    phase_details = phase_details or {}
    ml_shadow_details = ml_shadow_details or {}
    model_disagreement = model_disagreement or {}
    source_compatibility = source_compatibility or {}
    freshness = freshness or {}
    reasons: list[str] = []

    if not phase_details.get("active"):
        return SafeBlendResult(
            champion,
            _details(
                blend_weight=0.0,
                champion=champion,
                phase_shadow=phase_shadow,
                blended=champion,
                reasons=["phase_shadow_inactive"],
                phase_details=phase_details,
                ml_shadow_details=ml_shadow_details,
            ),
        )

    phase = str(phase_details.get("forecast_phase") or "unknown")
    scenario = str(phase_details.get("scenario_tracking") or "unknown")
    weight = {
        "morning_prior": 0.15,
        "midday_update": 0.35,
        "late_nowcast": 0.55,
    }.get(phase, 0.25)
    reasons.append(f"phase_base_weight:{phase}")

    ml_active = bool(ml_shadow_details.get("active"))
    ml_peak_passed = _optional_float(ml_shadow_details.get("probability_peak_already_passed"))
    ml_upside_ge_2 = _optional_float(ml_shadow_details.get("probability_upside_ge_2c"))
    phase_cutoff = scenario == "heating_cutoff_likely" or bool(phase_details.get("late_drop_override_active"))
    sharp_drop = float(phase_details.get("drop_from_observed_max_c") or 0.0) >= 3.0
    ml_confirms_cutoff = ml_active and (
        (ml_peak_passed is not None and ml_peak_passed >= 0.75)
        or (ml_upside_ge_2 is not None and ml_upside_ge_2 <= 0.20)
    )
    ml_contradicts_cutoff = ml_active and ml_upside_ge_2 is not None and ml_upside_ge_2 >= 0.60
    late_consensus_cutoff = phase == "late_nowcast" and phase_cutoff and sharp_drop and ml_confirms_cutoff

    if late_consensus_cutoff:
        weight = max(weight, 0.75)
        reasons.append("late_sharp_drop_confirmed_by_ml_survival_signal")
    elif phase_cutoff and ml_contradicts_cutoff:
        weight = min(weight, 0.20)
        reasons.append("ml_survival_signal_contradicts_phase_cutoff")

    severity = str(model_disagreement.get("severity") or "none")
    if severity == "high":
        cap = 0.55 if late_consensus_cutoff else 0.25
        if weight > cap:
            weight = cap
            reasons.append("high_model_disagreement_capped_blend")
    elif severity == "watch" and weight > 0.50:
        weight = 0.50
        reasons.append("model_disagreement_watch_capped_blend")

    compatibility_statuses = {
        str((source_compatibility.get(kind) or {}).get("status") or "missing")
        for kind in ("metar", "taf", "nwp")
    }
    if compatibility_statuses & {"unknown_mismatch", "forbidden_mismatch"}:
        weight = 0.0
        reasons.append("untrusted_runtime_source_disabled_blend")
    elif "known_compatible" in compatibility_statuses:
        weight *= 0.90
        reasons.append("known_compatible_runtime_source_discount")

    metar_freshness = str((freshness.get("metar") or {}).get("state") or "missing")
    nwp_freshness = str((freshness.get("nwp") or {}).get("state") or "missing")
    if metar_freshness != "fresh":
        weight *= 0.50
        reasons.append("metar_not_fresh_discount")
    if nwp_freshness != "fresh":
        weight *= 0.80
        reasons.append("nwp_not_fresh_discount")

    weight = float(np.clip(weight, 0.0, 0.75))
    blended = _blend_distributions(champion, phase_shadow, weight)
    return SafeBlendResult(
        blended,
        _details(
            blend_weight=weight,
            champion=champion,
            phase_shadow=phase_shadow,
            blended=blended,
            reasons=reasons,
            phase_details=phase_details,
            ml_shadow_details=ml_shadow_details,
            late_consensus_cutoff=late_consensus_cutoff,
        ),
    )


def _blend_distributions(
    champion: TmaxDistribution,
    phase_shadow: TmaxDistribution,
    weight: float,
) -> TmaxDistribution:
    bins = np.union1d(champion.bins_c, phase_shadow.bins_c)
    champion_probs = _probabilities_on_bins(champion, bins)
    phase_probs = _probabilities_on_bins(phase_shadow, bins)
    return TmaxDistribution(bins, (1.0 - weight) * champion_probs + weight * phase_probs)


def _probabilities_on_bins(distribution: TmaxDistribution, bins: np.ndarray) -> np.ndarray:
    mapping = dict(zip(distribution.bins_c.tolist(), distribution.probabilities.tolist()))
    return np.asarray([mapping.get(int(bin_c), 0.0) for bin_c in bins], dtype=float)


def _details(
    *,
    blend_weight: float,
    champion: TmaxDistribution,
    phase_shadow: TmaxDistribution,
    blended: TmaxDistribution,
    reasons: list[str],
    phase_details: dict,
    ml_shadow_details: dict,
    late_consensus_cutoff: bool = False,
) -> dict:
    return {
        "active": blend_weight > 0.0,
        "variant_version": "blended_shadow_candidate_v1",
        "status": "shadow_only_does_not_affect_operational_forecast",
        "blend_weight": blend_weight,
        "forecast_phase": phase_details.get("forecast_phase"),
        "scenario_tracking": phase_details.get("scenario_tracking"),
        "late_consensus_cutoff": late_consensus_cutoff,
        "champion_expected_tmax_c": champion.expected_tmax_c,
        "phase_shadow_expected_tmax_c": phase_shadow.expected_tmax_c,
        "blended_expected_tmax_c": blended.expected_tmax_c,
        "ml_signal_used": bool(ml_shadow_details.get("active")),
        "ml_distribution_directly_used": False,
        "ml_probability_peak_already_passed": ml_shadow_details.get("probability_peak_already_passed"),
        "ml_probability_upside_ge_1c": ml_shadow_details.get("probability_upside_ge_1c"),
        "ml_probability_upside_ge_2c": ml_shadow_details.get("probability_upside_ge_2c"),
        "ml_probability_upside_ge_3c": ml_shadow_details.get("probability_upside_ge_3c"),
        "reasons": reasons,
    }


def _optional_float(value) -> float | None:
    return None if value is None else float(value)
