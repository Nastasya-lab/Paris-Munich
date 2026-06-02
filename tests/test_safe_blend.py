from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.safe_blend import build_blended_shadow_candidate


def test_safe_blend_is_weak_in_morning_and_stays_between_smooth_inputs():
    champion = TmaxDistribution([20, 21, 22], [0.2, 0.5, 0.3])
    phase = TmaxDistribution([20, 21, 22], [0.5, 0.4, 0.1])

    result = build_blended_shadow_candidate(
        champion,
        phase,
        phase_details={"active": True, "forecast_phase": "morning_prior"},
        source_compatibility=_trusted_sources(),
        freshness=_fresh_sources(),
    )

    assert result.details["blend_weight"] == 0.15
    assert phase.expected_tmax_c < result.distribution.expected_tmax_c < champion.expected_tmax_c
    assert result.details["ml_distribution_directly_used"] is False


def test_safe_blend_trusts_late_sharp_drop_when_ml_survival_signal_agrees():
    champion = TmaxDistribution([29, 30, 31], [0.2, 0.5, 0.3])
    phase = TmaxDistribution([29, 30, 31], [0.9, 0.09, 0.01])

    result = build_blended_shadow_candidate(
        champion,
        phase,
        phase_details={
            "active": True,
            "forecast_phase": "late_nowcast",
            "scenario_tracking": "heating_cutoff_likely",
            "drop_from_observed_max_c": 8.0,
        },
        ml_shadow_details={
            "active": True,
            "probability_peak_already_passed": 0.92,
            "probability_upside_ge_2c": 0.06,
        },
        source_compatibility=_trusted_sources(),
        freshness=_fresh_sources(),
    )

    assert result.details["late_consensus_cutoff"] is True
    assert result.details["blend_weight"] == 0.75
    assert result.distribution.threshold_ge(31) < champion.threshold_ge(31)


def test_safe_blend_reduces_phase_weight_when_ml_survival_signal_contradicts_cutoff():
    champion = TmaxDistribution([24, 25, 26], [0.2, 0.5, 0.3])
    phase = TmaxDistribution([24, 25, 26], [0.8, 0.15, 0.05])

    result = build_blended_shadow_candidate(
        champion,
        phase,
        phase_details={
            "active": True,
            "forecast_phase": "late_nowcast",
            "scenario_tracking": "heating_cutoff_likely",
            "drop_from_observed_max_c": 1.0,
        },
        ml_shadow_details={"active": True, "probability_upside_ge_2c": 0.75},
        source_compatibility=_trusted_sources(),
        freshness=_fresh_sources(),
    )

    assert result.details["blend_weight"] == 0.20
    assert "ml_survival_signal_contradicts_phase_cutoff" in result.details["reasons"]


def test_safe_blend_disables_phase_weight_for_untrusted_runtime_source():
    champion = TmaxDistribution([20, 21], [0.4, 0.6])
    phase = TmaxDistribution([20, 21], [0.9, 0.1])
    sources = _trusted_sources()
    sources["metar"] = {"status": "unknown_mismatch"}

    result = build_blended_shadow_candidate(
        champion,
        phase,
        phase_details={"active": True, "forecast_phase": "midday_update"},
        source_compatibility=sources,
        freshness=_fresh_sources(),
    )

    assert result.details["blend_weight"] == 0.0
    assert result.distribution.expected_tmax_c == champion.expected_tmax_c


def _trusted_sources() -> dict:
    return {
        "metar": {"status": "exact_match"},
        "taf": {"status": "exact_match"},
        "nwp": {"status": "exact_match"},
    }


def _fresh_sources() -> dict:
    return {
        "metar": {"state": "fresh"},
        "taf": {"state": "fresh"},
        "nwp": {"state": "fresh"},
    }
