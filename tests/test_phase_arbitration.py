from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.phase_arbitration import build_phase_arbitrated_candidate


def test_phase_arbitration_keeps_champion_before_midday():
    champion = TmaxDistribution([20], [1.0])
    safe = TmaxDistribution([21], [1.0])

    result = build_phase_arbitrated_candidate(
        champion=champion,
        safe_blend=safe,
        seasonal_shadow=None,
        ml_shadow=None,
        local_hour=10.5,
    )

    assert result.details["selected_variant"] == "production_champion"
    assert result.distribution.expected_tmax_c == 20.0


def test_phase_arbitration_uses_safe_blend_in_main_heating_window():
    champion = TmaxDistribution([20], [1.0])
    safe = TmaxDistribution([21], [1.0])
    seasonal = TmaxDistribution([22], [1.0])

    result = build_phase_arbitrated_candidate(
        champion=champion,
        safe_blend=safe,
        seasonal_shadow=seasonal,
        ml_shadow=None,
        local_hour=14.0,
    )

    assert result.details["selected_variant"] == "shadow_safe_blend"
    assert result.distribution.expected_tmax_c == 21.0


def test_phase_arbitration_uses_ml_late_when_available_then_seasonal_evening():
    champion = TmaxDistribution([20], [1.0])
    safe = TmaxDistribution([21], [1.0])
    seasonal = TmaxDistribution([22], [1.0])
    ml = TmaxDistribution([23], [1.0])

    late = build_phase_arbitrated_candidate(
        champion=champion,
        safe_blend=safe,
        seasonal_shadow=seasonal,
        ml_shadow=ml,
        local_hour=17.0,
    )
    evening = build_phase_arbitrated_candidate(
        champion=champion,
        safe_blend=safe,
        seasonal_shadow=seasonal,
        ml_shadow=ml,
        local_hour=21.0,
    )

    assert late.details["selected_variant"] == "shadow_intraday_ml"
    assert late.distribution.expected_tmax_c == 23.0
    assert evening.details["selected_variant"] == "shadow_seasonal_intraday"
    assert evening.distribution.expected_tmax_c == 22.0
