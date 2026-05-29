from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.operations.acceptance import evaluate_forecast_acceptance


def test_acceptance_allows_ok_with_cautions():
    result = evaluate_forecast_acceptance(
        distribution=TmaxDistribution([10, 11], [0.4, 0.6]),
        forecast_quality={"status": "ok", "reasons": [], "cautions": ["known compatible source"]},
    )

    assert result["accepted"]
    assert result["cautions"] == ["known compatible source"]


def test_acceptance_blocks_degraded_forecast():
    result = evaluate_forecast_acceptance(
        distribution=TmaxDistribution([10, 11], [0.4, 0.6]),
        forecast_quality={"status": "degraded", "reasons": ["metar is stale"], "cautions": []},
    )

    assert not result["accepted"]
    assert "quality_status_ok" in result["blocking_reasons"]


def test_acceptance_blocks_payload_without_probabilities():
    result = evaluate_forecast_acceptance({"forecast_quality": {"status": "ok", "reasons": []}})

    assert not result["accepted"]
    assert "has_probability_bins" in result["blocking_reasons"]
