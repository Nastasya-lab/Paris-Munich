from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution


def test_metrics_finite():
    dist = TmaxDistribution([19, 20, 21], [0.2, 0.5, 0.3])
    assert nll_integer_bin(dist, 20) > 0
    assert brier(0.7, True) >= 0


def test_crps_is_zero_for_exact_integer_bin_forecast():
    exact = TmaxDistribution([19, 20, 21], [0.0, 1.0, 0.0])
    wrong = TmaxDistribution([19, 20, 21], [1.0, 0.0, 0.0])

    assert crps_discrete(exact, 20) == 0.0
    assert crps_discrete(wrong, 20) > 0.0
