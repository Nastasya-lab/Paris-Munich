from weather_tmax_bot.evaluation.metrics import brier, nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution


def test_metrics_finite():
    dist = TmaxDistribution([19, 20, 21], [0.2, 0.5, 0.3])
    assert nll_integer_bin(dist, 20) > 0
    assert brier(0.7, True) >= 0
