import numpy as np

from weather_tmax_bot.models.distribution import quantiles_to_distribution


def test_distribution_sums_to_one_and_monotone():
    dist = quantiles_to_distribution([0.1, 0.5, 0.9], [10, 15, 20], bin_min=0, bin_max=30)
    assert abs(dist.probabilities.sum() - 1.0) < 1e-6
    assert np.all(np.diff(np.cumsum(dist.probabilities)) >= -1e-12)


def test_truncate_below_observed_max():
    dist = quantiles_to_distribution([0.1, 0.5, 0.9], [10, 15, 20], bin_min=0, bin_max=30).truncate_below(18.2)
    assert dist.probabilities[dist.bins_c < 19].sum() == 0
