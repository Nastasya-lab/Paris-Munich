import numpy as np

from weather_tmax_bot.models.distribution import (
    TmaxDistribution,
    project_unimodal_distribution,
    quantiles_to_distribution,
    temperature_scale_distribution,
    unimodal_violation_count,
)


def test_distribution_sums_to_one_and_monotone():
    dist = quantiles_to_distribution([0.1, 0.5, 0.9], [10, 15, 20], bin_min=0, bin_max=30)
    assert abs(dist.probabilities.sum() - 1.0) < 1e-6
    assert np.all(np.diff(np.cumsum(dist.probabilities)) >= -1e-12)


def test_truncate_below_observed_max():
    dist = quantiles_to_distribution([0.1, 0.5, 0.9], [10, 15, 20], bin_min=0, bin_max=30).truncate_below(18.2)
    assert dist.probabilities[dist.bins_c < 19].sum() == 0


def test_project_unimodal_distribution_removes_internal_valley():
    dist = TmaxDistribution(np.array([35, 36, 37, 38, 39]), np.array([0.04, 0.47, 0.12, 0.27, 0.10]))

    projected = project_unimodal_distribution(dist)

    assert np.isclose(projected.probabilities.sum(), 1.0)
    assert unimodal_violation_count(projected) == 0
    mode = int(np.argmax(projected.probabilities))
    assert np.all(np.diff(projected.probabilities[: mode + 1]) >= -1e-12)
    assert np.all(np.diff(projected.probabilities[mode:]) <= 1e-12)


def test_project_unimodal_distribution_keeps_valid_unimodal_shape_close():
    dist = TmaxDistribution(np.array([35, 36, 37, 38, 39]), np.array([0.05, 0.18, 0.42, 0.25, 0.10]))

    projected = project_unimodal_distribution(dist)

    assert unimodal_violation_count(projected) == 0
    assert np.allclose(projected.probabilities, dist.probabilities)


def test_temperature_scale_distribution_sharpens_when_temperature_below_one():
    dist = TmaxDistribution(np.array([20, 21, 22]), np.array([0.2, 0.5, 0.3]))

    scaled = temperature_scale_distribution(dist, 0.7)

    assert np.isclose(scaled.probabilities.sum(), 1.0)
    assert scaled.probabilities[1] > dist.probabilities[1]
