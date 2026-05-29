from __future__ import annotations

import numpy as np

from weather_tmax_bot.models.distribution import TmaxDistribution


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def nll_integer_bin(dist: TmaxDistribution, actual: float, eps: float = 1e-12) -> float:
    rounded = int(round(actual))
    prob = dist.probabilities[dist.bins_c == rounded].sum()
    return float(-np.log(max(prob, eps)))


def brier(probability: float, event: bool) -> float:
    return float((probability - float(event)) ** 2)


def crps_discrete(dist: TmaxDistribution, actual: float) -> float:
    cdf = np.cumsum(dist.probabilities)
    obs_cdf = (dist.bins_c >= actual).astype(float)
    return float(np.mean((cdf - (1 - obs_cdf)) ** 2))
