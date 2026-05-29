from __future__ import annotations

import numpy as np

from weather_tmax_bot.models.distribution import TmaxDistribution


def average_distributions(distributions: list[TmaxDistribution], weights: list[float] | None = None) -> TmaxDistribution:
    if not distributions:
        raise ValueError("at least one distribution required")
    weights_arr = np.ones(len(distributions)) if weights is None else np.asarray(weights, dtype=float)
    weights_arr = weights_arr / weights_arr.sum()
    bins = distributions[0].bins_c
    probs = sum(w * d.probabilities for w, d in zip(weights_arr, distributions))
    return TmaxDistribution(bins, probs)
