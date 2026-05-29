from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from weather_tmax_bot.features.climatology_features import climatology_sample
from weather_tmax_bot.models.distribution import TmaxDistribution, empirical_distribution


class ClimatologyBaseline:
    def __init__(self, window_days: int = 30, bin_min: int = -35, bin_max: int = 45):
        self.window_days = window_days
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.targets = pd.DataFrame()

    def fit(self, daily_target: pd.DataFrame) -> "ClimatologyBaseline":
        self.targets = daily_target.copy()
        return self

    def predict_distribution(self, target_date: date, observed_max_so_far: float | None = None) -> TmaxDistribution:
        samples = climatology_sample(self.targets, target_date, self.window_days)
        if samples.empty and not self.targets.empty:
            samples = pd.to_numeric(self.targets["tmax_c"], errors="coerce").dropna()
        dist = empirical_distribution(samples.to_numpy(), self.bin_min, self.bin_max)
        return dist.truncate_below(observed_max_so_far)


class ResidualNWPBaseline:
    def fit(self, y_true: np.ndarray, model_tmax: np.ndarray) -> "ResidualNWPBaseline":
        mask = np.isfinite(y_true) & np.isfinite(model_tmax)
        self.residuals_ = y_true[mask] - model_tmax[mask]
        return self

    def predict_distribution(self, model_tmax_c: float) -> TmaxDistribution:
        return empirical_distribution(model_tmax_c + self.residuals_)
