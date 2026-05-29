from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from weather_tmax_bot.evaluation.metrics import nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution


class CDFIsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, predicted_cdf_values: np.ndarray, observed_indicators: np.ndarray) -> "CDFIsotonicCalibrator":
        self.model.fit(np.asarray(predicted_cdf_values, dtype=float), np.asarray(observed_indicators, dtype=float))
        return self

    def transform(self, cdf_values: np.ndarray) -> np.ndarray:
        calibrated = self.model.predict(np.asarray(cdf_values, dtype=float))
        return np.maximum.accumulate(np.clip(calibrated, 0.0, 1.0))


class IntegerCDFIsotonicCalibrator:
    """Calibrate a discrete Tmax CDF using validation forecast CDF points."""

    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, distributions: list[TmaxDistribution], actuals: np.ndarray) -> "IntegerCDFIsotonicCalibrator":
        xs = []
        ys = []
        for dist, actual in zip(distributions, actuals):
            cdf = np.cumsum(dist.probabilities)
            xs.extend(cdf.tolist())
            ys.extend((dist.bins_c >= actual).astype(float).tolist())
        self.model.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
        self.fitted = True
        return self

    def transform(self, distribution: TmaxDistribution) -> TmaxDistribution:
        if not self.fitted:
            return distribution
        cdf = np.cumsum(distribution.probabilities)
        calibrated_cdf = self.model.predict(cdf)
        calibrated_cdf = np.maximum.accumulate(np.clip(calibrated_cdf, 0.0, 1.0))
        calibrated_cdf[-1] = 1.0
        probs = np.diff(np.concatenate(([0.0], calibrated_cdf)))
        return TmaxDistribution(distribution.bins_c, probs)


def pit_values(distributions, actuals: np.ndarray) -> np.ndarray:
    pits = []
    for dist, actual in zip(distributions, actuals):
        lower = dist.probabilities[dist.bins_c < round(actual)].sum()
        at = dist.probabilities[dist.bins_c == round(actual)].sum()
        pits.append(lower + 0.5 * at)
    return np.asarray(pits)


class DiscreteSpreadCalibrator:
    """Validation-fitted probability spreader for underdispersed integer-bin forecasts."""

    def __init__(self, sigma_bins: float = 0.0):
        self.sigma_bins = sigma_bins

    def fit(self, distributions: list[TmaxDistribution], actuals: np.ndarray) -> "DiscreteSpreadCalibrator":
        best_sigma = 0.0
        best_score = float("inf")
        for sigma in np.linspace(0.0, 5.0, 21):
            calibrated = [self._apply_sigma(dist, sigma) for dist in distributions]
            coverage_50 = np.mean([_covered(dist, actual, 0.50) for dist, actual in zip(calibrated, actuals)])
            coverage_80 = np.mean([_covered(dist, actual, 0.80) for dist, actual in zip(calibrated, actuals)])
            coverage_90 = np.mean([_covered(dist, actual, 0.90) for dist, actual in zip(calibrated, actuals)])
            nll = np.mean([nll_integer_bin(dist, actual) for dist, actual in zip(calibrated, actuals)])
            score = abs(coverage_50 - 0.50) + abs(coverage_80 - 0.80) + abs(coverage_90 - 0.90) + 0.05 * nll
            if score < best_score:
                best_score = score
                best_sigma = float(sigma)
        self.sigma_bins = best_sigma
        return self

    def transform(self, distribution: TmaxDistribution) -> TmaxDistribution:
        return self._apply_sigma(distribution, self.sigma_bins)

    @staticmethod
    def _apply_sigma(distribution: TmaxDistribution, sigma_bins: float) -> TmaxDistribution:
        if sigma_bins <= 0:
            return distribution
        radius = max(1, int(np.ceil(4 * sigma_bins)))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel = kernel / kernel.sum()
        probs = np.convolve(distribution.probabilities, kernel, mode="same")
        return TmaxDistribution(distribution.bins_c, probs)


def _covered(dist: TmaxDistribution, actual: float, central_mass: float) -> bool:
    lower, upper = dist.interval(central_mass)
    return lower <= actual <= upper
