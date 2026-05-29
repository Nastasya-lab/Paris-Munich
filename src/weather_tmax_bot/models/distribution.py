from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TmaxDistribution:
    bins_c: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        self.bins_c = np.asarray(self.bins_c, dtype=int)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
        total = self.probabilities.sum()
        if total <= 0:
            raise ValueError("probabilities must have positive mass")
        self.probabilities = self.probabilities / total

    @property
    def expected_tmax_c(self) -> float:
        return float(np.sum(self.bins_c * self.probabilities))

    @property
    def most_likely_integer_c(self) -> int:
        return int(self.bins_c[np.argmax(self.probabilities)])

    @property
    def median_tmax_c(self) -> float:
        return float(self.quantile(0.5))

    def quantile(self, q: float) -> float:
        cdf = np.cumsum(self.probabilities)
        return float(self.bins_c[np.searchsorted(cdf, q, side="left")])

    def interval(self, central_mass: float) -> tuple[float, float]:
        tail = (1 - central_mass) / 2
        return self.quantile(tail), self.quantile(1 - tail)

    def threshold_ge(self, threshold_c: int) -> float:
        return float(self.probabilities[self.bins_c >= threshold_c].sum())

    def threshold_le(self, threshold_c: int) -> float:
        return float(self.probabilities[self.bins_c <= threshold_c].sum())

    def truncate_below(self, observed_max_so_far: float | None) -> "TmaxDistribution":
        if observed_max_so_far is None or np.isnan(observed_max_so_far):
            return self
        probs = self.probabilities.copy()
        probs[self.bins_c < np.ceil(observed_max_so_far)] = 0.0
        if probs.sum() <= 0:
            probs = np.zeros_like(probs)
            probs[np.argmax(self.bins_c >= np.ceil(observed_max_so_far))] = 1.0
        return TmaxDistribution(self.bins_c, probs)

    def to_payload(self) -> dict:
        return {
            "expected_tmax_c": self.expected_tmax_c,
            "median_tmax_c": self.median_tmax_c,
            "most_likely_integer_c": self.most_likely_integer_c,
            "intervals": {
                "50": list(self.interval(0.50)),
                "80": list(self.interval(0.80)),
                "90": list(self.interval(0.90)),
            },
            "probabilities_by_integer_c": {str(int(k)): float(v) for k, v in zip(self.bins_c, self.probabilities) if v > 1e-6},
            "threshold_probabilities": {
                "ge_20": self.threshold_ge(20),
                "ge_25": self.threshold_ge(25),
                "ge_30": self.threshold_ge(30),
                "le_0": self.threshold_le(0),
            },
        }


def fix_quantile_crossing(values: list[float] | np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(np.asarray(values, dtype=float))


def quantiles_to_distribution(
    quantiles: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
    bin_min: int = -35,
    bin_max: int = 45,
) -> TmaxDistribution:
    qs = np.asarray(quantiles, dtype=float)
    vals = fix_quantile_crossing(values)
    bins = np.arange(bin_min, bin_max + 1)
    edges = np.concatenate(([bin_min - 0.5], bins + 0.5))
    cdf_edges = np.interp(edges, vals, qs, left=0.0, right=1.0)
    cdf_edges = np.maximum.accumulate(np.clip(cdf_edges, 0.0, 1.0))
    probs = np.diff(cdf_edges)
    if probs.sum() <= 0:
        probs[np.argmin(np.abs(bins - np.median(vals)))] = 1.0
    return TmaxDistribution(bins, probs)


def empirical_distribution(samples, bin_min: int = -35, bin_max: int = 45) -> TmaxDistribution:
    bins = np.arange(bin_min, bin_max + 1)
    rounded = np.rint(np.asarray(samples, dtype=float)).astype(int)
    probs = np.array([(rounded == b).sum() for b in bins], dtype=float)
    if probs.sum() == 0:
        probs[bins == 15] = 1.0
    return TmaxDistribution(bins, probs)
