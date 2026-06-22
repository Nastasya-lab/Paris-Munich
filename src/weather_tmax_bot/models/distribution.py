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


def temperature_scale_distribution(distribution: TmaxDistribution, temperature: float) -> TmaxDistribution:
    """Rescale PMF confidence without changing the support."""
    temp = float(np.clip(temperature, 0.35, 5.0))
    probs = np.clip(distribution.probabilities, 1e-12, 1.0)
    scaled = np.exp(np.log(probs) / temp)
    return TmaxDistribution(distribution.bins_c, scaled)


def project_unimodal_distribution(distribution: TmaxDistribution) -> TmaxDistribution:
    """Project a discrete PMF to the closest least-squares unimodal PMF."""
    probs = np.asarray(distribution.probabilities, dtype=float)
    if len(probs) <= 2:
        return distribution
    best = None
    best_error = np.inf
    for mode_idx in range(len(probs)):
        left = _pava(probs[: mode_idx + 1], increasing=True)
        right = _pava(probs[mode_idx:], increasing=False)
        candidate = np.concatenate([left[:-1], [(left[-1] + right[0]) / 2.0], right[1:]])
        candidate = np.clip(candidate, 0.0, np.inf)
        if candidate.sum() <= 0:
            continue
        candidate = candidate / candidate.sum()
        error = float(np.sum((candidate - probs) ** 2))
        if error < best_error:
            best_error = error
            best = candidate
    if best is None:
        return distribution
    return TmaxDistribution(distribution.bins_c, best)


def unimodal_violation_count(distribution: TmaxDistribution) -> int:
    probs = np.asarray(distribution.probabilities, dtype=float)
    if len(probs) <= 2:
        return 0
    mode = int(np.argmax(probs))
    left_bad = int(np.sum(np.diff(probs[: mode + 1]) < -1e-9))
    right_bad = int(np.sum(np.diff(probs[mode:]) > 1e-9))
    return left_bad + right_bad


def _pava(values: np.ndarray, *, increasing: bool) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if len(data) <= 1:
        return data.copy()
    work = data if increasing else -data
    levels: list[float] = []
    weights: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, value in enumerate(work):
        levels.append(float(value))
        weights.append(1)
        starts.append(idx)
        ends.append(idx + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = merged
            weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()
    out = np.empty(len(work), dtype=float)
    for level, start, end in zip(levels, starts, ends):
        out[start:end] = level
    return out if increasing else -out


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
