from __future__ import annotations

from typing import Any


def evaluate_forecast_acceptance(payload: dict[str, Any] | None = None, *, distribution=None, forecast_quality: dict | None = None) -> dict:
    payload = payload or {}
    quality = forecast_quality or payload.get("forecast_quality", {}) or {}
    checks = {
        "quality_status_ok": quality.get("status") == "ok",
        "has_no_hard_reasons": not quality.get("reasons", []),
        "probabilities_sum_to_one": _probabilities_sum_to_one(payload, distribution),
        "has_probability_bins": _has_probability_bins(payload, distribution),
    }
    blocking_reasons = [name for name, passed in checks.items() if not passed]
    return {
        "accepted": not blocking_reasons,
        "checks": checks,
        "blocking_reasons": blocking_reasons,
        "cautions": quality.get("cautions", []),
    }


def _probabilities_sum_to_one(payload: dict, distribution) -> bool:
    if distribution is not None:
        return abs(float(distribution.probabilities.sum()) - 1.0) < 1e-6
    probs = payload.get("probabilities_by_integer_c", {})
    return bool(probs) and abs(sum(float(value) for value in probs.values()) - 1.0) < 1e-6


def _has_probability_bins(payload: dict, distribution) -> bool:
    if distribution is not None:
        return len(distribution.bins_c) > 0
    return bool(payload.get("probabilities_by_integer_c"))
