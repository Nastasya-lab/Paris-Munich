from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DisagreementThresholds:
    expected_watch_c: float = 1.5
    expected_high_c: float = 3.0
    threshold_watch_pp: float = 0.15
    threshold_high_pp: float = 0.30
    bin_watch_c: int = 2
    bin_high_c: int = 4


def assess_model_disagreement(
    forecast_variants: dict,
    *,
    thresholds: DisagreementThresholds | None = None,
) -> dict:
    """Compare operational and shadow distributions without changing the forecast."""

    thresholds = thresholds or DisagreementThresholds()
    variants = _extract_variants(forecast_variants)
    champion = variants.get("production_champion")
    if not champion:
        return {
            "status": "unavailable",
            "severity": "none",
            "reasons": ["production champion distribution missing"],
            "variants": variants,
        }

    comparisons = {}
    for name, payload in variants.items():
        if name == "production_champion":
            continue
        comparisons[name] = _compare(champion, payload)

    expected_spread = _range([item["expected_tmax_c"] for item in variants.values()])
    most_likely_bin_spread = _range([item["most_likely_integer_c"] for item in variants.values()])
    ge_25_spread = _range([item["threshold_probabilities"].get("ge_25") for item in variants.values()])
    ge_30_spread = _range([item["threshold_probabilities"].get("ge_30") for item in variants.values()])
    reasons = []
    severity_score = 0
    if expected_spread >= thresholds.expected_high_c:
        severity_score = max(severity_score, 2)
        reasons.append("expected_tmax_spread_high")
    elif expected_spread >= thresholds.expected_watch_c:
        severity_score = max(severity_score, 1)
        reasons.append("expected_tmax_spread_watch")
    if most_likely_bin_spread >= thresholds.bin_high_c:
        severity_score = max(severity_score, 2)
        reasons.append("most_likely_bin_spread_high")
    elif most_likely_bin_spread >= thresholds.bin_watch_c:
        severity_score = max(severity_score, 1)
        reasons.append("most_likely_bin_spread_watch")
    threshold_spread = max(ge_25_spread, ge_30_spread)
    if threshold_spread >= thresholds.threshold_high_pp:
        severity_score = max(severity_score, 2)
        reasons.append("threshold_probability_spread_high")
    elif threshold_spread >= thresholds.threshold_watch_pp:
        severity_score = max(severity_score, 1)
        reasons.append("threshold_probability_spread_watch")

    return {
        "status": "evaluated",
        "severity": {0: "none", 1: "watch", 2: "high"}[severity_score],
        "reasons": reasons,
        "summary": {
            "expected_tmax_spread_c": expected_spread,
            "most_likely_bin_spread_c": most_likely_bin_spread,
            "ge_25_probability_spread": ge_25_spread,
            "ge_30_probability_spread": ge_30_spread,
            "variant_count": len(variants),
        },
        "variants": variants,
        "comparison_to_champion": comparisons,
    }


def _extract_variants(forecast_variants: dict) -> dict:
    variants = {}
    comparable = {"production_champion", "shadow_seasonal_intraday", "shadow_intraday_ml"}
    for name, payload in (forecast_variants or {}).items():
        if name not in comparable:
            continue
        distribution = payload.get("distribution") or {}
        if not distribution:
            continue
        expected = distribution.get("expected_tmax_c")
        most_likely = distribution.get("most_likely_integer_c")
        thresholds = distribution.get("threshold_probabilities") or {}
        if expected is None or most_likely is None:
            continue
        variants[name] = {
            "expected_tmax_c": float(expected),
            "median_tmax_c": _optional_float(distribution.get("median_tmax_c")),
            "most_likely_integer_c": int(most_likely),
            "threshold_probabilities": {
                "ge_20": _optional_float(thresholds.get("ge_20"), 0.0),
                "ge_25": _optional_float(thresholds.get("ge_25"), 0.0),
                "ge_30": _optional_float(thresholds.get("ge_30"), 0.0),
                "le_0": _optional_float(thresholds.get("le_0"), 0.0),
            },
        }
    return variants


def _compare(champion: dict, challenger: dict) -> dict:
    champion_thresholds = champion["threshold_probabilities"]
    challenger_thresholds = challenger["threshold_probabilities"]
    return {
        "expected_tmax_delta_c": challenger["expected_tmax_c"] - champion["expected_tmax_c"],
        "most_likely_integer_delta_c": challenger["most_likely_integer_c"] - champion["most_likely_integer_c"],
        "ge_25_probability_delta": challenger_thresholds.get("ge_25", 0.0) - champion_thresholds.get("ge_25", 0.0),
        "ge_30_probability_delta": challenger_thresholds.get("ge_30", 0.0) - champion_thresholds.get("ge_30", 0.0),
    }


def _range(values: list[float | int | None]) -> float:
    clean = [float(value) for value in values if value is not None]
    if len(clean) < 2:
        return 0.0
    return max(clean) - min(clean)


def _optional_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    return float(value)
