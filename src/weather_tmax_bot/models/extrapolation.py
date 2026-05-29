from __future__ import annotations

import math
from typing import Any


def detect_feature_extrapolation(
    feature_snapshot: dict,
    model: Any,
    tolerance_fraction: float = 0.05,
    max_reported: int = 8,
) -> dict:
    ignored_features = {"issue_minute_utc", "issue_schedule_offset_minutes", "issue_off_schedule"}
    soft_features = {
        "observed_max_so_far_from_metar",
        "observed_min_so_far_from_metar",
        "last_metar_temp_c",
        "last_metar_dewpoint_c",
        "dewpoint_depression",
    }
    ranges = getattr(model, "feature_ranges", {}) or {}
    if not ranges:
        return {"extrapolated": False, "violations": [], "warnings": []}
    violations = []
    for feature, bounds in ranges.items():
        if feature in ignored_features:
            continue
        value = feature_snapshot.get(feature)
        if value is None or isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        lower = float(bounds["min"])
        upper = float(bounds["max"])
        span = max(upper - lower, 1.0)
        margin = span * tolerance_fraction
        if numeric < lower - margin or numeric > upper + margin:
            violations.append({"feature": feature, "value": numeric, "training_min": lower, "training_max": upper})
    reported = violations[:max_reported]
    warnings = []
    if violations:
        warnings.append(
            f"Extrapolation warning: {len(violations)} feature(s) outside training range; "
            f"examples: {', '.join(item['feature'] for item in reported)}."
        )
    severity = _severity(violations, soft_features)
    return {"extrapolated": bool(violations), "severity": severity, "violations": reported, "warnings": warnings}


def _severity(violations: list[dict], soft_features: set[str]) -> str:
    if not violations:
        return "none"
    if len(violations) <= 2 and all(item["feature"] in soft_features for item in violations):
        return "minor"
    return "severe"
