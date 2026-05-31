from __future__ import annotations


def assess_forecast_quality(feature_snapshot: dict, warnings: list[str] | None = None) -> dict:
    warnings = warnings or []
    freshness = feature_snapshot.get("freshness", {}) or {}
    extrapolation = feature_snapshot.get("extrapolation", {}) or {}
    source_compatibility = feature_snapshot.get("source_compatibility", {}) or {}
    reasons = []
    cautions = []
    invalid = False

    for source, status in freshness.items():
        state = status.get("state")
        if state == "future_timestamp":
            invalid = True
            reasons.append(f"{source} timestamp is after issue_time")
        elif state in {"stale", "missing"}:
            reasons.append(f"{source} is {state}")

    if extrapolation.get("extrapolated") and extrapolation.get("severity") == "severe":
        reasons.append("live features outside training range")
    elif extrapolation.get("extrapolated"):
        cautions.append("minor live feature extrapolation")

    offset = feature_snapshot.get("issue_schedule_offset_minutes")
    if offset is not None:
        try:
            if float(offset) > 30:
                reasons.append("issue time is outside configured training schedule")
            elif float(offset) > 10:
                reasons.append("issue time is slightly off configured training schedule")
        except (TypeError, ValueError):
            pass

    unknown_source = any(item.get("status") == "unknown_runtime_source" for item in source_compatibility.values())
    known_compatible = any(item.get("status") == "known_runtime_compatible" for item in source_compatibility.values())
    if unknown_source:
        reasons.append("unknown runtime source differs from training source")
    elif known_compatible:
        cautions.append("known compatible runtime source differs from training source")
    elif any("Source mismatch warning" in warning for warning in warnings):
        reasons.append("runtime source differs from training source")

    if any("calibration layer is still preliminary" in warning for warning in warnings):
        cautions.append("calibration is preliminary")

    status = "invalid" if invalid else ("degraded" if reasons else "ok")
    return {
        "status": status,
        "reasons": sorted(set(reasons)),
        "cautions": sorted(set(cautions)),
        "recommendation": _recommendation(status, reasons, cautions),
    }


def _recommendation(status: str, reasons: list[str], cautions: list[str] | None = None) -> str | None:
    cautions = cautions or []
    if status == "invalid":
        return "Do not use this forecast operationally; inspect temporal filtering and source timestamps."
    if not reasons and not cautions:
        return None
    if any("stale" in reason or "missing" in reason for reason in reasons):
        return "Refresh operational data before relying on this forecast, e.g. run predict with --auto-refresh."
    if any("issue time" in reason for reason in reasons):
        return "Prefer configured issue times 00/03/06/09/12/15/18 UTC or ICON availability-aware +01:40 slots."
    if any("known compatible runtime source" in caution for caution in cautions):
        return "Use forecast with monitoring; runtime source is known-compatible but should be tracked separately."
    if any("minor live feature extrapolation" in caution for caution in cautions):
        return "Use forecast with monitoring; live feature is slightly outside the training envelope."
    return "Treat this forecast as usable but lower confidence; review warnings and monitoring reports."
