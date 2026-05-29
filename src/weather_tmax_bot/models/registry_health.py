from __future__ import annotations

from pathlib import Path

from weather_tmax_bot.models.model_registry import load_registry, resolve_active_artifacts


def registry_health(
    model_dir: str | Path = "data/models",
    fallback_model_path: str | Path = "data/models/quantile_mvp.joblib",
) -> dict:
    registry = load_registry(model_dir)
    active = resolve_active_artifacts(model_dir=model_dir, fallback_model_path=fallback_model_path)
    fallback = Path(fallback_model_path)
    has_active_version = active.get("active_model_version") is not None
    checks = {
        "registry_loaded": True,
        "active_or_fallback_model_exists": active["model_path"] is not None and active["model_path"].exists(),
        "fallback_model_exists": fallback.exists() if not has_active_version else True,
        "active_calibrator_optional_or_exists": active["calibrator_path"] is None or active["calibrator_path"].exists(),
    }
    active_version = active.get("active_model_version")
    if active_version is not None:
        checks["active_version_registered"] = any(
            entry.get("version") == active_version and entry.get("artifact_type") == "model"
            for entry in registry.get("entries", [])
        )
    else:
        checks["active_version_registered"] = fallback.exists()
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_model_version": active.get("active_model_version"),
        "active_calibrator_version": active.get("active_calibrator_version"),
        "model_path": None if active["model_path"] is None else str(active["model_path"]),
        "calibrator_path": None if active["calibrator_path"] is None else str(active["calibrator_path"]),
    }
