from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib


def save_model(model, metadata: dict, model_dir: str | Path = "data/models") -> Path:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    version = metadata.get("model_version", "mvp")
    model_path = path / f"{version}.joblib"
    joblib.dump(model, model_path)
    (path / f"{version}.metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return model_path


def load_model(model_path: str | Path):
    return joblib.load(model_path)


def registry_path(model_dir: str | Path = "data/models") -> Path:
    return Path(model_dir) / "model_registry.json"


def load_registry(model_dir: str | Path = "data/models") -> dict[str, Any]:
    path = registry_path(model_dir)
    if not path.exists():
        return {"active_model_version": None, "active_calibrator_version": None, "entries": []}
    return json.loads(path.read_text(encoding="utf-8"))


def write_registry(registry: dict[str, Any], model_dir: str | Path = "data/models") -> Path:
    path = registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")
    return path


def register_artifact(
    *,
    version: str,
    artifact_type: str,
    path: str | Path,
    metadata_path: str | Path | None = None,
    metrics: dict | None = None,
    model_dir: str | Path = "data/models",
) -> dict:
    registry = load_registry(model_dir)
    entry = {
        "version": version,
        "artifact_type": artifact_type,
        "path": str(path),
        "metadata_path": None if metadata_path is None else str(metadata_path),
        "metrics": metrics or {},
    }
    registry["entries"] = [
        existing
        for existing in registry.get("entries", [])
        if not (existing.get("version") == version and existing.get("artifact_type") == artifact_type)
    ]
    registry["entries"].append(entry)
    write_registry(registry, model_dir)
    return entry


def promote_model(
    *,
    model_version: str,
    calibrator_version: str | None = None,
    reason: str = "manual",
    metrics: dict | None = None,
    model_dir: str | Path = "data/models",
) -> dict:
    registry = load_registry(model_dir)
    registry["active_model_version"] = model_version
    registry["active_calibrator_version"] = calibrator_version
    registry["promotion"] = {
        "reason": reason,
        "metrics": metrics or {},
    }
    write_registry(registry, model_dir)
    return registry


def resolve_active_artifacts(
    model_dir: str | Path = "data/models",
    fallback_model_path: str | Path = "data/models/quantile_mvp.joblib",
) -> dict[str, Path | None]:
    registry = load_registry(model_dir)
    entries = registry.get("entries", [])
    active_model = registry.get("active_model_version")
    active_calibrator = registry.get("active_calibrator_version")

    model_path = _path_for(entries, active_model, "model") if active_model else None
    if model_path is None:
        fallback = Path(fallback_model_path)
        model_path = fallback if fallback.exists() else None

    calibrator_path = _path_for(entries, active_calibrator, "calibrator") if active_calibrator else None
    if calibrator_path is None and model_path is not None:
        inferred = Path(str(model_path).replace(".joblib", ".calibrator.joblib"))
        calibrator_path = inferred if inferred.exists() else None

    return {
        "model_path": model_path,
        "calibrator_path": calibrator_path,
        "active_model_version": active_model,
        "active_calibrator_version": active_calibrator,
    }


def _path_for(entries: list[dict], version: str | None, artifact_type: str) -> Path | None:
    if version is None:
        return None
    for entry in entries:
        if entry.get("version") == version and entry.get("artifact_type") == artifact_type:
            path = _stored_path(entry["path"])
            return path if path.exists() else None
    return None


def _stored_path(value: str) -> Path:
    return Path(value.replace("\\", os.sep).replace("/", os.sep))
