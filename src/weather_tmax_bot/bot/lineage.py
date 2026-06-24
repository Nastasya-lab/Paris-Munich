from __future__ import annotations

import json
from pathlib import Path

from weather_tmax_bot.models.model_registry import load_registry, resolve_active_artifacts


def build_data_lineage(feature_snapshot: dict) -> dict:
    nwp_runs = []
    if feature_snapshot.get("latest_nwp_model_name"):
        nwp_runs.append(
            {
                "model_name": feature_snapshot.get("latest_nwp_model_name"),
                "source_id": feature_snapshot.get("latest_nwp_source_id"),
                "knowledge_time_utc": _string_or_none(feature_snapshot.get("max_nwp_knowledge_time_utc")),
            }
        )
    truth_source = (
        "iem.metar.archive.EDDM / awc.metar.live.EDDM"
        if feature_snapshot.get("target") == "METAR_Tmax"
        else "dwd.10min.air_temperature.01262"
    )
    return {
        "truth_source": truth_source,
        "metar_source_id": _string_or_none(feature_snapshot.get("latest_metar_source_id")),
        "metar_latest_time_utc": _string_or_none(feature_snapshot.get("latest_metar_time_utc")),
        "taf_source_id": _string_or_none(feature_snapshot.get("latest_taf_source_id")),
        "taf_issue_time_utc": _string_or_none(feature_snapshot.get("latest_taf_issue_time_utc")),
        "nwp_runs": nwp_runs,
        "max_feature_knowledge_time_utc": _max_time_string(
            feature_snapshot.get("max_metar_knowledge_time_utc"),
            feature_snapshot.get("max_nwp_knowledge_time_utc"),
            feature_snapshot.get("latest_taf_issue_time_utc"),
        ),
    }


def model_info(model_dir: str | Path = "data/models") -> dict:
    path = Path(model_dir)
    registry = load_registry(model_dir)
    active = resolve_active_artifacts(model_dir=model_dir)
    info = {
        "models": [],
        "registry": registry,
        "active_model": {
            "model_version": active.get("active_model_version"),
            "calibrator_version": active.get("active_calibrator_version"),
            "model_path": _path_string(active.get("model_path")),
            "calibrator_path": _path_string(active.get("calibrator_path")),
            "model_exists": active.get("model_path") is not None,
            "calibrator_exists": active.get("calibrator_path") is not None,
            "promotion": registry.get("promotion", {}),
        },
    }
    for metadata_path in sorted(path.glob("*.metadata.json")):
        try:
            info["models"].append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            info["models"].append({"metadata_path": str(metadata_path), "error": "invalid_json"})
    return info


def _string_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text in ("NaT", "nan", "None") else text


def _max_time_string(*values) -> str | None:
    cleaned = [_string_or_none(value) for value in values]
    cleaned = [value for value in cleaned if value is not None]
    return max(cleaned) if cleaned else None


def _path_string(path: Path | None) -> str | None:
    return None if path is None else str(path)
