from __future__ import annotations


TRAINING_SOURCES = {
    "metar": "iem.metar.archive.EDDM",
    "taf": "iem.taf.archive.EDDM",
}

KNOWN_COMPATIBLE_RUNTIME_SOURCES = {
    ("metar", "awc.metar.live.EDDM", "iem.metar.archive.EDDM"),
    ("taf", "awc.taf.live.EDDM", "iem.taf.archive.EDDM"),
}


def assess_source_compatibility(feature_snapshot: dict) -> dict:
    assessments = {
        "metar": _assess_one("metar", feature_snapshot.get("latest_metar_source_id")),
        "taf": _assess_one("taf", feature_snapshot.get("latest_taf_source_id")),
    }
    warnings = []
    for kind, item in assessments.items():
        if item["status"] == "unknown_runtime_source":
            warnings.append(
                f"Source compatibility warning: {kind.upper()} runtime source {item['runtime_source_id']} "
                f"is not a known compatible source for training source {item['training_source_id']}."
            )
        elif item["status"] == "known_runtime_compatible":
            warnings.append(
                f"Source compatibility note: {kind.upper()} runtime source {item['runtime_source_id']} differs from "
                f"training source {item['training_source_id']} but is marked known-compatible; monitor performance."
            )
    return {"sources": assessments, "warnings": warnings}


def _assess_one(kind: str, runtime_source_id: str | None) -> dict:
    training = TRAINING_SOURCES[kind]
    if not runtime_source_id:
        return {
            "status": "missing",
            "runtime_source_id": None,
            "training_source_id": training,
            "known_compatible": False,
        }
    if runtime_source_id == training:
        return {
            "status": "same_source",
            "runtime_source_id": runtime_source_id,
            "training_source_id": training,
            "known_compatible": True,
        }
    known = (kind, runtime_source_id, training) in KNOWN_COMPATIBLE_RUNTIME_SOURCES
    return {
        "status": "known_runtime_compatible" if known else "unknown_runtime_source",
        "runtime_source_id": runtime_source_id,
        "training_source_id": training,
        "known_compatible": known,
    }
