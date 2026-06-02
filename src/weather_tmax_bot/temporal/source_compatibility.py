from __future__ import annotations


TRAINING_SOURCE_PROFILES = {
    "metar": {
        "training_source_id": "iem.metar.archive.EDDM",
        "allowed_runtime_sources": {"iem.metar.archive.EDDM", "awc.metar.live.EDDM"},
        "known_compatible_runtime_sources": {"awc.metar.live.EDDM"},
    },
    "taf": {
        "training_source_id": "iem.taf.archive.EDDM",
        "allowed_runtime_sources": {"iem.taf.archive.EDDM", "awc.taf.live.EDDM"},
        "known_compatible_runtime_sources": {"awc.taf.live.EDDM"},
    },
    "nwp": {
        "training_source_id": "open_meteo.single_run.icon_d2",
        "allowed_runtime_sources": {"open_meteo.single_run.icon_d2", "open_meteo.live.icon_d2"},
        "known_compatible_runtime_sources": {"open_meteo.live.icon_d2"},
    },
}


def assess_source_compatibility(feature_snapshot: dict) -> dict:
    assessments = {
        "metar": _assess_one("metar", feature_snapshot.get("latest_metar_source_id")),
        "taf": _assess_one("taf", feature_snapshot.get("latest_taf_source_id")),
        "nwp": _assess_one("nwp", feature_snapshot.get("latest_nwp_source_id")),
    }
    warnings = []
    for kind, item in assessments.items():
        if item["status"] == "unknown_mismatch":
            warnings.append(
                f"Source compatibility warning: {kind.upper()} runtime source {item['runtime_source_id']} "
                f"is not a known compatible source for training source {item['training_source_id']}."
            )
        elif item["status"] == "forbidden_mismatch":
            warnings.append(
                f"Source compatibility error: {kind.upper()} runtime source {item['runtime_source_id']} "
                f"is forbidden for training source {item['training_source_id']}."
            )
        elif item["status"] == "known_compatible":
            warnings.append(
                f"Source compatibility note: {kind.upper()} runtime source {item['runtime_source_id']} differs from "
                f"training source {item['training_source_id']} but is marked known-compatible; monitor performance."
            )
    return {"sources": assessments, "warnings": warnings}


def _assess_one(kind: str, runtime_source_id: str | None) -> dict:
    profile = TRAINING_SOURCE_PROFILES[kind]
    training = profile["training_source_id"]
    if not runtime_source_id:
        return {
            "status": "missing",
            "role": kind,
            "runtime_source_id": None,
            "training_source_id": training,
            "compatibility_class": "missing",
            "known_compatible": False,
            "allowed_for_runtime": False,
            "blocking": False,
        }
    if runtime_source_id == training:
        return {
            "status": "exact_match",
            "role": kind,
            "runtime_source_id": runtime_source_id,
            "training_source_id": training,
            "compatibility_class": "exact",
            "known_compatible": True,
            "allowed_for_runtime": True,
            "blocking": False,
        }
    allowed = runtime_source_id in profile["allowed_runtime_sources"]
    known = runtime_source_id in profile["known_compatible_runtime_sources"]
    if allowed and known:
        status = "known_compatible"
        compatibility_class = "compatible_runtime_substitute"
        blocking = False
    elif allowed:
        status = "unknown_mismatch"
        compatibility_class = "allowed_but_unverified"
        blocking = False
    else:
        status = "forbidden_mismatch"
        compatibility_class = "forbidden"
        blocking = True
    return {
        "status": status,
        "role": kind,
        "runtime_source_id": runtime_source_id,
        "training_source_id": training,
        "compatibility_class": compatibility_class,
        "known_compatible": known,
        "allowed_for_runtime": allowed,
        "blocking": blocking,
    }
