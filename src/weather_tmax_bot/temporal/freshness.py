from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_THRESHOLDS_HOURS = {
    "metar": 2.0,
    "taf": 12.0,
    "nwp": 18.0,
}


def assess_feature_freshness(
    feature_snapshot: dict,
    issue_time_utc: datetime,
    thresholds_hours: dict[str, float] | None = None,
) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS_HOURS, **(thresholds_hours or {})}
    statuses = {
        "metar": _status(
            "METAR",
            feature_snapshot.get("latest_metar_time_utc"),
            issue_time_utc,
            thresholds["metar"],
            bool(feature_snapshot.get("metar_missing", False)),
        ),
        "taf": (
            _not_required_status("TAF")
            if feature_snapshot.get("taf_not_required")
            else _status(
                "TAF",
                feature_snapshot.get("latest_taf_issue_time_utc"),
                issue_time_utc,
                thresholds["taf"],
                bool(feature_snapshot.get("taf_missing", False)),
            )
        ),
        "nwp": _status(
            "NWP",
            feature_snapshot.get("max_nwp_knowledge_time_utc"),
            issue_time_utc,
            thresholds["nwp"],
            bool(feature_snapshot.get("nwp_missing", False)),
        ),
    }
    warnings = [status["warning"] for status in statuses.values() if status.get("warning")]
    return {"statuses": statuses, "warnings": warnings}


def _not_required_status(label: str) -> dict:
    return {
        "state": "not_required",
        "latest_time_utc": None,
        "age_hours": None,
        "max_age_hours": None,
        "warning": None,
        "label": label,
    }


def assess_archive_freshness(root: str | Path = ".", issue_time_utc: datetime | None = None) -> dict:
    root = Path(root)
    issue = issue_time_utc or datetime.now(timezone.utc)
    snapshots = {
        "metar": _latest_value(root / "data/forecasts/awc_metar_live_EDDM.parquet", "observation_time_utc"),
        "taf": _latest_value(root / "data/forecasts/awc_taf_live_EDDM.parquet", "issue_time_utc"),
        "nwp": _latest_value(root / "data/forecasts/open_meteo_archive.parquet", "knowledge_time_utc"),
    }
    feature_snapshot = {
        "latest_metar_time_utc": snapshots["metar"],
        "latest_taf_issue_time_utc": snapshots["taf"],
        "max_nwp_knowledge_time_utc": snapshots["nwp"],
        "metar_missing": snapshots["metar"] is None,
        "taf_missing": snapshots["taf"] is None,
        "nwp_missing": snapshots["nwp"] is None,
    }
    return assess_feature_freshness(feature_snapshot, issue)


def _status(label: str, value, issue_time_utc: datetime, max_age_hours: float, missing: bool) -> dict:
    timestamp = _to_timestamp(value)
    if missing or timestamp is None:
        return {
            "state": "missing",
            "latest_time_utc": None,
            "age_hours": None,
            "max_age_hours": max_age_hours,
            "warning": f"{label} data is missing for this as-of feature view.",
        }
    issue = pd.Timestamp(issue_time_utc).tz_convert("UTC") if pd.Timestamp(issue_time_utc).tzinfo else pd.Timestamp(issue_time_utc, tz="UTC")
    age_hours = (issue - timestamp).total_seconds() / 3600
    if age_hours < 0:
        return {
            "state": "future_timestamp",
            "latest_time_utc": timestamp.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "warning": f"{label} timestamp is after issue_time; check temporal filtering.",
        }
    if age_hours > max_age_hours:
        return {
            "state": "stale",
            "latest_time_utc": timestamp.isoformat(),
            "age_hours": age_hours,
            "max_age_hours": max_age_hours,
            "warning": f"{label} data is stale: age {age_hours:.1f}h exceeds {max_age_hours:.1f}h.",
        }
    return {
        "state": "fresh",
        "latest_time_utc": timestamp.isoformat(),
        "age_hours": age_hours,
        "max_age_hours": max_age_hours,
        "warning": None,
    }


def _latest_value(path: Path, column: str):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or column not in df.columns:
        return None
    values = pd.to_datetime(df[column], utc=True, errors="coerce").dropna()
    if values.empty:
        return None
    return values.max()


def _to_timestamp(value) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)
