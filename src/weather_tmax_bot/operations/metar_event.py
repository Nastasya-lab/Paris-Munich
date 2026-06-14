from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from weather_tmax_bot.notifications.telegram import format_metar_event_message, notify_if_configured
from weather_tmax_bot.operations.predict_run import run_prediction
from weather_tmax_bot.operations.refresh import refresh_awc_live, refresh_spatial_awc_live
from weather_tmax_bot.features.spatial_metar import SPATIAL_STATIONS_BY_AIRPORT
from weather_tmax_bot.operations.run_report import operational_prediction_payload


DEFAULT_NOTIFY_THRESHOLDS = {
    "expected_delta_c": 0.7,
    "threshold_delta": 0.10,
    "shadow_expected_gap_c": 1.0,
    "drop_from_max_c": 3.0,
}


def run_metar_event_cycle(
    *,
    airport: str,
    target_date_local: date,
    issue_time_utc: datetime,
    notify: bool = True,
    log: bool = True,
    root: str | Path = ".",
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
) -> dict:
    """Refresh live METAR and emit an intraday update for every new report.

    This is intentionally separate from the ICON/NWP scheduled cycle. It is a
    light polling job: no new METAR means no forecast, no log row, and no chat
    noise. A new METAR is always logged and, when notifications are enabled,
    sent to Telegram so the intraday probability path is visible.
    """
    root = Path(root)
    forecast_log_path = Path(os.getenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(forecast_log_path)))
    before_metar = _latest_metar_time(root, airport)
    refresh_summary = {"airport": airport, "target_date_local": target_date_local.isoformat(), "sources": {"awc": refresh_awc_live(airport, root)}}
    if SPATIAL_STATIONS_BY_AIRPORT.get(airport.upper()):
        refresh_summary["sources"]["spatial_awc"] = refresh_spatial_awc_live(airport=airport, root=root)
    after_metar = _latest_metar_time(root, airport)
    latest_metar_record = _latest_metar_record(root, airport)
    if after_metar is None:
        return {
            "status": "no_metar_available",
            "airport": airport,
            "target_date_local": target_date_local.isoformat(),
            "issue_time_utc": issue_time_utc.isoformat(),
            "latest_metar_time_utc": None,
            "refresh_summary": refresh_summary,
            "forecast_logged": False,
            "notification_sent": False,
        }
    if before_metar is not None and after_metar <= before_metar:
        return {
            "status": "no_new_metar",
            "airport": airport,
            "target_date_local": target_date_local.isoformat(),
            "issue_time_utc": issue_time_utc.isoformat(),
            "previous_metar_time_utc": before_metar.isoformat(),
            "latest_metar_time_utc": after_metar.isoformat(),
            "refresh_summary": refresh_summary,
            "forecast_logged": False,
            "notification_sent": False,
        }

    effective_issue_time_utc = _effective_issue_time(issue_time_utc, latest_metar_record)
    previous_record = _latest_forecast_record(forecast_log_path, airport=airport, target_date_local=target_date_local)
    result = run_prediction(
        airport=airport,
        target_date_local=target_date_local,
        issue_time_utc=effective_issue_time_utc,
        log=log,
        mode="metar_event",
    )
    payload = operational_prediction_payload(
        airport=airport,
        target_date_local=target_date_local,
        issue_time_utc=effective_issue_time_utc,
        result=result,
    )
    payload["refresh_summary"] = refresh_summary
    payload["latest_metar_record"] = latest_metar_record
    comparison = compare_forecast_to_previous(payload, previous_record)
    should_notify, notification_reasons = should_notify_metar_event(payload, comparison)
    telegram_notification = None
    if notify and should_notify:
        telegram_notification = notify_if_configured(format_metar_event_message(payload, comparison, notification_reasons))
    return {
        "status": "new_metar_forecast",
        "airport": airport,
        "target_date_local": target_date_local.isoformat(),
        "issue_time_utc": effective_issue_time_utc.isoformat(),
        "requested_issue_time_utc": issue_time_utc.isoformat(),
        "previous_metar_time_utc": None if before_metar is None else before_metar.isoformat(),
        "latest_metar_time_utc": after_metar.isoformat(),
        "latest_metar_record": latest_metar_record,
        "forecast_id": result["forecast_id"],
        "model_version": result["metadata"]["model_version"],
        "forecast": {
            "expected_tmax_c": payload["expected_tmax_c"],
            "median_tmax_c": payload["median_tmax_c"],
            "most_likely_integer_c": payload["most_likely_integer_c"],
            "threshold_probabilities": payload["threshold_probabilities"],
            "forecast_components": payload.get("forecast_components", {}),
        },
        "comparison_to_previous": comparison,
        "notification_reasons": notification_reasons,
        "notification_needed": should_notify,
        "notification_sent": bool((telegram_notification or {}).get("sent", False)),
        "telegram_notification": telegram_notification,
        "refresh_summary": refresh_summary,
        "forecast_quality": result["forecast_quality"],
        "forecast_acceptance": result["forecast_acceptance"],
        "warnings": result["warnings"],
    }


def compare_forecast_to_previous(current: dict, previous_record: dict | None) -> dict:
    current_summary = _forecast_summary_from_payload(current)
    previous_summary = _forecast_summary_from_record(previous_record) if previous_record else None
    if not previous_summary:
        return {
            "has_previous": False,
            "current": current_summary,
            "previous": None,
            "deltas": {},
        }
    deltas = {
        "expected_tmax_delta_c": current_summary["expected_tmax_c"] - previous_summary["expected_tmax_c"],
        "median_tmax_delta_c": current_summary["median_tmax_c"] - previous_summary["median_tmax_c"],
        "most_likely_integer_changed": current_summary["most_likely_integer_c"] != previous_summary["most_likely_integer_c"],
        "most_likely_integer_delta_c": current_summary["most_likely_integer_c"] - previous_summary["most_likely_integer_c"],
    }
    for key in ("ge_20", "ge_25", "ge_30", "le_0"):
        deltas[f"{key}_delta"] = current_summary["threshold_probabilities"].get(key, 0.0) - previous_summary[
            "threshold_probabilities"
        ].get(key, 0.0)
    return {
        "has_previous": True,
        "current": current_summary,
        "previous": previous_summary,
        "deltas": deltas,
    }


def should_notify_metar_event(
    payload: dict,
    comparison: dict,
    *,
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, list[str]]:
    thresholds = {**DEFAULT_NOTIFY_THRESHOLDS, **(thresholds or {})}
    reasons: list[str] = []
    deltas = comparison.get("deltas", {}) or {}
    if not comparison.get("has_previous"):
        reasons.append("first_new_metar_forecast")
    if abs(float(deltas.get("expected_tmax_delta_c", 0.0))) >= thresholds["expected_delta_c"]:
        reasons.append("expected_tmax_changed")
    if deltas.get("most_likely_integer_changed"):
        reasons.append("most_likely_bin_changed")
    for key in ("ge_20_delta", "ge_25_delta", "ge_30_delta"):
        if abs(float(deltas.get(key, 0.0))) >= thresholds["threshold_delta"]:
            reasons.append(f"{key.removesuffix('_delta')}_probability_changed")

    components = payload.get("forecast_components", {}) or {}
    intraday = components.get("intraday_update", {}) or {}
    if float(intraday.get("drop_from_observed_max_c") or 0.0) >= thresholds["drop_from_max_c"]:
        reasons.append("temperature_dropped_from_observed_max")
    if float(intraday.get("peak_passed_probability") or 0.0) >= 0.85:
        reasons.append("peak_likely_passed")

    shadow = components.get("shadow_mode", {}) or {}
    shadow_gap = abs(float((shadow.get("comparison_to_champion") or {}).get("expected_tmax_delta_c") or 0.0))
    if shadow_gap >= thresholds["shadow_expected_gap_c"]:
        reasons.append("shadow_differs_from_champion")
    if not reasons:
        reasons.append("routine_new_metar_update")
    return True, sorted(set(reasons))


def _latest_metar_time(root: Path, airport: str) -> pd.Timestamp | None:
    path = root / f"data/forecasts/awc_metar_live_{airport}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "observation_time_utc" not in df.columns:
        return None
    times = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce").dropna()
    if times.empty:
        return None
    return times.max()


def _latest_metar_record(root: Path, airport: str) -> dict[str, Any] | None:
    path = root / f"data/forecasts/awc_metar_live_{airport}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df.empty or "observation_time_utc" not in df.columns:
        return None
    out = df.copy()
    out["observation_time_utc"] = pd.to_datetime(out["observation_time_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["observation_time_utc"]).sort_values("observation_time_utc")
    if out.empty:
        return None
    row = out.iloc[-1]
    fields = {
        "observation_time_utc": row.get("observation_time_utc"),
        "knowledge_time_utc": row.get("knowledge_time_utc"),
        "ingest_time_utc": row.get("ingest_time_utc"),
        "temperature_c": row.get("temperature_c"),
        "dewpoint_c": row.get("dewpoint_c"),
        "raw_metar": row.get("raw_metar"),
        "source_id": row.get("source_id"),
    }
    return {key: _json_safe_value(value) for key, value in fields.items() if _json_safe_value(value) is not None}


def _effective_issue_time(issue_time_utc: datetime, latest_metar_record: dict[str, Any] | None) -> datetime:
    candidates = [pd.Timestamp(issue_time_utc)]
    for key in ("knowledge_time_utc", "ingest_time_utc"):
        value = (latest_metar_record or {}).get(key)
        if value is not None:
            candidates.append(pd.Timestamp(value))
    return max(timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC") for timestamp in candidates).to_pydatetime()


def _json_safe_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        return value.isoformat()
    if isinstance(value, (int, float, str, bool)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _latest_forecast_record(path: Path, *, airport: str, target_date_local: date) -> dict | None:
    if not path.exists():
        return None
    latest = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("airport") == airport and record.get("target_date_local") == target_date_local.isoformat():
            latest = record
    return latest


def _forecast_summary_from_payload(payload: dict) -> dict:
    distribution = _distribution_from_mapping(payload.get("probabilities_by_integer_c") or {})
    return {
        "forecast_id": payload.get("forecast_id"),
        "issue_time_utc": payload.get("issue_time_utc"),
        "expected_tmax_c": float(payload.get("expected_tmax_c", 0.0)),
        "median_tmax_c": float(payload.get("median_tmax_c", 0.0)),
        "most_likely_integer_c": int(payload.get("most_likely_integer_c", 0)),
        "threshold_probabilities": payload.get("threshold_probabilities", {}),
        "probabilities_by_integer_c": {str(k): v for k, v in distribution.items()},
    }


def _forecast_summary_from_record(record: dict | None) -> dict | None:
    if not record:
        return None
    distribution = _distribution_from_mapping(record.get("probability_distribution") or {})
    return {
        "forecast_id": record.get("forecast_id"),
        "issue_time_utc": record.get("issue_time_utc"),
        "expected_tmax_c": float(record.get("expected_tmax_c", 0.0)),
        "median_tmax_c": float(record.get("median_tmax_c", 0.0)),
        "most_likely_integer_c": int(record.get("most_likely_integer_c", 0)),
        "probabilities_by_integer_c": {str(k): v for k, v in distribution.items()},
        "threshold_probabilities": {
            "ge_20": _threshold_ge(distribution, 20),
            "ge_25": _threshold_ge(distribution, 25),
            "ge_30": _threshold_ge(distribution, 30),
            "le_0": sum(prob for bin_c, prob in distribution.items() if bin_c <= 0),
        },
    }


def _distribution_from_mapping(values: dict) -> dict[int, float]:
    distribution: dict[int, float] = {}
    for key, value in values.items():
        try:
            probability = float(value)
            if probability > 0:
                distribution[int(key)] = probability
        except (TypeError, ValueError):
            continue
    return dict(sorted(distribution.items()))


def _threshold_ge(distribution: dict[int, float], threshold: int) -> float:
    return sum(prob for bin_c, prob in distribution.items() if bin_c >= threshold)
