from __future__ import annotations

import os
from typing import Any

import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def telegram_configured() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_telegram_message(text: str, *, parse_mode: str | None = None, timeout: int = 15) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "telegram_not_configured"}
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text[:3900],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    response = requests.post(TELEGRAM_API.format(token=token), json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    return {"sent": bool(body.get("ok")), "response": body}


def notify_if_configured(text: str) -> dict[str, Any]:
    if os.getenv("WEATHER_TMAX_ENABLE_TELEGRAM", "1").strip().lower() in {"0", "false", "no"}:
        return {"sent": False, "reason": "telegram_disabled"}
    return send_telegram_message(text)


def format_operational_cycle_message(summary: dict) -> str:
    acceptance = summary.get("forecast_acceptance", {})
    quality = summary.get("forecast_quality", {})
    forecast = summary.get("forecast", {})
    refresh = summary.get("refresh_summary") or {}
    freshness_passed = (refresh.get("freshness_gate") or {}).get("passed")
    status = "ACCEPTED" if summary.get("accepted") else "REJECTED"
    lines = [
        f"Weather Tmax Bot: {status}",
        f"Airport: {summary.get('airport')}",
        f"Target date: {summary.get('target_date_local')}",
        f"Issue UTC: {summary.get('issue_time_utc')}",
        f"Model: {summary.get('model_version')}",
        f"Forecast ID: {summary.get('forecast_id')}",
        *_format_temperature_summary(forecast),
        *_format_probability_bins(forecast.get("probabilities_by_integer_c", {})),
        *_format_thresholds(forecast.get("threshold_probabilities", {})),
        f"Quality: {quality.get('status')}",
        f"Freshness gate: {freshness_passed}",
        *_format_freshness(refresh),
        f"Acceptance blocking: {', '.join(acceptance.get('blocking_reasons', [])) or 'none'}",
        f"Cautions: {', '.join(acceptance.get('cautions', [])) or 'none'}",
        f"Recommendation: {summary.get('recommendation')}",
    ]
    return "\n".join(lines)


def _format_temperature_summary(forecast: dict) -> list[str]:
    if not forecast:
        return ["Forecast values: unavailable"]
    interval = forecast.get("intervals", {}).get("80", [])
    interval_text = "unavailable"
    if len(interval) == 2:
        interval_text = f"{float(interval[0]):.1f}C to {float(interval[1]):.1f}C"
    return [
        "",
        "Forecast:",
        f"Expected Tmax: {float(forecast['expected_tmax_c']):.1f}C",
        f"Median Tmax: {float(forecast['median_tmax_c']):.1f}C",
        f"Most likely bin: {int(forecast['most_likely_integer_c'])}C",
        f"80% interval: {interval_text}",
    ]


def _format_probability_bins(probabilities: dict) -> list[str]:
    if not probabilities:
        return []
    rows = sorted((int(bin_c), float(probability)) for bin_c, probability in probabilities.items())
    material = [(bin_c, probability) for bin_c, probability in rows if probability >= 0.01]
    if not material:
        material = sorted(rows, key=lambda row: row[1], reverse=True)[:5]
        material.sort()
    bins = ", ".join(f"{bin_c}C {probability:.1%}" for bin_c, probability in material)
    return [f"Bins >= 1%: {bins}"]


def _format_thresholds(thresholds: dict) -> list[str]:
    if not thresholds:
        return []
    return [
        "Thresholds: "
        f">=20C {float(thresholds.get('ge_20', 0.0)):.1%}, "
        f">=25C {float(thresholds.get('ge_25', 0.0)):.1%}, "
        f">=30C {float(thresholds.get('ge_30', 0.0)):.1%}, "
        f"<=0C {float(thresholds.get('le_0', 0.0)):.1%}",
    ]


def _format_freshness(refresh: dict) -> list[str]:
    statuses = ((refresh.get("freshness_gate") or {}).get("freshness") or {}).get("statuses", {})
    if not statuses:
        return []
    parts = []
    for source in ("metar", "taf", "nwp"):
        status = statuses.get(source, {})
        age = status.get("age_hours")
        age_text = "unknown" if age is None else f"{float(age):.1f}h"
        parts.append(f"{source.upper()} {status.get('state', 'unknown')} ({age_text})")
    return ["Freshness: " + ", ".join(parts)]


def format_outcome_update_message(result: dict) -> str:
    status = result.get("status", {})
    refresh = result.get("refresh_summary") or {}
    lines = [
        "Weather Tmax Bot: outcome update",
        f"Pending rows: {status.get('pending_rows')}",
        f"Ready rows: {status.get('ready_rows')}",
        f"Dates to refresh: {', '.join(status.get('dates_to_refresh', [])) or 'none'}",
        f"Ran refresh: {result.get('ran_refresh')}",
        f"Reports updated: {result.get('reports_updated')}",
        f"Monitoring rows: {refresh.get('forecast_monitoring_rows')}",
        f"Recommendation: {result.get('recommendation')}",
    ]
    return "\n".join(lines)


def format_healthcheck_message(readiness: dict) -> str:
    status = "READY" if readiness.get("ready_for_forward_ops") else "BLOCKED"
    lines = [
        f"Weather Tmax Bot healthcheck: {status}",
        f"Forward ops: {readiness.get('ready_for_forward_ops')}",
        f"Outcome monitoring: {readiness.get('ready_for_outcome_monitoring')}",
        f"Blocking: {', '.join(readiness.get('blocking_reasons', [])) or 'none'}",
        f"Accepted forecasts: {readiness.get('accepted_operational_forecasts')}",
        f"Pending truth rows: {readiness.get('pending_truth_rows')}",
        f"Next action: {readiness.get('next_action')}",
    ]
    return "\n".join(lines)
