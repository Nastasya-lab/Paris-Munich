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
        f"Quality: {quality.get('status')}",
        f"Freshness gate: {freshness_passed}",
        f"Acceptance blocking: {', '.join(acceptance.get('blocking_reasons', [])) or 'none'}",
        f"Cautions: {', '.join(acceptance.get('cautions', [])) or 'none'}",
        f"Recommendation: {summary.get('recommendation')}",
    ]
    return "\n".join(lines)


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
