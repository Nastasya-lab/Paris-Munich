from __future__ import annotations

import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import typer


def main(
    job: str = typer.Argument(..., help="forecast, metar-event, outcome, or health"),
    base_url: str | None = typer.Option(None),
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    issue_time: str = typer.Option("now"),
    timeout: int = typer.Option(120),
    poll_timeout_seconds: int = typer.Option(0, help="For metar-event, keep polling until a new METAR appears."),
    poll_interval_seconds: int = typer.Option(30, help="For metar-event polling, seconds between checks."),
):
    base = (base_url or os.getenv("MUNICH_API_BASE_URL") or "").rstrip("/")
    if not base:
        raise typer.BadParameter("Set --base-url or MUNICH_API_BASE_URL")
    api_key = os.getenv("OPERATIONAL_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    if job == "forecast":
        target = target_date or datetime.now(ZoneInfo("Europe/Berlin")).date().isoformat()
        response = requests.post(
            f"{base}/operational-cycle",
            params={
                "airport": airport,
                "target_date": target,
                "issue_time": issue_time,
                "auto_refresh": True,
                "refresh_awc": True,
                "refresh_nwp": True,
                "log": True,
                "update_reports": False,
                "notify": True,
            },
            headers=headers,
            timeout=timeout,
        )
    elif job == "metar-event":
        target = target_date or datetime.now(ZoneInfo("Europe/Berlin")).date().isoformat()
        result = _run_metar_event_with_optional_polling(
            base=base,
            headers=headers,
            airport=airport,
            target=target,
            issue_time=issue_time,
            request_timeout=timeout,
            poll_timeout_seconds=poll_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
        print(json.dumps(result, indent=2, default=str))
        return
    elif job == "outcome":
        response = requests.post(
            f"{base}/pending-truth-cron",
            params={"fetch": True, "update_reports": True, "notify": True},
            headers=headers,
            timeout=timeout,
        )
    elif job == "health":
        response = requests.post(
            f"{base}/scheduler-healthcheck",
            params={"notify_on_success": True, "notify_on_failure": True},
            headers=headers,
            timeout=timeout,
        )
    else:
        raise typer.BadParameter("job must be forecast, metar-event, outcome, or health")
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, default=str))


def _run_metar_event_with_optional_polling(
    *,
    base: str,
    headers: dict[str, str],
    airport: str,
    target: str,
    issue_time: str,
    request_timeout: int,
    poll_timeout_seconds: int,
    poll_interval_seconds: int,
) -> dict:
    attempts = []
    deadline = time.monotonic() + max(0, poll_timeout_seconds)
    interval = max(5, poll_interval_seconds)
    while True:
        response = requests.post(
            f"{base}/metar-event-cycle",
            params={
                "airport": airport,
                "target_date": target,
                "issue_time": issue_time,
                "log": True,
                "notify": True,
            },
            headers=headers,
            timeout=request_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        attempts.append(
            {
                "status": payload.get("status"),
                "latest_metar_time_utc": payload.get("latest_metar_time_utc"),
                "notification_sent": payload.get("notification_sent"),
            }
        )
        if payload.get("status") == "new_metar_forecast" or time.monotonic() >= deadline:
            payload["polling"] = {
                "enabled": poll_timeout_seconds > 0,
                "attempts": attempts,
                "attempt_count": len(attempts),
                "poll_timeout_seconds": poll_timeout_seconds,
                "poll_interval_seconds": interval,
            }
            return payload
        time.sleep(interval)


if __name__ == "__main__":
    typer.run(main)
