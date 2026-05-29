from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import typer


def main(
    job: str = typer.Argument(..., help="forecast or outcome"),
    base_url: str | None = typer.Option(None),
    airport: str = typer.Option("EDDM"),
    target_date: str | None = typer.Option(None),
    issue_time: str = typer.Option("now"),
    timeout: int = typer.Option(120),
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
                "update_reports": True,
                "notify": True,
            },
            headers=headers,
            timeout=timeout,
        )
    elif job == "outcome":
        response = requests.post(
            f"{base}/pending-truth-cron",
            params={"fetch": True, "update_reports": True, "notify": True},
            headers=headers,
            timeout=timeout,
        )
    else:
        raise typer.BadParameter("job must be forecast or outcome")
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2, default=str))


if __name__ == "__main__":
    typer.run(main)
