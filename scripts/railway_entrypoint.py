from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    job = resolve_job(os.getenv("RAILWAY_SERVICE_NAME"), os.getenv("WEATHER_TMAX_JOB"))
    _run([sys.executable, "scripts/railway_bootstrap.py"])
    if job == "api":
        _run([sys.executable, "scripts/10_start_api.py"])
        return
    _run(build_api_job_command(job))


def resolve_job(service_name: str | None, explicit_job: str | None = None) -> str:
    if explicit_job:
        return explicit_job.strip().lower()
    normalized = (service_name or "").strip().lower()
    if "metar" in normalized and "cron" in normalized:
        if "lfpb" in normalized or "paris" in normalized:
            return "lfpb-metar-event"
        return "metar-event"
    if "forecast" in normalized and "cron" in normalized:
        if "lfpb" in normalized or "paris" in normalized:
            return "lfpb-forecast"
        return "forecast"
    if "lfpb" in normalized and "metar" in normalized:
        return "lfpb-metar-event"
    if "lfpb" in normalized and "forecast" in normalized:
        return "lfpb-forecast"
    if "outcome" in normalized and "cron" in normalized:
        return "outcome"
    if "daily" in normalized and "report" in normalized and "cron" in normalized:
        return "daily-report"
    if "health" in normalized and "cron" in normalized:
        return "health"
    return "api"


def build_api_job_command(job: str) -> list[str]:
    if job == "forecast-all":
        return [sys.executable, "scripts/55_multi_airport_job.py", "forecast-all"]
    if job == "metar-event-all-once":
        return [sys.executable, "scripts/55_multi_airport_job.py", "metar-event-all-once"]
    if job == "lfpb-forecast":
        return [sys.executable, "scripts/53_lfpb_forecast_job.py"]
    if job == "lfpb-metar-event":
        return [
            sys.executable,
            "scripts/54_lfpb_metar_event_job.py",
            "--poll-timeout-seconds",
            os.getenv("METAR_POLL_TIMEOUT_SECONDS", "600"),
            "--poll-interval-seconds",
            os.getenv("METAR_POLL_INTERVAL_SECONDS", "30"),
        ]
    command = [sys.executable, "scripts/33_call_api_job.py", job]
    if job == "metar-event":
        command.extend(
            [
                "--poll-timeout-seconds",
                os.getenv("METAR_POLL_TIMEOUT_SECONDS", "600"),
                "--poll-interval-seconds",
                os.getenv("METAR_POLL_INTERVAL_SECONDS", "30"),
            ]
        )
    return command


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
