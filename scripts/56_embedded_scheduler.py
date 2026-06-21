from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


FORECAST_UTC_HOURS = {1, 4, 7, 10, 13, 16, 19, 22}


@dataclass(frozen=True)
class SchedulerJob:
    name: str
    due_key: str
    command: list[str]


def main() -> None:
    poll_seconds = int(os.getenv("EMBEDDED_SCHEDULER_POLL_SECONDS", "60"))
    state_path = Path(os.getenv("EMBEDDED_SCHEDULER_STATE_PATH", "data/logs/embedded_scheduler_state.json"))
    startup_delay = int(os.getenv("EMBEDDED_SCHEDULER_STARTUP_DELAY_SECONDS", "20"))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    time.sleep(max(0, startup_delay))

    while True:
        now = datetime.now(timezone.utc)
        state = _load_state(state_path)
        for job in due_jobs(now):
            if state.get(job.name) == job.due_key:
                continue
            result = _run_job(job)
            state[job.name] = job.due_key
            state[f"{job.name}_last_finished_utc"] = datetime.now(timezone.utc).isoformat()
            state[f"{job.name}_last_returncode"] = result.returncode
            if result.returncode != 0:
                state[f"{job.name}_last_error"] = (result.stderr or result.stdout or "").strip()[-2000:]
            else:
                state.pop(f"{job.name}_last_error", None)
            _save_state(state_path, state)
        time.sleep(max(10, poll_seconds))


def due_jobs(now: datetime) -> list[SchedulerJob]:
    minute_key = now.strftime("%Y-%m-%dT%H:%M")
    jobs = [
        SchedulerJob(
            name="metar_event_all_once",
            due_key=minute_key,
            command=[sys.executable, "scripts/55_multi_airport_job.py", "metar-event-all-once"],
        )
    ]
    if now.minute == 30 and now.hour in FORECAST_UTC_HOURS:
        jobs.append(
            SchedulerJob(
                name="forecast_all",
                due_key=minute_key,
                command=[sys.executable, "scripts/55_multi_airport_job.py", "forecast-all"],
            )
        )
    if now.hour == 6 and now.minute == 30:
        jobs.append(
            SchedulerJob(
                name="outcome",
                due_key=now.strftime("%Y-%m-%d"),
                command=[sys.executable, "scripts/33_call_api_job.py", "outcome"],
            )
        )
    if now.hour == 18 and now.minute == 15:
        jobs.append(
            SchedulerJob(
                name="daily_report",
                due_key=now.strftime("%Y-%m-%d"),
                command=[sys.executable, "scripts/33_call_api_job.py", "daily-report"],
            )
        )
    if now.hour == 18 and now.minute == 20:
        jobs.append(
            SchedulerJob(
                name="live_baseline",
                due_key=now.strftime("%Y-%m-%d"),
                command=[sys.executable, "scripts/73_build_live_baseline_report.py"],
            )
        )
    return jobs


def _run_job(job: SchedulerJob) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if not env.get("MUNICH_API_BASE_URL"):
        port = env.get("PORT", "8000")
        env["MUNICH_API_BASE_URL"] = f"http://127.0.0.1:{port}"
    return subprocess.run(job.command, env=env, text=True, capture_output=True, check=False)


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_state(path: Path, state: dict[str, object]) -> None:
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    main()
