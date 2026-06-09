from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def main() -> None:
    args = _parse_args()
    if args.job == "forecast-all":
        results = [
            _run_step(
                "EDDM forecast",
                [
                    sys.executable,
                    "scripts/33_call_api_job.py",
                    "forecast",
                    "--airport",
                    "EDDM",
                    "--issue-time",
                    args.issue_time,
                ],
            ),
            _run_step(
                "LFPB forecast",
                [
                    sys.executable,
                    "scripts/53_lfpb_forecast_job.py",
                    "--issue-time",
                    args.issue_time,
                ],
            ),
        ]
    elif args.job == "metar-event-all-once":
        results = [
            _run_step(
                "EDDM METAR once",
                [
                    sys.executable,
                    "scripts/33_call_api_job.py",
                    "metar-event",
                    "--airport",
                    "EDDM",
                    "--issue-time",
                    args.issue_time,
                    "--poll-timeout-seconds",
                    "0",
                    "--poll-interval-seconds",
                    "30",
                ],
            ),
            _run_step(
                "LFPB METAR once",
                [
                    sys.executable,
                    "scripts/54_lfpb_metar_event_job.py",
                    "--issue-time",
                    args.issue_time,
                    "--poll-timeout-seconds",
                    "0",
                ],
            ),
        ]
    else:
        raise ValueError(f"Unsupported multi-airport job: {args.job}")
    report = {
        "job": args.job,
        "created_at_local": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "results": results,
        "ok": all(result["returncode"] == 0 for result in results),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ok"] and args.fail_on_any_error:
        raise SystemExit(1)


def _run_step(label: str, command: list[str]) -> dict:
    env = os.environ.copy()
    started = datetime.now(ZoneInfo("UTC")).isoformat()
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env={**env, "WEATHER_TMAX_MULTI_AIRPORT_CHILD": "1"},
        )
        if completed.stdout:
            print(f"\n===== {label} stdout =====\n{completed.stdout}")
        if completed.stderr:
            print(f"\n===== {label} stderr =====\n{completed.stderr}", file=sys.stderr)
        return {
            "label": label,
            "command": " ".join(command),
            "started_at_utc": started,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-1200:],
            "stderr_tail": completed.stderr[-1200:],
        }
    except Exception as exc:
        return {
            "label": label,
            "command": " ".join(command),
            "started_at_utc": started,
            "returncode": 999,
            "error": repr(exc),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-airport Railway job.")
    parser.add_argument("job", choices=["forecast-all", "metar-event-all-once"])
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--fail-on-any-error", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
