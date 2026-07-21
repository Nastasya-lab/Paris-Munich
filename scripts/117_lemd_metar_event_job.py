from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from weather_tmax_bot.operations.refresh import refresh_awc_live


STATE_PATH = Path("data/logs/lemd_metar_event_state.json")
METAR_PATH = Path("data/forecasts/awc_metar_live_LEMD.parquet")


def main() -> None:
    args = _parse_args()
    refresh = refresh_awc_live("LEMD")
    latest = _latest_metar_time()
    previous = _load_state().get("latest_metar_time_utc")
    if latest is not None and latest != previous:
        command = [
            sys.executable,
            "scripts/116_lemd_forecast_job.py",
            "--issue-time",
            args.issue_time,
            "--update-trigger",
            "new_metar",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
        _save_state(latest)
        print(json.dumps({"status": "new_metar_forecast", "airport": "LEMD", "latest_metar_time_utc": latest, "refresh": refresh}, indent=2))
        return
    print(json.dumps({"status": "no_new_metar", "airport": "LEMD", "latest_metar_time_utc": latest, "previous_metar_time_utc": previous, "refresh": refresh}, indent=2))


def _latest_metar_time() -> str | None:
    if not METAR_PATH.exists():
        return None
    frame = pd.read_parquet(METAR_PATH)
    latest = pd.to_datetime(frame.get("observation_time_utc"), utc=True, errors="coerce").max()
    return None if pd.isna(latest) else latest.isoformat()


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(latest: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps({"latest_metar_time_utc": latest, "updated_at_utc": datetime.now(UTC).isoformat()}, indent=2),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll LEMD METAR and forecast on a new report.")
    parser.add_argument("--issue-time", default="now")
    return parser.parse_args()


if __name__ == "__main__":
    main()
