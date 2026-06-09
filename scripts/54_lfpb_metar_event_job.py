from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from weather_tmax_bot.operations.refresh import refresh_awc_live


STATE_PATH = Path("data/logs/lfpb_metar_event_state.json")
METAR_PATH = Path("data/forecasts/awc_metar_live_LFPB.parquet")


def main() -> None:
    args = _parse_args()
    _activate_lfpb_telegram()
    refresh = refresh_awc_live("LFPB")
    latest = _latest_metar_time()
    previous = _load_state().get("latest_metar_time_utc")
    is_new = latest is not None and latest != previous
    if is_new:
        _run_forecast(args)
        _save_state(latest)
        print(json.dumps({"status": "new_metar_forecast", "latest_metar_time_utc": latest, "refresh": refresh}, indent=2))
        return
    print(
        json.dumps(
            {
                "status": "no_new_metar",
                "latest_metar_time_utc": latest,
                "previous_metar_time_utc": previous,
                "refresh": refresh,
            },
            indent=2,
        )
    )


def _run_forecast(args: argparse.Namespace) -> None:
    target_date = args.target_date or datetime.now(ZoneInfo("Europe/Paris")).date().isoformat()
    command = [
        sys.executable,
        "scripts/48_predict_lfpb_metar_tmax.py",
        "--airport",
        "LFPB",
        "--target-date",
        target_date,
        "--issue-time",
        args.issue_time,
        "--auto-refresh",
        "--refresh-nwp",
        "--notify",
        "--model-path",
        "data/models/lfpb_metar_tmax_icon_d2_v1.joblib",
        "--metadata-path",
        "data/models/lfpb_metar_tmax_icon_d2_v1.metadata.json",
        "--report-path",
        "data/reports/latest_lfpb_icon_d2_metar_tmax_prediction.json",
    ]
    if not args.log:
        command.append("--no-log")
    subprocess.run(command, check=True)


def _latest_metar_time() -> str | None:
    if not METAR_PATH.exists():
        return None
    frame = pd.read_parquet(METAR_PATH)
    if frame.empty or "observation_time_utc" not in frame.columns:
        return None
    latest = pd.to_datetime(frame["observation_time_utc"], utc=True, errors="coerce").max()
    if pd.isna(latest):
        return None
    return latest.isoformat()


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _save_state(latest_metar_time_utc: str) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(
            {
                "latest_metar_time_utc": latest_metar_time_utc,
                "updated_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _activate_lfpb_telegram() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_LFPB")
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    if chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = chat_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll for a new LFPB METAR and notify forecast on each new report.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--poll-timeout-seconds", type=int, default=0, help="Ignored; kept for backward compatibility.")
    parser.add_argument("--poll-interval-seconds", type=int, default=30, help="Ignored; kept for backward compatibility.")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
