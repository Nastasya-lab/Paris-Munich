from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


def main() -> None:
    args = _parse_args()
    _activate_lfpb_telegram()
    command = [
        sys.executable,
        "scripts/48_predict_lfpb_metar_tmax.py",
        "--airport",
        "LFPB",
        "--target-date",
        args.target_date or datetime.now(ZoneInfo("Europe/Paris")).date().isoformat(),
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


def _activate_lfpb_telegram() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_LFPB")
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    if chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = chat_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LFPB ICON-D2 METAR Tmax forecast and notify Telegram.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
