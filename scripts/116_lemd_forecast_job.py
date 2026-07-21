from __future__ import annotations

import argparse
import os
import subprocess
import sys


DEFAULT_CHAT_ID = "-1004409683948"


def main() -> None:
    args = _parse_args()
    _activate_telegram()
    command = [
        sys.executable,
        "scripts/115_predict_lemd_metar_tmax.py",
        "--issue-time",
        args.issue_time,
        "--update-trigger",
        args.update_trigger,
        "--notify",
    ]
    if args.target_date:
        command.extend(["--target-date", args.target_date])
    subprocess.run(command, check=True)


def _activate_telegram() -> None:
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN_LEMD")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
    )
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID_LEMD") or DEFAULT_CHAT_ID


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one LEMD forecast and notify Madrid Telegram.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--update-trigger", choices=["scheduled_forecast", "new_metar"], default="scheduled_forecast")
    return parser.parse_args()


if __name__ == "__main__":
    main()
