from __future__ import annotations

import argparse
import os
import subprocess
import sys


DEFAULT_CHAT_ID = "-1004469237763"


def main() -> None:
    args = _parse_args()
    _activate_telegram()
    command = [
        sys.executable,
        "scripts/123_predict_rcss_metar_tmax.py",
        "--issue-time",
        args.issue_time,
        "--update-trigger",
        args.update_trigger,
        "--notify",
    ]
    if args.target_date:
        command.extend(["--target-date", args.target_date])
    subprocess.run(command, check=True)
    _run_polymarket_paper()


def _run_polymarket_paper() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/83_lfpb_polymarket_paper_job.py",
            "--airport",
            "RCSS",
            "--notify",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(f"\n===== RCSS Polymarket paper stdout =====\n{completed.stdout}")
    if completed.stderr:
        print(f"\n===== RCSS Polymarket paper stderr =====\n{completed.stderr}", file=sys.stderr)
    if completed.returncode != 0:
        print(
            "RCSS Polymarket paper job failed; the weather forecast remains successful.",
            file=sys.stderr,
        )


def _activate_telegram() -> None:
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN_RCSS")
        or os.getenv("TELEGRAM_BOT_TOKEN")
        or os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
    )
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID_RCSS") or DEFAULT_CHAT_ID


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RCSS forecast and notify Taipei Telegram.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--update-trigger", choices=["scheduled_forecast", "new_metar"], default="scheduled_forecast")
    return parser.parse_args()


if __name__ == "__main__":
    main()
