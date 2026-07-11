from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


REPORT_PATH = Path("data/reports/latest_eham_icon_d2_metar_tmax_prediction.json")


def main() -> None:
    args = _parse_args()
    _activate_eham_telegram()
    command = [
        sys.executable,
        "scripts/105_predict_eham_metar_tmax.py",
        "--airport",
        "EHAM",
        "--target-date",
        args.target_date or datetime.now(ZoneInfo("Europe/Amsterdam")).date().isoformat(),
        "--issue-time",
        args.issue_time,
        "--auto-refresh",
        "--refresh-nwp",
        "--notify",
        "--model-path",
        "data/models/eham_metar_tmax_icon_d2_v1.joblib",
        "--metadata-path",
        "data/models/eham_metar_tmax_icon_d2_v1.metadata.json",
        "--promote-spatial-candidate",
        "--no-hf-icon-eu-shadow",
        "--report-path",
        str(REPORT_PATH),
    ]
    if not args.log:
        command.append("--no-log")
    subprocess.run(command, check=True)
    _run_polymarket_paper()


def _run_polymarket_paper() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/83_lfpb_polymarket_paper_job.py",
            "--airport",
            "EHAM",
            "--notify",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(f"\n===== EHAM Polymarket paper stdout =====\n{completed.stdout}")
    if completed.stderr:
        print(f"\n===== EHAM Polymarket paper stderr =====\n{completed.stderr}", file=sys.stderr)
    if completed.returncode != 0:
        print(
            "EHAM Polymarket paper job failed; the weather forecast remains successful.",
            file=sys.stderr,
        )


def _activate_eham_telegram() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN_EHAM") or os.getenv("TELEGRAM_BOT_TOKEN_LFPB") or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_EHAM") or "-1004216691526"
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    os.environ["TELEGRAM_CHAT_ID"] = chat_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one EHAM ICON-D2 METAR Tmax forecast and notify Telegram.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
