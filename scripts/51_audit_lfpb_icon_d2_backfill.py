from __future__ import annotations

import json
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    frame = pd.read_parquet(path)
    next_offset, next_run, remaining = _next_offset(frame)
    by_day = (
        frame.assign(target_date_local=frame["target_date_local"].astype(str))
        .groupby("target_date_local")
        .agg(
            rows=("target_date_local", "size"),
            nonnull_model_tmax=("model_tmax_c", lambda s: int(s.notna().sum())),
            unique_runs=("model_run_time_utc", "nunique"),
        )
        .reset_index()
    )
    report = {
        "archive_path": str(path),
        "rows": int(len(frame)),
        "unique_hashes": int(frame["raw_record_hash"].nunique()) if "raw_record_hash" in frame.columns else None,
        "unique_model_runs": int(frame["model_run_time_utc"].nunique()),
        "target_start": str(frame["target_date_local"].astype(str).min()),
        "target_end": str(frame["target_date_local"].astype(str).max()),
        "target_days": int(frame["target_date_local"].nunique()),
        "model_tmax_nonnull_rate": float(frame["model_tmax_c"].notna().mean()),
        "latest_model_run_time_utc": pd.Timestamp(frame["model_run_time_utc"].max()).isoformat(),
        "next_resume_offset": next_offset,
        "next_run_time_utc": None if next_run is None else next_run.isoformat(),
        "remaining_runs_after_next": remaining,
        "daily_tail": json.loads(by_day.tail(20).to_json(orient="records")),
    }
    Path("data/reports/lfpb_icon_d2_backfill_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    by_day.to_csv("data/reports/lfpb_icon_d2_backfill_by_day.csv", index=False)
    print(json.dumps(report, indent=2, default=str))


def _next_offset(frame: pd.DataFrame) -> tuple[int, datetime | None, int]:
    latest = pd.Timestamp(frame["model_run_time_utc"].max()).to_pydatetime()
    start = date.fromisoformat("2025-05-31") - timedelta(days=2)
    end = date.fromisoformat("2026-05-30")
    runs = []
    current = start
    while current <= end:
        for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
            runs.append(datetime.combine(current, time(hour), tzinfo=timezone.utc))
        current += timedelta(days=1)
    offset = runs.index(latest) + 1
    next_run = runs[offset] if offset < len(runs) else None
    return offset, next_run, len(runs) - offset


if __name__ == "__main__":
    main()
