from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, time as dt_time, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.data.open_meteo import fetch_open_meteo_single_run_extract


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Open-Meteo issued Single Runs NWP extracts.")
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--start-date", required=True, help="First local target date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="Last local target date, YYYY-MM-DD")
    parser.add_argument("--model", default="icon_d2")
    parser.add_argument("--run-hours", default="0,3,6,9,12,15,18,21")
    parser.add_argument("--forecast-days", type=int, default=3)
    parser.add_argument("--availability-latency-hours", type=float, default=3.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--run-offset", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="Continue after the latest model_run_time_utc already present in the output file.")
    parser.add_argument("--retry-attempts", type=int, default=1)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    airport = _load_airport(args.airport)
    output = args.output or f"data/forecasts/open_meteo_single_runs_{args.model}_{args.airport}.parquet"

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    run_hours = [int(part.strip()) for part in args.run_hours.split(",") if part.strip()]
    target_dates = list(_date_range(start, end))
    all_run_times = list(_run_times(start - timedelta(days=args.forecast_days - 1), end, run_hours))
    run_offset = _resume_offset(output, all_run_times) if args.resume else args.run_offset
    run_times = all_run_times
    if run_offset:
        run_times = run_times[run_offset:]
    if args.max_runs is not None:
        run_times = run_times[: args.max_runs]

    failures = []
    buffered = []
    rows_fetched = 0
    target_dates_with_rows: set[str] = set()
    for index, run_time in enumerate(run_times, start=1):
        try:
            frame = _fetch_with_retries(
                attempts=args.retry_attempts,
                retry_sleep_seconds=args.retry_sleep_seconds,
                airport_icao=args.airport,
                latitude=float(airport["latitude"]),
                longitude=float(airport["longitude"]),
                run_time_utc=run_time,
                target_dates_local=target_dates,
                timezone_name=str(airport["timezone"]),
                model_name=args.model,
                forecast_days=args.forecast_days,
                availability_latency_hours=args.availability_latency_hours,
            )
            if not frame.empty:
                buffered.append(frame)
                rows_fetched += len(frame)
                target_dates_with_rows.update(frame["target_date_local"].astype(str).unique().tolist())
        except Exception as exc:  # noqa: BLE001 - continue and report skipped runs.
            failures.append({"run_time_utc": run_time.isoformat(), "error": str(exc)})
        if buffered and not args.dry_run and (len(buffered) >= args.batch_size or index == len(run_times)):
            NWPArchive(output).append_extract(pd.concat(buffered, ignore_index=True))
            buffered = []
        if index % max(args.batch_size, 1) == 0 or index == len(run_times):
            print(
                json.dumps(
                    {
                        "progress": f"{index}/{len(run_times)}",
                        "rows_fetched": rows_fetched,
                        "failures": len(failures),
                        "latest_run_time_utc": run_time.isoformat(),
                    },
                    default=str,
                ),
                flush=True,
            )
        time.sleep(args.sleep_seconds)

    summary = {
        "airport": args.airport,
        "model": args.model,
        "source_id": f"open_meteo.single_run.{args.model}",
        "target_start_date": start.isoformat(),
        "target_end_date": end.isoformat(),
        "run_offset": run_offset,
        "run_times_requested": len(run_times),
        "rows_fetched": rows_fetched,
        "target_dates_with_rows": sorted(target_dates_with_rows),
        "failures": failures[:10],
        "failure_count": len(failures),
        "dry_run": args.dry_run,
        "output": output,
    }
    print(json.dumps(summary, indent=2, default=str))


def _date_range(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _run_times(start: date, end: date, run_hours: list[int]):
    for day in _date_range(start, end):
        for hour in run_hours:
            yield datetime.combine(day, dt_time(hour=hour), tzinfo=timezone.utc)


def _fetch_with_retries(attempts: int, retry_sleep_seconds: float, **kwargs) -> pd.DataFrame:
    attempts = max(1, int(attempts))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fetch_open_meteo_single_run_extract(**kwargs)
        except Exception as exc:  # noqa: BLE001 - retain original behavior after retries.
            last_error = exc
            if attempt < attempts:
                time.sleep(retry_sleep_seconds)
    raise last_error or RuntimeError("Open-Meteo fetch failed")


def _resume_offset(output: str, run_times: list[datetime]) -> int:
    path = Path(output)
    if not path.exists():
        return 0
    frame = pd.read_parquet(path)
    if frame.empty or "model_run_time_utc" not in frame.columns:
        return 0
    latest = pd.to_datetime(frame["model_run_time_utc"], utc=True, errors="coerce").max()
    if pd.isna(latest):
        return 0
    latest_dt = latest.to_pydatetime()
    for index, run_time in enumerate(run_times):
        if run_time == latest_dt:
            return index + 1
    return 0


def _load_airport(icao: str) -> dict:
    config_path = Path("config/airports.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    airports = payload.get("airports", {})
    if icao not in airports:
        raise SystemExit(f"Airport {icao} is not configured in {config_path}")
    airport = airports[icao]
    required = {"latitude", "longitude", "timezone"}
    missing = sorted(required.difference(airport))
    if missing:
        raise SystemExit(f"Airport {icao} is missing fields: {missing}")
    return airport


if __name__ == "__main__":
    main()
