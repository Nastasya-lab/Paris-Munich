from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.data.open_meteo import fetch_open_meteo_single_run_extract


def main() -> None:
    args = _parse_args()
    airport = _load_airport(args.airport)
    output = Path(args.output)
    selected = _selected_run_targets(
        dataset_path=Path(args.dataset),
        baseline_nwp_path=Path(args.baseline_nwp),
    )
    selected = _drop_existing(selected, output)
    selected = _order_selection(selected, args.order, args.max_runs)
    if args.max_runs is not None and args.order != "spread":
        selected = selected.head(args.max_runs)
    if selected.empty:
        print(json.dumps({"status": "complete", "remaining_runs": 0, "output": str(output)}, indent=2))
        return

    tasks = []
    for _, row in selected.iterrows():
        target_dates = [date.fromisoformat(value) for value in str(row["target_dates_local"]).split(",")]
        tasks.append(
            {
                "airport_icao": args.airport,
                "latitude": float(airport["latitude"]),
                "longitude": float(airport["longitude"]),
                "run_time_utc": pd.Timestamp(row["model_run_time_utc"]).to_pydatetime(),
                "target_dates_local": target_dates,
                "timezone_name": str(airport["timezone"]),
                "model_name": args.model,
                "forecast_days": args.forecast_days,
                "availability_latency_hours": args.availability_latency_hours,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    frames = []
    rows_fetched = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(
                _fetch_with_retries,
                task,
                attempts=args.retry_attempts,
                sleep_seconds=args.retry_sleep_seconds,
            ): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                frame = future.result()
                if not frame.empty:
                    frames.append(frame)
                    rows_fetched += len(frame)
            except Exception as exc:  # noqa: BLE001 - keep backfill resilient.
                failures.append({"model_run_time_utc": task["run_time_utc"].isoformat(), "error": str(exc)})
            if frames and (len(frames) >= args.batch_size or index == len(tasks)):
                NWPArchive(str(output)).append_extract(pd.concat(frames, ignore_index=True))
                frames = []
            if index % args.progress_every == 0 or index == len(tasks):
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/{len(tasks)}",
                            "rows_fetched": rows_fetched,
                            "failures": len(failures),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )

    summary = {
        "airport": args.airport,
        "model": args.model,
        "selected_runs_requested": len(tasks),
        "rows_fetched": rows_fetched,
        "failures": failures[:20],
        "failure_count": len(failures),
        "output": str(output),
    }
    print(json.dumps(summary, indent=2, default=str))


def _selected_run_targets(dataset_path: Path, baseline_nwp_path: Path) -> pd.DataFrame:
    dataset = pd.read_parquet(dataset_path)
    nwp = pd.read_parquet(baseline_nwp_path)
    dataset["target_date_local"] = dataset["target_date_local"].astype(str)
    dataset["issue_time_utc"] = pd.to_datetime(dataset["issue_time_utc"], utc=True)
    nwp["target_date_local"] = nwp["target_date_local"].astype(str)
    nwp["knowledge_time_utc"] = pd.to_datetime(nwp["knowledge_time_utc"], utc=True)
    nwp["model_run_time_utc"] = pd.to_datetime(nwp["model_run_time_utc"], utc=True)
    nwp = nwp[nwp["model_tmax_c"].notna()].sort_values("knowledge_time_utc")

    rows = []
    for _, row in dataset.iterrows():
        candidates = nwp[
            (nwp["target_date_local"] == row["target_date_local"])
            & (nwp["knowledge_time_utc"] <= row["issue_time_utc"])
        ]
        if candidates.empty:
            continue
        latest = candidates.iloc[-1]
        rows.append(
            {
                "model_run_time_utc": latest["model_run_time_utc"],
                "target_date_local": row["target_date_local"],
            }
        )
    selected = pd.DataFrame(rows).drop_duplicates()
    grouped = (
        selected.groupby("model_run_time_utc", as_index=False)["target_date_local"]
        .agg(lambda values: ",".join(sorted(set(values))))
        .rename(columns={"target_date_local": "target_dates_local"})
        .sort_values("model_run_time_utc")
        .reset_index(drop=True)
    )
    return grouped


def _drop_existing(selected: pd.DataFrame, output: Path) -> pd.DataFrame:
    if not output.exists():
        return selected
    existing = pd.read_parquet(output)
    if existing.empty or not {"model_run_time_utc", "target_date_local"}.issubset(existing.columns):
        return selected
    existing["model_run_time_utc"] = pd.to_datetime(existing["model_run_time_utc"], utc=True)
    existing["target_date_local"] = existing["target_date_local"].astype(str)
    existing_pairs = set(zip(existing["model_run_time_utc"].astype(str), existing["target_date_local"]))
    keep = []
    for _, row in selected.iterrows():
        run = str(pd.Timestamp(row["model_run_time_utc"]))
        targets = str(row["target_dates_local"]).split(",")
        keep.append(any((run, target) not in existing_pairs for target in targets))
    return selected[keep].reset_index(drop=True)


def _order_selection(selected: pd.DataFrame, order: str, max_runs: int | None) -> pd.DataFrame:
    if selected.empty:
        return selected
    selected = selected.sort_values("model_run_time_utc").reset_index(drop=True)
    if order == "reverse":
        return selected.iloc[::-1].reset_index(drop=True)
    if order == "spread" and max_runs is not None and len(selected) > max_runs:
        positions = sorted(set(np.linspace(0, len(selected) - 1, max_runs).round().astype(int).tolist()))
        return selected.iloc[positions].reset_index(drop=True)
    return selected


def _fetch_with_retries(task: dict, *, attempts: int, sleep_seconds: float) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return fetch_open_meteo_single_run_extract(**task)
        except Exception as exc:  # noqa: BLE001 - retry transient API/rate-limit failures.
            last_error = exc
            message = str(exc).lower()
            if attempt >= attempts:
                break
            if "429" in message or "timeout" in message or "temporarily" in message:
                time.sleep(sleep_seconds * attempt)
                continue
            break
    raise last_error or RuntimeError("fetch failed")


def _load_airport(icao: str) -> dict:
    payload = yaml.safe_load(Path("config/airports.yaml").read_text(encoding="utf-8"))
    return payload["airports"][icao]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill only ICON-D2 runs selected by LFPB as-of dataset rows.")
    parser.add_argument("--airport", default="LFPB")
    parser.add_argument("--model", default="icon_d2")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB.parquet")
    parser.add_argument("--baseline-nwp", default="data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    parser.add_argument("--output", default="data/forecasts/open_meteo_single_runs_icon_d2_LFPB_enhanced.parquet")
    parser.add_argument("--forecast-days", type=int, default=3)
    parser.add_argument("--availability-latency-hours", type=float, default=3.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--order", choices=["chronological", "reverse", "spread"], default="chronological")
    parser.add_argument("--retry-attempts", type=int, default=4)
    parser.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
