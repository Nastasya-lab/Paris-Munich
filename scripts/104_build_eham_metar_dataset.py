from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.iem import IEMAdapter
from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.build_metar_target import build_daily_metar_tmax
from weather_tmax_bot.features.metar_upside_dataset import build_metar_remaining_upside_dataset


AIRPORT = "EHAM"
TIMEZONE = "Europe/Amsterdam"
DEFAULT_NEIGHBORS = ["EHRD", "EHLE", "EHKD"]


def main() -> None:
    args = _parse_args()
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    adapter = IEMAdapter()
    metar = _fetch_or_read_metar(adapter, AIRPORT, start, end, Path(args.metar_path), fetch=args.fetch)
    for station in args.neighbor_station:
        _fetch_or_read_metar(
            adapter,
            station,
            start,
            end,
            Path(args.neighbor_dir) / f"metar_iem_{station}.parquet",
            fetch=args.fetch_neighbors,
        )

    target = build_daily_metar_tmax(
        metar,
        airport_icao=AIRPORT,
        timezone_name=TIMEZONE,
        source_id="iem.metar.archive.EHAM",
        expected_reports_per_day=48,
    )
    write_parquet(target, args.target_path)

    dataset = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao=AIRPORT,
        timezone_name=TIMEZONE,
        local_issue_hours=args.local_issue_hour,
    )
    write_parquet(dataset, args.output_path)

    report = {
        "airport": AIRPORT,
        "timezone": TIMEZONE,
        "period": [start.isoformat(), end.isoformat()],
        "metar_rows": int(len(metar)),
        "target_days": int(target["target_date_local"].nunique()) if not target.empty else 0,
        "target_ok_days": int(target["quality_flags"].eq("ok").sum()) if not target.empty else 0,
        "dataset_rows": int(len(dataset)),
        "dataset_days": int(dataset["target_date_local"].nunique()) if not dataset.empty else 0,
        "issue_hours": sorted(float(v) for v in dataset["local_issue_hour"].dropna().unique()) if not dataset.empty else [],
        "neighbor_stations": list(args.neighbor_station),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (report_dir / "eham_metar_upside_dataset_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _fetch_or_read_metar(
    adapter: IEMAdapter,
    station: str,
    start: datetime,
    end: datetime,
    path: Path,
    *,
    fetch: bool,
) -> pd.DataFrame:
    if path.exists() and not fetch:
        return pd.read_parquet(path)
    frame = adapter.fetch_metar(station, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(frame, path)
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EHAM METAR Tmax target and remaining-upside dataset.")
    parser.add_argument("--start", default="2025-07-27T00:00:00Z")
    parser.add_argument("--end", default="2026-05-30T23:59:00Z")
    parser.add_argument("--metar-path", default="data/interim/metar_iem_EHAM.parquet")
    parser.add_argument("--target-path", default="data/processed/metar_tmax_target_EHAM.parquet")
    parser.add_argument("--output-path", default="data/processed/metar_upside_dataset_EHAM_intraday_enhanced.parquet")
    parser.add_argument("--neighbor-dir", default="data/interim")
    parser.add_argument("--neighbor-station", nargs="*", default=DEFAULT_NEIGHBORS)
    parser.add_argument("--local-issue-hour", nargs="*", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--fetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-neighbors", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
