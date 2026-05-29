from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.utils.hashing import stable_hash


NWP_COLUMNS = [
    "airport_icao",
    "target_date_local",
    "model_name",
    "model_run_time_utc",
    "model_availability_time_utc",
    "knowledge_time_utc",
    "forecast_horizon_hours",
    "model_tmax_c",
    "source_id",
    "raw_record_hash",
]


class NWPArchive:
    def __init__(self, path: str | Path = "data/forecasts/nwp_archive.parquet"):
        self.path = Path(path)

    def append_extract(self, rows: pd.DataFrame) -> Path:
        if self.path.exists():
            rows = pd.concat([pd.read_parquet(self.path), rows], ignore_index=True)
        if "raw_record_hash" in rows.columns:
            rows = rows.drop_duplicates(subset=["raw_record_hash"], keep="last")
        if "ingest_time_utc" in rows.columns:
            rows = rows.sort_values("ingest_time_utc")
        return write_parquet(rows, self.path)


def nwp_row(
    airport: str,
    target_date: date,
    model_name: str,
    model_run_time_utc: datetime,
    model_availability_time_utc: datetime,
    model_tmax_c: float | None,
    source_id: str,
) -> dict:
    row = {
        "airport_icao": airport,
        "target_date_local": target_date.isoformat(),
        "model_name": model_name,
        "model_run_time_utc": model_run_time_utc.astimezone(timezone.utc),
        "model_availability_time_utc": model_availability_time_utc.astimezone(timezone.utc),
        "knowledge_time_utc": model_availability_time_utc.astimezone(timezone.utc),
        "forecast_horizon_hours": None,
        "model_tmax_c": model_tmax_c,
        "source_id": source_id,
    }
    row["raw_record_hash"] = stable_hash(row)
    return row
