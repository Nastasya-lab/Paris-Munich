from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.metar_upside_dataset import build_metar_remaining_upside_dataset


RAIN_COLUMNS = [
    "rain_6min_missing",
    "rain_mm_last_30m",
    "rain_mm_last_1h",
    "rain_mm_last_3h",
    "rain_mm_since_midnight",
    "rain_max_6min_last_3h",
]


def main() -> None:
    metar = pd.read_parquet("data/interim/metar_iem_LFPB.parquet")
    target = pd.read_parquet("data/processed/metar_tmax_target_LFPB.parquet")
    existing = pd.read_parquet("data/processed/metar_upside_dataset_LFPB.parquet")
    enhanced = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
    )
    enhanced = _copy_existing_rain_features(enhanced, existing)
    output = Path("data/processed/metar_upside_dataset_LFPB_intraday_enhanced.parquet")
    write_parquet(enhanced, output)
    new_columns = sorted(set(enhanced.columns) - set(existing.columns))
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "rows": len(enhanced),
        "days": int(enhanced["target_date_local"].nunique()),
        "output": str(output),
        "new_columns": new_columns,
        "rain_features_copied_from_existing_dataset": True,
        "leakage_check_pass_rate": float(enhanced["leakage_check_passed"].mean()),
    }
    Path("data/reports/lfpb_intraday_enhanced_dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


def _copy_existing_rain_features(enhanced: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    key = ["target_date_local", "issue_time_utc", "local_issue_hour"]
    if not set(key + RAIN_COLUMNS).issubset(existing.columns):
        return enhanced
    out = enhanced.drop(columns=[column for column in RAIN_COLUMNS if column in enhanced.columns], errors="ignore")
    rain = existing[key + RAIN_COLUMNS].copy()
    return out.merge(rain, on=key, how="left")


if __name__ == "__main__":
    main()
