from __future__ import annotations

from datetime import datetime, time, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.features.nwp_features import build_nwp_features


ISSUE_HOURS_UTC = [0, 3, 6, 9, 12, 15, 18]


def build_nwp_feature_table(
    nwp: pd.DataFrame,
    target_dates: list[str],
    issue_hours_utc: list[int] | None = None,
) -> pd.DataFrame:
    issue_hours_utc = issue_hours_utc or ISSUE_HOURS_UTC
    if nwp.empty:
        return pd.DataFrame()
    rows = []
    prepared = nwp.copy()
    prepared["target_date_local"] = prepared["target_date_local"].astype(str)
    prepared["model_availability_time_utc"] = pd.to_datetime(prepared["model_availability_time_utc"], utc=True)
    for target_date_str in target_dates:
        target_date = pd.to_datetime(target_date_str).date()
        day = prepared[prepared["target_date_local"] == target_date.isoformat()].copy()
        for hour in issue_hours_utc:
            issue = datetime.combine(target_date, time(hour=hour), tzinfo=timezone.utc)
            row = {
                "target_date_local": target_date.isoformat(),
                "issue_time_utc": issue.isoformat(),
                "issue_hour_utc": hour,
            }
            row.update(build_nwp_features(day, issue))
            if not day.empty:
                available = day[day["model_availability_time_utc"] <= pd.Timestamp(issue)]
                if not available.empty:
                    row["latest_nwp_model_name"] = available.iloc[-1].get("model_name")
                    row["latest_nwp_source_id"] = available.iloc[-1].get("source_id")
                    row["max_nwp_knowledge_time_utc"] = available.iloc[-1].get("knowledge_time_utc")
            rows.append(row)
    return pd.DataFrame(rows)


def build_nwp_feature_table_from_files(
    nwp_path: str | Path = "data/forecasts/open_meteo_archive.parquet",
    target_path: str | Path = "data/processed/daily_target.parquet",
) -> pd.DataFrame:
    nwp = pd.read_parquet(nwp_path)
    target_dates = sorted(nwp["target_date_local"].astype(str).unique().tolist())
    if Path(target_path).exists():
        target = pd.read_parquet(target_path)
        target_dates = sorted(set(target_dates).union(set(target["target_date_local"].astype(str).tolist())))
    return build_nwp_feature_table(nwp, target_dates)
