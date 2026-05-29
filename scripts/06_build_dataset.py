from datetime import datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.climatology_features import day_of_year_sin_cos
from weather_tmax_bot.features.issue_time_features import ISSUE_HOURS_UTC, build_issue_time_features
from weather_tmax_bot.features.taf_features import build_taf_features
from weather_tmax_bot.temporal.leakage_detector import LeakageDetector


def main():
    target = pd.read_parquet("data/processed/daily_target.parquet")
    target = target[target["quality_flags"] == "ok"].copy()
    metar_features = _read_optional("data/processed/metar_features_iem_EDDM.parquet")
    metar_feature_map = {
        (row["target_date_local"], int(row["issue_hour_utc"])): row.to_dict()
        for _, row in metar_features.iterrows()
    } if not metar_features.empty else {}
    nwp_features = _read_optional("data/processed/nwp_features_open_meteo.parquet")
    # Forward NWP rows are not backfilled into historical training until outcomes exist.
    known_targets = set(target["target_date_local"].astype(str).tolist())
    nwp_features = nwp_features[nwp_features["target_date_local"].astype(str).isin(known_targets)] if not nwp_features.empty else nwp_features
    nwp_feature_map = {
        (row["target_date_local"], int(row["issue_hour_utc"])): row.to_dict()
        for _, row in nwp_features.iterrows()
    } if not nwp_features.empty else {}
    taf = _read_optional("data/interim/taf_iem_EDDM.parquet")
    rows = []
    detector = LeakageDetector()
    for _, row in target.iterrows():
        target_date = pd.to_datetime(row["target_date_local"]).date()
        for hour in ISSUE_HOURS_UTC:
            issue = datetime.combine(target_date, time(hour=hour), tzinfo=timezone.utc)
            feature_row = {
                "airport_icao": row["airport_icao"],
                "station_id": row["station_id"],
                "target_date_local": target_date.isoformat(),
                "issue_time_utc": issue,
                "issue_hour_utc": hour,
                "truth_source_id": row["source_id"],
                "obs_coverage_missing_ratio": row["missing_ratio"],
                "max_feature_knowledge_time_utc": pd.NaT,
                "metar_missing": True,
                "taf_missing": True,
                "nwp_missing": True,
            }
            feature_row.update(build_issue_time_features(issue, target_date, ISSUE_HOURS_UTC))
            feature_row.update(day_of_year_sin_cos(target_date))
            metar_feature_row = metar_feature_map.get((target_date.isoformat(), hour))
            if metar_feature_row is not None:
                feature_row.update(
                    {
                        key: value
                        for key, value in metar_feature_row.items()
                        if key not in {"target_date_local", "issue_time_utc", "issue_hour_utc"}
                    }
                )
            nwp_feature_row = nwp_feature_map.get((target_date.isoformat(), hour))
            if nwp_feature_row is not None:
                feature_row.update(
                    {
                        key: value
                        for key, value in nwp_feature_row.items()
                        if key not in {"target_date_local", "issue_time_utc", "issue_hour_utc"}
                    }
                )
            if not taf.empty:
                feature_row.update(build_taf_features(taf, issue, target_date))
                taf_times = pd.to_datetime(taf["issue_time_utc"], utc=True, errors="coerce")
                taf_slice = taf[taf_times <= pd.Timestamp(issue)]
                if not taf_slice.empty:
                    feature_row["latest_taf_issue_time_utc"] = taf_slice["issue_time_utc"].max()
            audit = detector.audit_feature_frame(
                pd.DataFrame([{k: v for k, v in feature_row.items() if k != "tmax_c"}]),
                issue_time_utc=issue,
                target_date_local=target_date,
            )
            feature_row["leakage_check_passed"] = audit["passed"]
            feature_row["tmax_c"] = row["tmax_c"]
            rows.append(feature_row)
    dataset = pd.DataFrame(rows)
    dataset = dataset.replace({np.nan: None})
    write_parquet(dataset, "data/processed/training_dataset.parquet")
    print(f"Wrote {len(dataset)} training rows")


def _read_optional(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


if __name__ == "__main__":
    main()
