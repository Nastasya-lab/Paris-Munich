from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.dwd_observations import DWDAdapter
from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.operational_monitoring import build_operational_monitoring_tables
from weather_tmax_bot.evaluation.outcome_analysis import build_outcome_analysis
from weather_tmax_bot.evaluation.outcomes import build_forecast_outcome_status, update_forecast_outcomes
from weather_tmax_bot.evaluation.promotion_gate import write_safe_blend_promotion_gate_report, write_shadow_promotion_gate_report
from weather_tmax_bot.features.build_target import build_daily_tmax


def plan_pending_truth_refresh(
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    target_path: str | Path = "data/processed/daily_target.parquet",
    as_of_date: date | None = None,
    min_lag_days: int = 1,
) -> dict:
    as_of = as_of_date or date.today()
    status = build_forecast_outcome_status(forecast_log_path=forecast_log_path, target_path=target_path, output_path=None)
    if status.empty:
        return {
            "as_of_date": as_of.isoformat(),
            "min_lag_days": min_lag_days,
            "dates_to_refresh": [],
            "pending_rows": 0,
            "ready_rows": 0,
            "outcome_status_counts": {},
        }
    status_counts = {str(key): int(value) for key, value in status["outcome_status"].value_counts(dropna=False).items()}
    pending = status[status["outcome_status"] == "pending_truth"].copy()
    if pending.empty:
        return {
            "as_of_date": as_of.isoformat(),
            "min_lag_days": min_lag_days,
            "dates_to_refresh": [],
            "pending_rows": 0,
            "ready_rows": 0,
            "outcome_status_counts": status_counts,
        }
    cutoff = as_of - timedelta(days=min_lag_days)
    pending["target_date"] = pd.to_datetime(pending["target_date_local"]).dt.date
    ready = pending[pending["target_date"] <= cutoff]
    dates = sorted({value.isoformat() for value in ready["target_date"]})
    return {
        "as_of_date": as_of.isoformat(),
        "min_lag_days": min_lag_days,
        "cutoff_date": cutoff.isoformat(),
        "dates_to_refresh": dates,
        "pending_rows": len(pending),
        "ready_rows": len(ready),
        "outcome_status_counts": status_counts,
    }


def refresh_pending_truth(
    airport: str = "EDDM",
    station_id: str = "01262",
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    observation_path: str | Path = "data/interim/dwd_10min_temperature_01262.parquet",
    target_path: str | Path = "data/processed/daily_target.parquet",
    monitoring_path: str | Path = "data/reports/forecast_monitoring.parquet",
    outcome_status_path: str | Path = "data/reports/forecast_outcome_status.parquet",
    reports_dir: str | Path = "data/reports",
    fetch: bool = False,
    as_of_date: date | None = None,
    min_lag_days: int = 1,
    adapter_factory=DWDAdapter,
) -> dict:
    plan = plan_pending_truth_refresh(
        forecast_log_path=forecast_log_path,
        target_path=target_path,
        as_of_date=as_of_date,
        min_lag_days=min_lag_days,
    )
    summary = {"airport": airport, "station_id": station_id, "plan": plan, "fetched_rows": 0, "target_rows": None}
    dates = [date.fromisoformat(value) for value in plan["dates_to_refresh"]]
    if not fetch or not dates:
        return summary

    start = min(dates)
    end = max(dates)
    adapter = adapter_factory()
    fetched = adapter.fetch_observations(airport=airport, start=start, end=end, station_id=station_id)
    merged = _merge_observations(observation_path, fetched)
    if merged.empty:
        summary.update(
            {
                "fetched_rows": len(fetched),
                "merged_observation_rows": 0,
                "target_rows": _rows(target_path),
                "refresh_status": "no_observations_available",
            }
        )
        return summary
    write_parquet(merged, observation_path)
    target = build_daily_tmax(merged, airport_icao=airport)
    write_parquet(target, target_path)
    variant_monitoring_path = Path(reports_dir) / "forecast_variant_monitoring.parquet"
    monitoring = update_forecast_outcomes(
        forecast_log_path=forecast_log_path,
        target_path=target_path,
        output_path=monitoring_path,
        variant_output_path=variant_monitoring_path,
    )
    status = build_forecast_outcome_status(
        forecast_log_path=forecast_log_path,
        target_path=target_path,
        output_path=outcome_status_path,
    )
    build_operational_monitoring_tables(
        monitoring_path=monitoring_path,
        forecast_log_path=forecast_log_path,
        outcome_status_path=outcome_status_path,
        output_dir=reports_dir,
    )
    outcome_analysis = build_outcome_analysis(
        monitoring_path=monitoring_path,
        output_json_path=Path(reports_dir) / "outcome_analysis.json",
        output_markdown_path="docs/outcome_analysis.md",
    )
    promotion_gate = write_shadow_promotion_gate_report(
        variant_monitoring_path=variant_monitoring_path,
        json_path=Path(reports_dir) / "shadow_promotion_gate.json",
        markdown_path="docs/shadow_promotion_gate.md",
    )
    safe_blend_gate = write_safe_blend_promotion_gate_report(
        variant_monitoring_path=variant_monitoring_path,
        json_path=Path(reports_dir) / "safe_blend_promotion_gate.json",
        markdown_path="docs/safe_blend_promotion_gate.md",
    )
    summary.update(
        {
            "fetched_rows": len(fetched),
            "merged_observation_rows": len(merged),
            "target_rows": len(target),
            "forecast_monitoring_rows": len(monitoring),
            "forecast_outcome_status_rows": len(status),
            "outcome_analysis_status": outcome_analysis["status"],
            "outcome_analysis_rows": outcome_analysis["rows"],
            "shadow_promotion_gate_status": promotion_gate["status"],
            "safe_blend_promotion_gate_status": safe_blend_gate["status"],
        }
    )
    return summary


def _rows(path: str | Path) -> int:
    p = Path(path)
    return 0 if not p.exists() else len(pd.read_parquet(p))


def _merge_observations(observation_path: str | Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    path = Path(observation_path)
    frames = []
    if path.exists():
        frames.append(pd.read_parquet(path))
    if not new_rows.empty:
        frames.append(new_rows)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged["observation_time_utc"] = pd.to_datetime(merged["observation_time_utc"], utc=True)
    subset = [col for col in ("station_id", "observation_time_utc") if col in merged.columns]
    if subset:
        merged = merged.drop_duplicates(subset=subset, keep="last")
    return merged.sort_values("observation_time_utc").reset_index(drop=True)
