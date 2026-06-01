import json
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from weather_tmax_bot.operations.truth_refresh import _merge_observations, plan_pending_truth_refresh, refresh_pending_truth


def test_plan_pending_truth_refresh_identifies_completed_pending_dates(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    target_path = tmp_path / "daily_target.parquet"
    log_path.write_text(
        json.dumps(
            {
                "forecast_id": "f1",
                "airport": "EDDM",
                "issue_time_utc": "2026-07-15T06:00:00Z",
                "target_date_local": "2026-07-15",
                "model_version": "m1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-14"],
            "tmax_c": [25.0],
            "quality_flags": ["ok"],
        }
    ).to_parquet(target_path, index=False)

    plan = plan_pending_truth_refresh(log_path, target_path, as_of_date=pd.to_datetime("2026-07-17").date())

    assert plan["dates_to_refresh"] == ["2026-07-15"]
    assert plan["ready_rows"] == 1
    assert plan["outcome_status_counts"]["pending_truth"] == 1


def test_refresh_pending_truth_without_fetch_only_returns_plan(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    target_path = tmp_path / "daily_target.parquet"
    log_path.write_text("", encoding="utf-8")
    pd.DataFrame(columns=["airport_icao", "target_date_local", "tmax_c", "quality_flags"]).to_parquet(
        target_path,
        index=False,
    )

    summary = refresh_pending_truth(forecast_log_path=log_path, target_path=target_path, fetch=False)

    assert summary["fetched_rows"] == 0
    assert summary["plan"]["dates_to_refresh"] == []


def test_merge_observations_deduplicates_station_time(tmp_path):
    existing = tmp_path / "obs.parquet"
    pd.DataFrame(
        {
            "station_id": ["01262"],
            "observation_time_utc": [datetime(2026, 1, 1, 0, tzinfo=timezone.utc)],
            "temperature_c": [1.0],
        }
    ).to_parquet(existing, index=False)
    new_rows = pd.DataFrame(
        {
            "station_id": ["01262"],
            "observation_time_utc": [datetime(2026, 1, 1, 0, tzinfo=timezone.utc)],
            "temperature_c": [2.0],
        }
    )

    merged = _merge_observations(existing, new_rows)

    assert len(merged) == 1
    assert merged.iloc[0]["temperature_c"] == 2.0


def test_refresh_pending_truth_fetch_merges_target_and_outcomes(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    obs_path = tmp_path / "obs.parquet"
    target_path = tmp_path / "daily_target.parquet"
    monitoring_path = tmp_path / "forecast_monitoring.parquet"
    status_path = tmp_path / "forecast_outcome_status.parquet"
    reports_dir = tmp_path / "reports"
    log_path.write_text(
        json.dumps(
            {
                "forecast_id": "f1",
                "airport": "EDDM",
                "issue_time_utc": "2026-07-15T06:00:00+00:00",
                "target_date_local": "2026-07-15",
                "model_version": "m1",
                "probability_distribution": {"24": 0.4, "25": 0.6},
                "expected_tmax_c": 24.6,
                "median_tmax_c": 25.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(columns=["airport_icao", "target_date_local", "tmax_c", "quality_flags"]).to_parquet(
        target_path,
        index=False,
    )

    class FakeDWDAdapter:
        def fetch_observations(self, airport, start, end, station_id):
            times = [datetime(2026, 7, 15, 0, tzinfo=timezone.utc) + timedelta(minutes=10 * i) for i in range(144)]
            temps = [20.0] * 144
            temps[72] = 25.0
            return pd.DataFrame(
                {
                    "station_id": [station_id] * 144,
                    "observation_time_utc": times,
                    "temperature_c": temps,
                    "source_id": ["dwd.10min.air_temperature.01262"] * 144,
                    "source_version": ["test"] * 144,
                }
            )

    summary = refresh_pending_truth(
        forecast_log_path=log_path,
        observation_path=obs_path,
        target_path=target_path,
        monitoring_path=monitoring_path,
        outcome_status_path=status_path,
        reports_dir=reports_dir,
        fetch=True,
        as_of_date=date(2026, 7, 17),
        adapter_factory=FakeDWDAdapter,
    )

    assert summary["fetched_rows"] == 144
    assert summary["forecast_monitoring_rows"] == 1
    assert pd.read_parquet(monitoring_path).iloc[0]["actual_tmax_c"] == 25.0
    assert (reports_dir / "operational_by_model.parquet").exists()


def test_refresh_pending_truth_fetch_handles_empty_observation_response(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    obs_path = tmp_path / "obs.parquet"
    target_path = tmp_path / "daily_target.parquet"
    log_path.write_text(
        json.dumps(
            {
                "forecast_id": "f1",
                "airport": "EDDM",
                "issue_time_utc": "2026-07-15T06:00:00+00:00",
                "target_date_local": "2026-07-15",
                "model_version": "m1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    existing_target = pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-14"],
            "tmax_c": [23.0],
            "quality_flags": ["ok"],
        }
    )
    existing_target.to_parquet(target_path, index=False)

    class EmptyDWDAdapter:
        def fetch_observations(self, airport, start, end, station_id):
            return pd.DataFrame()

    summary = refresh_pending_truth(
        forecast_log_path=log_path,
        observation_path=obs_path,
        target_path=target_path,
        fetch=True,
        as_of_date=date(2026, 7, 17),
        adapter_factory=EmptyDWDAdapter,
    )

    assert summary["refresh_status"] == "no_observations_available"
    assert summary["target_rows"] == 1
    assert pd.read_parquet(target_path).equals(existing_target)
