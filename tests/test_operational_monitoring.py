import pandas as pd

from weather_tmax_bot.evaluation.operational_monitoring import (
    build_operational_monitoring_tables,
    forecast_log_inventory,
    pending_forecast_summary,
    summarize_acceptance,
    summarize_availability,
    summarize_by_model,
    summarize_source_mismatch,
)


def _monitoring_frame():
    return pd.DataFrame(
        {
            "forecast_id": ["f1", "f2"],
            "model_version": ["m1", "m1"],
            "error_expected_c": [1.0, -3.0],
            "nll": [2.0, 3.0],
            "crps": [0.5, 1.5],
            "brier_ge_20": [0.1, 0.2],
            "brier_ge_25": [0.2, 0.3],
            "brier_ge_30": [0.3, 0.4],
            "metar_source_mismatch": [True, False],
            "taf_source_mismatch": [False, False],
            "metar_missing": [False, False],
            "taf_missing": [False, True],
            "nwp_missing": [True, True],
            "forecast_accepted": [True, False],
        }
    )


def test_operational_summaries():
    df = _monitoring_frame()
    by_model = summarize_by_model(df)
    mismatch = summarize_source_mismatch(df)
    availability = summarize_availability(df)
    acceptance = summarize_acceptance(df)

    assert by_model.iloc[0]["forecasts"] == 2
    assert by_model.iloc[0]["mae_expected"] == 2.0
    assert set(mismatch["any_source_mismatch"]) == {False, True}
    assert availability[availability["source"] == "nwp"].iloc[0]["missing_rate"] == 1.0
    assert set(acceptance["forecast_accepted"]) == {"accepted", "rejected"}


def test_build_operational_monitoring_tables_writes_outputs(tmp_path):
    path = tmp_path / "forecast_monitoring.parquet"
    _monitoring_frame().to_parquet(path, index=False)

    tables = build_operational_monitoring_tables(monitoring_path=path, output_dir=tmp_path)

    assert len(tables["by_model"]) == 1
    assert len(tables["acceptance"]) == 2
    assert (tmp_path / "operational_by_model.parquet").exists()
    assert (tmp_path / "operational_acceptance.parquet").exists()


def test_forecast_log_inventory(tmp_path):
    log = tmp_path / "forecast_log.jsonl"
    log.write_text(
        '{"forecast_id":"f1","model_version":"m1","airport":"EDDM","target_date_local":"2026-07-15",'
        '"issue_time_utc":"2026-07-15T06:00:00Z","raw_input_metadata":'
        '{"latest_metar_source_id":"awc.metar.live.EDDM","nwp_missing":true,'
        '"forecast_quality":{"status":"degraded"},'
        '"forecast_acceptance":{"accepted":false,"blocking_reasons":["quality_status_ok"]}}}\n',
        encoding="utf-8",
    )

    inventory = forecast_log_inventory(log)

    assert inventory.iloc[0]["logged_forecasts"] == 1
    assert inventory.iloc[0]["metar_sources"] == "awc.metar.live.EDDM"
    assert inventory.iloc[0]["nwp_missing_rate"] == 1.0
    assert inventory.iloc[0]["rejected_rate"] == 1.0
    assert inventory.iloc[0]["quality_statuses"] == "degraded"
    assert inventory.iloc[0]["acceptance_blocking_reasons"] == "quality_status_ok"


def test_pending_forecast_summary(tmp_path):
    path = tmp_path / "forecast_outcome_status.parquet"
    pd.DataFrame(
        {
            "forecast_id": ["f1", "f2"],
            "model_version": ["m1", "m1"],
            "outcome_status": ["pending_truth", "scored"],
            "target_date_local": ["2026-07-15", "2026-07-14"],
            "forecast_accepted": [False, True],
            "forecast_quality_status": ["degraded", "ok"],
            "forecast_acceptance_blocking_reasons": ["quality_status_ok", None],
        }
    ).to_parquet(path, index=False)

    summary = pending_forecast_summary(path)

    assert set(summary["outcome_status"]) == {"pending_truth", "scored"}
    assert set(summary["forecast_accepted"]) == {"accepted", "rejected"}


def test_pending_forecast_summary_fills_missing_quality(tmp_path):
    path = tmp_path / "forecast_outcome_status.parquet"
    pd.DataFrame(
        {
            "forecast_id": ["f1"],
            "model_version": ["m1"],
            "outcome_status": ["pending_truth"],
            "target_date_local": ["2026-07-15"],
            "forecast_accepted": [None],
            "forecast_quality_status": [None],
        }
    ).to_parquet(path, index=False)

    summary = pending_forecast_summary(path)

    assert summary.iloc[0]["forecast_accepted"] == "unknown"
    assert summary.iloc[0]["forecast_quality_status"] == "unknown"
