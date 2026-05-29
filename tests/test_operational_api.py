import pandas as pd

from weather_tmax_bot.evaluation.operational_api import operational_monitoring_payload


def test_operational_monitoring_payload_reads_tables(tmp_path):
    reports = tmp_path / "data" / "reports"
    reports.mkdir(parents=True)
    pd.DataFrame({"model_version": ["m1"], "forecasts": [2]}).to_parquet(
        reports / "operational_by_model.parquet",
        index=False,
    )
    pd.DataFrame({"model_version": ["m1"], "forecast_accepted": ["accepted"], "forecasts": [1]}).to_parquet(
        reports / "operational_acceptance.parquet",
        index=False,
    )

    payload = operational_monitoring_payload(tmp_path)

    assert payload["by_model"][0]["model_version"] == "m1"
    assert payload["acceptance"][0]["forecast_accepted"] == "accepted"
    assert payload["source_mismatch"] == []
