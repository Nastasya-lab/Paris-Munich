from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def test_freshness_gate_fails_on_stale_source(tmp_path):
    forecasts = tmp_path / "data" / "forecasts"
    forecasts.mkdir(parents=True)
    pd.DataFrame({"observation_time_utc": ["2026-07-15T08:00:00Z"]}).to_parquet(
        forecasts / "awc_metar_live_EDDM.parquet",
        index=False,
    )
    pd.DataFrame({"issue_time_utc": ["2026-07-15T11:00:00Z"]}).to_parquet(
        forecasts / "awc_taf_live_EDDM.parquet",
        index=False,
    )
    pd.DataFrame({"knowledge_time_utc": ["2026-07-15T10:00:00Z"]}).to_parquet(
        forecasts / "open_meteo_archive.parquet",
        index=False,
    )

    result = evaluate_freshness_gate(tmp_path, datetime(2026, 7, 15, 12, tzinfo=timezone.utc))

    assert not result["passed"]
    assert any(item["source"] == "metar" for item in result["failures"])


def test_freshness_gate_can_ignore_missing_sources(tmp_path):
    result = evaluate_freshness_gate(
        tmp_path,
        datetime(2026, 7, 15, 12, tzinfo=timezone.utc),
        fail_on_missing=False,
    )

    assert result["passed"]
