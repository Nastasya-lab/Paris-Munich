from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.temporal.freshness import assess_archive_freshness, assess_feature_freshness


def test_assess_feature_freshness_marks_fresh_and_stale():
    issue = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)
    result = assess_feature_freshness(
        {
            "latest_metar_time_utc": "2026-07-15T11:30:00Z",
            "latest_taf_issue_time_utc": "2026-07-14T18:00:00Z",
            "max_nwp_knowledge_time_utc": "2026-07-15T02:00:00Z",
            "metar_missing": False,
            "taf_missing": False,
            "nwp_missing": False,
        },
        issue,
    )
    assert result["statuses"]["metar"]["state"] == "fresh"
    assert result["statuses"]["taf"]["state"] == "stale"
    assert result["statuses"]["nwp"]["state"] == "fresh"
    assert any("TAF data is stale" in warning for warning in result["warnings"])


def test_assess_feature_freshness_marks_missing_and_future_timestamp():
    issue = datetime(2026, 7, 15, 12, tzinfo=timezone.utc)
    result = assess_feature_freshness(
        {
            "latest_metar_time_utc": None,
            "latest_taf_issue_time_utc": "2026-07-15T13:00:00Z",
            "max_nwp_knowledge_time_utc": None,
            "metar_missing": True,
            "taf_missing": False,
            "nwp_missing": True,
        },
        issue,
    )
    assert result["statuses"]["metar"]["state"] == "missing"
    assert result["statuses"]["taf"]["state"] == "future_timestamp"
    assert result["statuses"]["nwp"]["state"] == "missing"


def test_assess_archive_freshness_reads_latest_files(tmp_path):
    forecasts = tmp_path / "data" / "forecasts"
    forecasts.mkdir(parents=True)
    pd.DataFrame({"observation_time_utc": ["2026-07-15T11:30:00Z"]}).to_parquet(
        forecasts / "awc_metar_live_EDDM.parquet",
        index=False,
    )
    result = assess_archive_freshness(tmp_path, datetime(2026, 7, 15, 12, tzinfo=timezone.utc))
    assert result["statuses"]["metar"]["state"] == "fresh"
    assert result["statuses"]["taf"]["state"] == "missing"
