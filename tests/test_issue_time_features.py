from datetime import date, datetime, timezone

from weather_tmax_bot.features.issue_time_features import build_issue_time_features
from weather_tmax_bot.operations.quality import assess_forecast_quality


def test_issue_time_features_for_scheduled_issue():
    features = build_issue_time_features(
        datetime(2026, 5, 29, 6, tzinfo=timezone.utc),
        date(2026, 5, 29),
    )

    assert features["issue_schedule_offset_minutes"] == 0
    assert features["issue_off_schedule"] is False


def test_issue_time_features_for_slightly_off_schedule_issue():
    features = build_issue_time_features(
        datetime(2026, 5, 29, 6, 10, tzinfo=timezone.utc),
        date(2026, 5, 29),
    )

    assert features["issue_schedule_offset_minutes"] == 10
    assert features["issue_off_schedule"] is True


def test_issue_time_quality_warns_for_far_off_schedule():
    quality = assess_forecast_quality(
        {
            "issue_schedule_offset_minutes": 150,
            "freshness": {"metar": {"state": "fresh"}},
        },
        [],
    )

    assert quality["status"] == "degraded"
    assert "issue time is outside configured training schedule" in quality["reasons"]
