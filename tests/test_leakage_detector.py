from datetime import date, datetime, timezone

import pandas as pd
import pytest

from weather_tmax_bot.temporal.leakage_detector import LeakageDetector
from weather_tmax_bot.utils.validation import LeakageError


def test_future_knowledge_fails():
    features = pd.DataFrame({"knowledge_time_utc": ["2026-01-02T00:00:00Z"], "x": [1]})
    with pytest.raises(LeakageError):
        LeakageDetector().audit_feature_frame(features, datetime(2026, 1, 1, tzinfo=timezone.utc), date(2026, 1, 1))


def test_target_column_fails():
    features = pd.DataFrame({"x": [1], "tmax_c": [20]})
    with pytest.raises(LeakageError):
        LeakageDetector().audit_feature_frame(features, datetime(2026, 1, 1, tzinfo=timezone.utc))
