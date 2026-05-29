from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.temporal.knowledge_view import KnowledgeView


def test_as_of_filters_future_knowledge():
    df = pd.DataFrame(
        {
            "knowledge_time_utc": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "source_id": ["a", "a"],
            "value": [1, 2],
        }
    )
    got = KnowledgeView(df).as_of(datetime(2026, 1, 1, 12, tzinfo=timezone.utc))
    assert got["value"].tolist() == [1]
