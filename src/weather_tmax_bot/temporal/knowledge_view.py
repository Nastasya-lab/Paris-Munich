from __future__ import annotations

from datetime import datetime
from typing import Iterable

import pandas as pd

from weather_tmax_bot.utils.time import ensure_utc


REQUIRED_TEMPORAL_COLUMNS = {"knowledge_time_utc", "source_id"}


class KnowledgeView:
    def __init__(self, records: pd.DataFrame):
        missing = REQUIRED_TEMPORAL_COLUMNS - set(records.columns)
        if missing:
            raise ValueError(f"knowledge records missing required columns: {sorted(missing)}")
        self.records = records.copy()
        self.records["knowledge_time_utc"] = pd.to_datetime(self.records["knowledge_time_utc"], utc=True)

    def as_of(self, as_of: datetime, sources: Iterable[str] | None = None) -> pd.DataFrame:
        as_of_utc = ensure_utc(as_of)
        df = self.records[self.records["knowledge_time_utc"] <= pd.Timestamp(as_of_utc)].copy()
        if sources is not None:
            df = df[df["source_id"].isin(list(sources))]
        return df.reset_index(drop=True)


def get_knowledge_view(records: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    return KnowledgeView(records).as_of(as_of)
