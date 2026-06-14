from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from weather_tmax_bot.utils.time import ensure_utc
from weather_tmax_bot.utils.validation import LeakageError


class LeakageDetector:
    def audit_feature_frame(
        self,
        features: pd.DataFrame,
        issue_time_utc: datetime,
        target_date_local: date | None = None,
        target_columns: set[str] | None = None,
    ) -> dict[str, object]:
        issue_time_utc = ensure_utc(issue_time_utc)
        target_columns = target_columns or {"tmax_c", "target", "target_tmax_c"}

        present_targets = target_columns.intersection(features.columns)
        if present_targets:
            raise LeakageError(f"target column leaked into features: {sorted(present_targets)}")

        temporal_columns = {
            "knowledge_time_utc",
            "observation_time_utc",
            "taf_issue_time_utc",
            "nwp_availability_time_utc",
        }
        temporal_columns.update(
            col
            for col in features.columns
            if col.endswith("knowledge_time_utc") or col.endswith("availability_time_utc")
        )
        for col in sorted(temporal_columns):
            if col in features.columns and not features.empty:
                values = pd.to_datetime(features[col], utc=True, errors="coerce").dropna()
                if not values.empty and values.max().to_pydatetime() > issue_time_utc:
                    raise LeakageError(f"{col} after issue_time used as feature: {values.max()} > {issue_time_utc}")

        if "target_date_local" in features.columns and target_date_local is not None:
            bad = features["target_date_local"].astype(str) != target_date_local.isoformat()
            if bad.any():
                raise LeakageError("feature rows contain a different local target day")

        max_knowledge = None
        knowledge_columns = [col for col in features.columns if col.endswith("knowledge_time_utc")]
        for col in knowledge_columns:
            values = pd.to_datetime(features[col], utc=True, errors="coerce").dropna()
            if values.empty:
                continue
            value = values.max().to_pydatetime()
            max_knowledge = value if max_knowledge is None else max(max_knowledge, value)

        return {
            "passed": True,
            "issue_time_utc": issue_time_utc.isoformat(),
            "max_feature_knowledge_time_utc": None if max_knowledge is None else max_knowledge.isoformat(),
        }
