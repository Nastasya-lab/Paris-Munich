from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from weather_tmax_bot.data.provider import WeatherDataProvider
from weather_tmax_bot.temporal.knowledge_view import KnowledgeView
from weather_tmax_bot.utils.validation import DataAvailabilityError


class MostlyRightAdapter(WeatherDataProvider):
    """Optional wrapper. The project never depends on this adapter for core safety."""

    def __init__(self):
        try:
            import mostlyright  # type: ignore
        except Exception as exc:
            raise DataAvailabilityError("MostlyRight SDK is not installed") from exc
        self.sdk = mostlyright
        self._records = pd.DataFrame(columns=["knowledge_time_utc", "source_id"])

    def fetch_observations(self, airport: str, start: date, end: date) -> pd.DataFrame:
        raise DataAvailabilityError("MostlyRight observations are not the DWD truth source for EDDM")

    def fetch_metar(self, airport: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise DataAvailabilityError("Use MostlyRight METAR only after station/source coverage is verified for this SDK version")

    def fetch_taf(self, airport: str, start: datetime, end: datetime) -> pd.DataFrame:
        raise DataAvailabilityError("MostlyRight TAF support is not assumed by this MVP")

    def fetch_nwp(self, airport: str, issue_time: datetime, target_date: date) -> pd.DataFrame:
        raise DataAvailabilityError("MostlyRight NWP/Open-Meteo rows must be imported with issued-run metadata preserved")

    def get_source_metadata(self, source_id: str) -> dict:
        return {"source_id": source_id, "provider": "mostlyright.sdk", "optional": True}

    def get_knowledge_view(self, as_of: datetime) -> pd.DataFrame:
        return KnowledgeView(self._records).as_of(as_of)
