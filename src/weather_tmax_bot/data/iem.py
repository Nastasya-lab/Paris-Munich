from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO

import pandas as pd
import requests

from weather_tmax_bot.data.metar import parse_metar
from weather_tmax_bot.data.provider import WeatherDataProvider
from weather_tmax_bot.data.source_registry import SourceRegistry
from weather_tmax_bot.data.taf import parse_taf
from weather_tmax_bot.temporal.availability import metar_knowledge_time, taf_knowledge_time
from weather_tmax_bot.temporal.knowledge_view import KnowledgeView
from weather_tmax_bot.utils.hashing import text_hash


IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_TAF_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/taf.py"


class IEMAdapter(WeatherDataProvider):
    def __init__(self):
        self.registry = SourceRegistry()
        self._records = pd.DataFrame(columns=["knowledge_time_utc", "source_id"])

    def fetch_observations(self, airport: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError("IEM is not the EDDM truth source")

    def fetch_metar(self, airport: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "station": airport,
            "data": [
                "tmpc",
                "dwpc",
                "relh",
                "drct",
                "sknt",
                "gust",
                "mslp",
                "alti",
                "vsby",
                "skyc1",
                "skyc2",
                "skyc3",
                "wxcodes",
                "metar",
            ],
            "year1": start.year,
            "month1": start.month,
            "day1": start.day,
            "year2": end.year,
            "month2": end.month,
            "day2": end.day,
            "tz": "UTC",
            "format": "onlycomma",
            "latlon": "no",
            "elev": "no",
            "missing": "M",
            "trace": "T",
            "direct": "no",
            "report_type": ["1", "2"],
        }
        response = requests.get(IEM_ASOS_URL, params=params, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), na_values=["M"])
        if df.empty:
            return _empty_metar_frame()
        df["observation_time_utc"] = pd.to_datetime(df["valid"], utc=True)
        df = df[(df["observation_time_utc"] >= pd.Timestamp(start)) & (df["observation_time_utc"] <= pd.Timestamp(end))]
        rows = []
        ingest = datetime.now(timezone.utc)
        for _, row in df.iterrows():
            raw = str(row.get("metar", ""))
            parsed = parse_metar(raw) if raw and raw != "nan" else {}
            obs_time = row["observation_time_utc"].to_pydatetime()
            rows.append(
                {
                    "station": row.get("station", airport),
                    "observation_time_utc": obs_time,
                    "knowledge_time_utc": metar_knowledge_time(obs_time),
                    "ingest_time_utc": ingest,
                    "source_id": f"iem.metar.archive.{airport}",
                    "source_version": "iem.asos.csv",
                    "raw_metar": raw,
                    "temperature_c": _coalesce(row.get("tmpc"), parsed.get("temperature_c")),
                    "dewpoint_c": _coalesce(row.get("dwpc"), parsed.get("dewpoint_c")),
                    "qnh_hpa": parsed.get("qnh_hpa"),
                    "wind_direction_deg": _coalesce(row.get("drct"), parsed.get("wind_direction_deg")),
                    "wind_speed_kt": _coalesce(row.get("sknt"), parsed.get("wind_speed_kt")),
                    "gust_kt": _coalesce(row.get("gust"), parsed.get("gust_kt")),
                    "visibility": row.get("vsby"),
                    "weather_codes": row.get("wxcodes"),
                    "cloud_layers": " ".join(str(row.get(c)) for c in ("skyc1", "skyc2", "skyc3") if pd.notna(row.get(c))),
                    "ceiling_ft": None,
                    "cavok": bool(parsed.get("cavok", False)),
                    "parser_quality_flag": "iem_csv_plus_basic_parser",
                    "raw_record_hash": text_hash(raw),
                }
            )
        out = pd.DataFrame(rows)
        self._records = pd.concat([self._records, out], ignore_index=True)
        return out

    def fetch_taf(self, airport: str, start: datetime, end: datetime) -> pd.DataFrame:
        params = {
            "station": airport,
            "year1": start.year,
            "month1": start.month,
            "day1": start.day,
            "hour1": start.hour,
            "minute1": start.minute,
            "year2": end.year,
            "month2": end.month,
            "day2": end.day,
            "hour2": end.hour,
            "minute2": end.minute,
            "tz": "UTC",
            "fmt": "comma",
        }
        response = requests.get(IEM_TAF_URL, params=params, timeout=120)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        if df.empty:
            return _empty_taf_frame()
        rows = []
        ingest = datetime.now(timezone.utc)
        for _, row in df.iterrows():
            raw = str(row.get("raw", ""))
            issue = pd.to_datetime(row.get("valid"), utc=True, errors="coerce")
            if pd.isna(issue):
                continue
            valid_from = pd.to_datetime(row.get("fx_valid"), utc=True, errors="coerce")
            valid_to = pd.to_datetime(row.get("fx_valid_end"), utc=True, errors="coerce")
            parsed = parse_taf(raw)
            issue_dt = issue.to_pydatetime()
            parsed.update(
                {
                    "station": row.get("station", airport),
                    "issue_time_utc": issue_dt,
                    "knowledge_time_utc": taf_knowledge_time(issue_dt),
                    "valid_from_utc": None if pd.isna(valid_from) else valid_from.to_pydatetime(),
                    "valid_to_utc": None if pd.isna(valid_to) else valid_to.to_pydatetime(),
                    "ingest_time_utc": ingest,
                    "source_id": f"iem.taf.archive.{airport}",
                    "source_version": "iem.taf.csv",
                    "raw_taf": raw,
                    "raw_record_hash": text_hash(raw),
                }
            )
            rows.append(parsed)
        out = pd.DataFrame(rows)
        self._records = pd.concat([self._records, out], ignore_index=True)
        return out

    def fetch_nwp(self, airport: str, issue_time: datetime, target_date: date) -> pd.DataFrame:
        raise NotImplementedError("IEM adapter does not provide NWP")

    def get_source_metadata(self, source_id: str) -> dict:
        return self.registry.get(source_id).metadata

    def get_knowledge_view(self, as_of: datetime) -> pd.DataFrame:
        return KnowledgeView(self._records).as_of(as_of)


def _coalesce(primary, fallback):
    return fallback if pd.isna(primary) else primary


def _empty_metar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "station",
            "observation_time_utc",
            "knowledge_time_utc",
            "ingest_time_utc",
            "source_id",
            "raw_metar",
            "temperature_c",
        ]
    )


def _empty_taf_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["station", "issue_time_utc", "knowledge_time_utc", "valid_from_utc", "valid_to_utc", "source_id", "raw_taf"])
