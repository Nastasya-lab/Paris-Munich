from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

from weather_tmax_bot.data.metar import parse_metar
from weather_tmax_bot.data.taf import parse_taf
from weather_tmax_bot.temporal.availability import metar_knowledge_time, taf_knowledge_time
from weather_tmax_bot.utils.hashing import stable_hash, text_hash


AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
AWC_TAF_URL = "https://aviationweather.gov/api/data/taf"


class AWCAdapter:
    def __init__(self, user_agent: str = "weather-tmax-bot/0.1"):
        self.headers = {"User-Agent": user_agent}

    def fetch_latest_metar(self, airport: str = "EDDM", hours: int | None = None) -> pd.DataFrame:
        params = {"ids": airport, "format": "json"}
        if hours is not None:
            params["hours"] = int(hours)
        response = requests.get(
            AWC_METAR_URL,
            params=params,
            headers=self.headers,
            timeout=30,
        )
        if response.status_code == 204:
            return _empty_metar_frame()
        response.raise_for_status()
        payload = response.json()
        rows = []
        ingest = datetime.now(timezone.utc)
        for item in payload if isinstance(payload, list) else [payload]:
            raw = item.get("rawOb") or item.get("raw_text") or item.get("rawTAF") or ""
            obs_time = _parse_awc_time(item.get("obsTime") or item.get("reportTime"))
            if pd.isna(obs_time):
                obs_time = pd.Timestamp(ingest)
            parsed = parse_metar(raw) if raw else {}
            obs_dt = obs_time.floor("s").to_pydatetime()
            rows.append(
                {
                    "station": item.get("icaoId", airport),
                    "observation_time_utc": obs_dt,
                    "knowledge_time_utc": metar_knowledge_time(obs_dt),
                    "ingest_time_utc": ingest,
                    "source_id": f"awc.metar.live.{airport}",
                    "source_version": "awc.api.data.metar.json",
                    "source_url_or_reference": AWC_METAR_URL,
                    "raw_metar": raw,
                    "temperature_c": _numeric_or_none(_coalesce(item.get("temp"), parsed.get("temperature_c"))),
                    "dewpoint_c": _numeric_or_none(_coalesce(item.get("dewp"), parsed.get("dewpoint_c"))),
                    "qnh_hpa": parsed.get("qnh_hpa"),
                    "wind_direction_deg": _numeric_or_none(_coalesce(item.get("wdir"), parsed.get("wind_direction_deg"))),
                    "wind_speed_kt": _numeric_or_none(_coalesce(item.get("wspd"), parsed.get("wind_speed_kt"))),
                    "gust_kt": _numeric_or_none(_coalesce(item.get("wgst"), parsed.get("gust_kt"))),
                    "visibility": item.get("visib"),
                    "weather_codes": item.get("wxString"),
                    "cloud_layers": stable_hash(item.get("clouds", [])),
                    "ceiling_ft": item.get("ceil"),
                    "cavok": bool(parsed.get("cavok", False)),
                    "parser_quality_flag": "awc_json_plus_basic_parser",
                    "raw_record_hash": text_hash(raw or stable_hash(item)),
                }
            )
        return pd.DataFrame(rows)

    def fetch_latest_taf(self, airport: str = "EDDM") -> pd.DataFrame:
        response = requests.get(
            AWC_TAF_URL,
            params={"ids": airport, "format": "json"},
            headers=self.headers,
            timeout=30,
        )
        if response.status_code == 204:
            return _empty_taf_frame()
        response.raise_for_status()
        payload = response.json()
        rows = []
        ingest = datetime.now(timezone.utc)
        for item in payload if isinstance(payload, list) else [payload]:
            raw = item.get("rawTAF") or item.get("raw_text") or ""
            issue_time = _parse_awc_time(item.get("issueTime") or item.get("reportTime"))
            if pd.isna(issue_time):
                issue_time = pd.Timestamp(ingest)
            valid_from = _parse_awc_time(item.get("validTimeFrom"))
            valid_to = _parse_awc_time(item.get("validTimeTo"))
            issue_dt = issue_time.floor("s").to_pydatetime()
            parsed = parse_taf(raw) if raw else {}
            parsed.update(
                {
                    "station": item.get("icaoId", airport),
                    "issue_time_utc": issue_dt,
                    "knowledge_time_utc": taf_knowledge_time(issue_dt),
                    "valid_from_utc": None if pd.isna(valid_from) else valid_from.floor("s").to_pydatetime(),
                    "valid_to_utc": None if pd.isna(valid_to) else valid_to.floor("s").to_pydatetime(),
                    "ingest_time_utc": ingest,
                    "source_id": f"awc.taf.live.{airport}",
                    "source_version": "awc.api.data.taf.json",
                    "source_url_or_reference": AWC_TAF_URL,
                    "raw_taf": raw,
                    "raw_record_hash": text_hash(raw or stable_hash(item)),
                }
            )
            rows.append(parsed)
        return pd.DataFrame(rows)


def _coalesce(primary, fallback):
    return fallback if primary is None or pd.isna(primary) else primary


def _numeric_or_none(value):
    parsed = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def _parse_awc_time(value):
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) or str(value).isdigit():
        numeric = float(value)
        unit = "s" if numeric < 10_000_000_000 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(value, utc=True, errors="coerce")


def _empty_metar_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["station", "observation_time_utc", "knowledge_time_utc", "source_id", "raw_metar"])


def _empty_taf_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["station", "issue_time_utc", "knowledge_time_utc", "valid_from_utc", "valid_to_utc", "source_id", "raw_taf"])
