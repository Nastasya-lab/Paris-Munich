from __future__ import annotations

import re
from datetime import datetime, timezone

import pandas as pd

from weather_tmax_bot.temporal.availability import metar_knowledge_time
from weather_tmax_bot.utils.hashing import text_hash

TEMP_RE = re.compile(r"\b(M?\d{2})/(M?\d{2})\b")
QNH_RE = re.compile(r"\bQ(\d{4})\b")
WIND_RE = re.compile(r"\b(\d{3}|VRB)(\d{2})(G(\d{2}))?KT\b")


def _decode_signed(token: str) -> float:
    return -float(token[1:]) if token.startswith("M") else float(token)


def parse_metar(raw_metar: str) -> dict:
    temp = TEMP_RE.search(raw_metar)
    qnh = QNH_RE.search(raw_metar)
    wind = WIND_RE.search(raw_metar)
    weather_matches = re.findall(r"(?<![A-Z0-9])(?:-|\+|VC)?(?:TSRA|SHRA|FZRA|RA|SN|TS|FG|BR|DZ)(?![A-Z0-9])", raw_metar)
    return {
        "raw_metar": raw_metar,
        "temperature_c": _decode_signed(temp.group(1)) if temp else None,
        "dewpoint_c": _decode_signed(temp.group(2)) if temp else None,
        "qnh_hpa": float(qnh.group(1)) if qnh else None,
        "wind_direction_deg": None if not wind or wind.group(1) == "VRB" else float(wind.group(1)),
        "wind_speed_kt": float(wind.group(2)) if wind else None,
        "gust_kt": float(wind.group(4)) if wind and wind.group(4) else None,
        "cavok": "CAVOK" in raw_metar,
        "weather_codes": " ".join(weather_matches),
        "raw_record_hash": text_hash(raw_metar),
        "parser_quality_flag": "parsed_basic",
    }


def metar_records_from_raw(rows: list[tuple[str, datetime]], source_id: str = "iem.metar.archive.EDDM") -> pd.DataFrame:
    parsed = []
    ingest = datetime.now(timezone.utc)
    for raw, obs_time in rows:
        payload = parse_metar(raw)
        payload.update(
            {
                "station": raw.split()[0] if raw.split() else None,
                "observation_time_utc": obs_time.astimezone(timezone.utc),
                "knowledge_time_utc": metar_knowledge_time(obs_time),
                "ingest_time_utc": ingest,
                "source_id": source_id,
                "source_version": "mvp.basic_metar_parser",
            }
        )
        parsed.append(payload)
    return pd.DataFrame(parsed)
