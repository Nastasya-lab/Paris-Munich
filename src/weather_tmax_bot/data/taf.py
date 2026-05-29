from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pandas as pd

from weather_tmax_bot.temporal.availability import taf_knowledge_time
from weather_tmax_bot.utils.hashing import text_hash

TX_RE = re.compile(r"\bTX(M?\d{2})/(\d{4})Z\b")
TN_RE = re.compile(r"\bTN(M?\d{2})/(\d{4})Z\b")


def _decode_signed(token: str) -> float:
    return -float(token[1:]) if token.startswith("M") else float(token)


def parse_taf(raw_taf: str) -> dict:
    tx = TX_RE.search(raw_taf)
    tn = TN_RE.search(raw_taf)
    return {
        "raw_taf": raw_taf,
        "taf_tx_c": _decode_signed(tx.group(1)) if tx else None,
        "taf_tn_c": _decode_signed(tn.group(1)) if tn else None,
        "taf_has_rain": any(code in raw_taf for code in (" RA", " SHRA")),
        "taf_has_thunder": " TS" in raw_taf,
        "taf_has_fog": " FG" in raw_taf,
        "taf_has_snow": " SN" in raw_taf,
        "taf_hours_cavok": raw_taf.count("CAVOK"),
        "taf_prob30_bad_weather": "PROB30" in raw_taf and any(code in raw_taf for code in ("TS", "FG", "SN", "RA")),
        "taf_prob40_bad_weather": "PROB40" in raw_taf and any(code in raw_taf for code in ("TS", "FG", "SN", "RA")),
        "raw_record_hash": text_hash(raw_taf),
        "parser_quality_flag": "parsed_basic",
    }


def taf_records_from_raw(
    rows: list[tuple[str, datetime, datetime | None, datetime | None]],
    source_id: str = "iem.taf.archive.EDDM",
) -> pd.DataFrame:
    parsed = []
    ingest = datetime.now(timezone.utc)
    for raw, issue_time, valid_from, valid_to in rows:
        payload = parse_taf(raw)
        payload.update(
            {
                "station": raw.split()[1] if raw.startswith("TAF") and len(raw.split()) > 1 else raw.split()[0],
                "issue_time_utc": issue_time.astimezone(timezone.utc),
                "knowledge_time_utc": taf_knowledge_time(issue_time),
                "valid_from_utc": valid_from or issue_time,
                "valid_to_utc": valid_to or issue_time + timedelta(hours=24),
                "ingest_time_utc": ingest,
                "source_id": source_id,
                "source_version": "mvp.basic_taf_parser",
            }
        )
        parsed.append(payload)
    return pd.DataFrame(parsed)
