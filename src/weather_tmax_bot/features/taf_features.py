from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from weather_tmax_bot.utils.time import ensure_utc


def build_taf_features(taf: pd.DataFrame, issue_time_utc: datetime, target_date_local: date) -> dict:
    issue = ensure_utc(issue_time_utc)
    if taf.empty:
        return {"taf_missing": True}
    df = taf.copy()
    df["knowledge_time_utc"] = pd.to_datetime(df["knowledge_time_utc"], utc=True)
    df["issue_time_utc"] = pd.to_datetime(df["issue_time_utc"], utc=True)
    df = df[df["knowledge_time_utc"] <= pd.Timestamp(issue)].sort_values("issue_time_utc")
    if df.empty:
        return {"taf_missing": True}
    last = df.iloc[-1]
    age = (pd.Timestamp(issue) - last["issue_time_utc"]).total_seconds() / 3600
    return {
        "taf_missing": False,
        "latest_taf_issue_time_utc": last.get("issue_time_utc"),
        "latest_taf_source_id": last.get("source_id"),
        "taf_hours_cavok": last.get("taf_hours_cavok", 0),
        "taf_has_rain": bool(last.get("taf_has_rain", False)),
        "taf_has_shower": "SH" in str(last.get("raw_taf", "")),
        "taf_has_thunder": bool(last.get("taf_has_thunder", False)),
        "taf_has_fog": bool(last.get("taf_has_fog", False)),
        "taf_has_snow": bool(last.get("taf_has_snow", False)),
        "taf_wind_shift_flag": "BECMG" in str(last.get("raw_taf", "")) or "FM" in str(last.get("raw_taf", "")),
        "taf_prob30_bad_weather": bool(last.get("taf_prob30_bad_weather", False)),
        "taf_prob40_bad_weather": bool(last.get("taf_prob40_bad_weather", False)),
        "taf_tx_c": last.get("taf_tx_c", np.nan),
        "taf_tn_c": last.get("taf_tn_c", np.nan),
        "taf_age_hours_at_issue": age,
    }
