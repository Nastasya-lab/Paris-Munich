from __future__ import annotations

from datetime import datetime

import pandas as pd

from weather_tmax_bot.utils.time import ensure_utc


def build_nwp_features(nwp: pd.DataFrame, issue_time_utc: datetime) -> dict:
    issue = ensure_utc(issue_time_utc)
    base = {"nwp_missing": True, "model_tmax_c": None}
    if nwp.empty:
        return base
    df = nwp.copy()
    df["model_availability_time_utc"] = pd.to_datetime(df["model_availability_time_utc"], utc=True)
    df = df[df["model_availability_time_utc"] <= pd.Timestamp(issue)].sort_values("model_availability_time_utc")
    if df.empty:
        return base
    latest = df.iloc[-1]
    out = {
        "nwp_missing": False,
        "model_tmax_c": latest.get("model_tmax_c"),
        "latest_nwp_model_name": latest.get("model_name"),
        "latest_nwp_source_id": latest.get("source_id"),
        "max_nwp_knowledge_time_utc": latest.get("knowledge_time_utc", latest.get("model_availability_time_utc")),
    }
    for col in df.columns:
        if col.startswith("model_") and col not in out:
            out[col] = latest.get(col)
    return out
