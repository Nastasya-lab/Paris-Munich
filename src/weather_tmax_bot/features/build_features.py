from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from weather_tmax_bot.features.climatology_features import day_of_year_sin_cos
from weather_tmax_bot.features.issue_time_features import ISSUE_HOURS_UTC, build_issue_time_features
from weather_tmax_bot.features.metar_features import build_metar_features
from weather_tmax_bot.features.nwp_features import build_nwp_features
from weather_tmax_bot.features.taf_features import build_taf_features
from weather_tmax_bot.temporal.leakage_detector import LeakageDetector


def build_feature_row(
    airport_icao: str,
    issue_time_utc: datetime,
    target_date_local: date,
    metar: pd.DataFrame | None = None,
    taf: pd.DataFrame | None = None,
    nwp: pd.DataFrame | None = None,
) -> dict:
    features = {
        "airport_icao": airport_icao,
        "issue_time_utc": issue_time_utc,
        "target_date_local": target_date_local.isoformat(),
        "issue_hour_utc": issue_time_utc.hour,
    }
    features.update(build_issue_time_features(issue_time_utc, target_date_local, ISSUE_HOURS_UTC))
    features.update(day_of_year_sin_cos(target_date_local))
    features.update(build_metar_features(metar if metar is not None else pd.DataFrame(), issue_time_utc, target_date_local))
    features.update(build_taf_features(taf if taf is not None else pd.DataFrame(), issue_time_utc, target_date_local))
    features.update(build_nwp_features(nwp if nwp is not None else pd.DataFrame(), issue_time_utc))
    audit = LeakageDetector().audit_feature_frame(
        pd.DataFrame([features]), issue_time_utc=issue_time_utc, target_date_local=target_date_local
    )
    features["leakage_check_passed"] = audit["passed"]
    features["max_feature_knowledge_time_utc"] = audit["max_feature_knowledge_time_utc"]
    return features
