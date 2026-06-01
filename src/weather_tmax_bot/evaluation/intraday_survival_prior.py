from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution

LOCAL_TZ = ZoneInfo("Europe/Berlin")
SEASON_ORDER = ("winter_DJF", "spring_MAM", "summer_JJA", "autumn_SON")


@dataclass(frozen=True)
class SurvivalAdjustment:
    distribution: TmaxDistribution
    observed_max_bin_c: int
    original_upside_probability: float
    adjusted_upside_probability: float
    survival_prior: float
    formula: str
    strength: float


def season_for_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter_DJF"
    if month in (3, 4, 5):
        return "spring_MAM"
    if month in (6, 7, 8):
        return "summer_JJA"
    return "autumn_SON"


def build_daily_first_metar_max(metar: pd.DataFrame, *, min_obs_count: int = 36) -> pd.DataFrame:
    if metar.empty:
        return pd.DataFrame()
    frame = metar.copy()
    frame["observation_time_utc"] = pd.to_datetime(frame["observation_time_utc"], utc=True, errors="coerce")
    frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    frame = frame.dropna(subset=["observation_time_utc", "temperature_c"])
    if frame.empty:
        return pd.DataFrame()
    frame["local_time"] = frame["observation_time_utc"].dt.tz_convert(LOCAL_TZ)
    frame["target_date_local"] = frame["local_time"].dt.date

    rows = []
    for target_date, day in frame.groupby("target_date_local", sort=True):
        day = day.sort_values("local_time")
        if len(day) < min_obs_count:
            continue
        tmax = float(day["temperature_c"].max())
        first = day[day["temperature_c"] == tmax].iloc[0]
        rows.append(
            {
                "target_date_local": target_date,
                "season": season_for_month(target_date.month),
                "tmax_c": tmax,
                "first_max_time_local": first["local_time"],
                "first_max_hour_local": int(first["local_time"].hour),
                "obs_count": int(len(day)),
            }
        )
    return pd.DataFrame(rows)


def build_seasonal_hourly_survival_table(
    daily_first_max: pd.DataFrame,
    *,
    train_before: date | None = None,
) -> pd.DataFrame:
    frame = daily_first_max.copy()
    if train_before is not None:
        frame = frame[frame["target_date_local"] < train_before]
    rows = []
    for season in SEASON_ORDER:
        group = frame[frame["season"] == season]
        for hour in range(24):
            rows.append(
                {
                    "season": season,
                    "local_hour": hour,
                    "training_days": int(len(group)),
                    "peak_ahead_days": int((group["first_max_hour_local"] > hour).sum()),
                    "survival_prior": float((group["first_max_hour_local"] > hour).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def lookup_survival_prior(table: pd.DataFrame, *, month: int, local_hour: float) -> float:
    season = season_for_month(month)
    hour = int(np.floor(local_hour))
    row = table[(table["season"] == season) & (table["local_hour"] == hour)]
    if row.empty or pd.isna(row.iloc[0]["survival_prior"]):
        return 1.0
    return float(np.clip(row.iloc[0]["survival_prior"], 0.0, 1.0))


def adjust_upside_probability(
    distribution: TmaxDistribution,
    *,
    observed_max_so_far_c: float,
    survival_prior: float,
    formula: str,
    strength: float,
) -> SurvivalAdjustment:
    observed_bin = int(np.ceil(observed_max_so_far_c))
    probs = distribution.probabilities.copy()
    observed_mask = distribution.bins_c == observed_bin
    upside_mask = distribution.bins_c > observed_bin
    original_upside = float(probs[upside_mask].sum())
    survival = float(np.clip(survival_prior, 0.0, 1.0))
    strength = float(np.clip(strength, 0.0, 1.0))

    if formula == "cap_blend":
        target_upside = original_upside - strength * max(0.0, original_upside - survival)
    elif formula == "multiply":
        target_upside = original_upside * survival**strength
    else:
        raise ValueError(f"unknown survival adjustment formula: {formula}")

    target_upside = float(np.clip(target_upside, 0.0, original_upside))
    removed = original_upside - target_upside
    if original_upside > 0:
        probs[upside_mask] *= target_upside / original_upside
    if observed_mask.any():
        probs[observed_mask] += removed
    else:
        raise ValueError(f"distribution is missing observed Tmax bin {observed_bin}")
    adjusted = TmaxDistribution(distribution.bins_c, probs)
    return SurvivalAdjustment(
        distribution=adjusted,
        observed_max_bin_c=observed_bin,
        original_upside_probability=original_upside,
        adjusted_upside_probability=target_upside,
        survival_prior=survival,
        formula=formula,
        strength=strength,
    )
