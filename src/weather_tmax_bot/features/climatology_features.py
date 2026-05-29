from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd


def day_of_year_sin_cos(target_date: date) -> dict:
    angle = 2 * np.pi * target_date.timetuple().tm_yday / 366
    return {"doy_sin": float(np.sin(angle)), "doy_cos": float(np.cos(angle)), "month": target_date.month}


def climatology_sample(targets: pd.DataFrame, target_date: date, window_days: int = 30) -> pd.Series:
    if targets.empty:
        return pd.Series(dtype=float)
    df = targets.copy()
    dates = pd.to_datetime(df["target_date_local"]).dt.date
    doy = target_date.timetuple().tm_yday
    dist = dates.map(lambda d: min(abs(d.timetuple().tm_yday - doy), 366 - abs(d.timetuple().tm_yday - doy)))
    return pd.to_numeric(df.loc[dist <= window_days, "tmax_c"], errors="coerce").dropna()
