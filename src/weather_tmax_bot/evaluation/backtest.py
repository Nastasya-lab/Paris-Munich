from __future__ import annotations

from datetime import date

import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.baselines import ClimatologyBaseline


def backtest_climatology(daily_target: pd.DataFrame, start_test: date | None = None) -> pd.DataFrame:
    df = daily_target.copy().sort_values("target_date_local")
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    if start_test is None:
        start_test = df["target_date_local"].quantile(0.8)
    rows = []
    for _, row in df[df["target_date_local"] >= start_test].iterrows():
        train = df[df["target_date_local"] < row["target_date_local"]]
        dist = ClimatologyBaseline().fit(train).predict_distribution(row["target_date_local"])
        actual = float(row["tmax_c"])
        rows.append(
            {
                "target_date_local": row["target_date_local"],
                "actual_tmax_c": actual,
                "expected_tmax_c": dist.expected_tmax_c,
                "median_tmax_c": dist.median_tmax_c,
                "nll": nll_integer_bin(dist, actual),
                "crps": crps_discrete(dist, actual),
                "brier_ge_20": brier(dist.threshold_ge(20), actual >= 20),
                "brier_ge_25": brier(dist.threshold_ge(25), actual >= 25),
                "brier_ge_30": brier(dist.threshold_ge(30), actual >= 30),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        out.attrs["mae_median"] = mae(out["actual_tmax_c"], out["median_tmax_c"])
        out.attrs["rmse_mean"] = rmse(out["actual_tmax_c"], out["expected_tmax_c"])
    return out
