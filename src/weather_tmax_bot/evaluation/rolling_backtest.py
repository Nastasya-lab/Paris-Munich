from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.calibration import DiscreteSpreadCalibrator, IntegerCDFIsotonicCalibrator, pit_values
from weather_tmax_bot.models.quantile_model import QuantileTmaxModel


@dataclass(frozen=True)
class RollingWindow:
    train_end: str
    calibrate_start: str
    calibrate_end: str
    test_start: str
    test_end: str


FULL_WINDOWS = [
    RollingWindow("2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    RollingWindow("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]
DEFAULT_WINDOWS = [
    RollingWindow("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]


def expanding_quantile_backtest(
    dataset: pd.DataFrame,
    issue_hours_utc: list[int] | None = None,
    windows: list[RollingWindow] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue_hours_utc = issue_hours_utc or [6, 18]
    windows = windows or DEFAULT_WINDOWS
    df = dataset.copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    rows = []
    for issue_hour in issue_hours_utc:
        hour_df = df[df["issue_hour_utc"] == issue_hour].copy()
        for window in windows:
            train = hour_df[hour_df["target_date_local"] <= _date(window.train_end)]
            calibrate = hour_df[
                (hour_df["target_date_local"] >= _date(window.calibrate_start))
                & (hour_df["target_date_local"] <= _date(window.calibrate_end))
            ]
            test = hour_df[
                (hour_df["target_date_local"] >= _date(window.test_start))
                & (hour_df["target_date_local"] <= _date(window.test_end))
            ]
            if len(train) < 365 or len(calibrate) < 30 or len(test) < 30:
                continue
            model = QuantileTmaxModel().fit(train.drop(columns=["tmax_c"]), train["tmax_c"])
            calibration_distributions = [
                model.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]))
                for _, row in calibrate.iterrows()
            ]
            calibrator = DiscreteSpreadCalibrator().fit(
                calibration_distributions, calibrate["tmax_c"].to_numpy(dtype=float)
            )
            isotonic = IntegerCDFIsotonicCalibrator().fit(
                calibration_distributions, calibrate["tmax_c"].to_numpy(dtype=float)
            )
            for _, row in test.iterrows():
                raw_dist = model.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]))
                rows.append(_score_row(row, raw_dist, "raw", window))
                rows.append(_score_row(row, calibrator.transform(raw_dist), "calibrated_spread", window, calibrator.sigma_bins))
                rows.append(_score_row(row, isotonic.transform(raw_dist), "calibrated_isotonic_cdf", window))
    result = pd.DataFrame(rows)
    summary = summarize_rolling_backtest(result)
    return result, summary


def summarize_rolling_backtest(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    group_cols = ["forecast_variant", "issue_hour_utc"]
    return (
        result.groupby(group_cols, observed=True)
        .apply(_summary_frame, include_groups=False)
        .reset_index()
    )


def seasonal_breakdown(result: pd.DataFrame) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame()
    df = result.copy()
    df["month"] = pd.to_datetime(df["target_date_local"]).dt.month
    df["season"] = df["month"].map(_season)
    return (
        df.groupby(["forecast_variant", "season"], observed=True)
        .apply(_summary_frame, include_groups=False)
        .reset_index()
    )


def _score_row(row: pd.Series, dist, variant: str, window: RollingWindow, sigma_bins: float | None = None) -> dict:
    actual = float(row["tmax_c"])
    return {
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "forecast_variant": variant,
        "window_train_end": window.train_end,
        "window_test_start": window.test_start,
        "window_test_end": window.test_end,
        "calibrator_sigma_bins": sigma_bins,
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_ge_20": brier(dist.threshold_ge(20), actual >= 20),
        "brier_ge_25": brier(dist.threshold_ge(25), actual >= 25),
        "brier_ge_30": brier(dist.threshold_ge(30), actual >= 30),
        "prob_ge_20": dist.threshold_ge(20),
        "prob_ge_25": dist.threshold_ge(25),
        "prob_ge_30": dist.threshold_ge(30),
        "pit": pit_values([dist], [actual])[0],
        "covered_50": dist.interval(0.50)[0] <= actual <= dist.interval(0.50)[1],
        "covered_80": dist.interval(0.80)[0] <= actual <= dist.interval(0.80)[1],
        "covered_90": dist.interval(0.90)[0] <= actual <= dist.interval(0.90)[1],
    }


def _summary_frame(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": len(group),
            "mae_median": mae(group["actual_tmax_c"], group["median_tmax_c"]),
            "rmse_mean": rmse(group["actual_tmax_c"], group["expected_tmax_c"]),
            "mean_nll": float(group["nll"].mean()),
            "mean_crps": float(group["crps"].mean()),
            "coverage_50": float(group["covered_50"].mean()),
            "coverage_80": float(group["covered_80"].mean()),
            "coverage_90": float(group["covered_90"].mean()),
            "brier_ge_20": float(group["brier_ge_20"].mean()),
            "brier_ge_25": float(group["brier_ge_25"].mean()),
            "brier_ge_30": float(group["brier_ge_30"].mean()),
        }
    )


def _date(value: str):
    return pd.to_datetime(value).date()


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"
