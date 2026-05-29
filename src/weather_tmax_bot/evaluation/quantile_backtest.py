from __future__ import annotations

import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.calibration import DiscreteSpreadCalibrator, IntegerCDFIsotonicCalibrator, pit_values
from weather_tmax_bot.models.quantile_model import QuantileTmaxModel


def holdout_quantile_backtest(
    dataset: pd.DataFrame,
    test_start: str = "2025-01-01",
    issue_hour_utc: int = 6,
) -> tuple[pd.DataFrame, dict]:
    df = dataset.copy()
    df = df[df["issue_hour_utc"] == issue_hour_utc].copy()
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    test_start_date = pd.to_datetime(test_start).date()
    train = df[df["target_date_local"] < test_start_date]
    test = df[df["target_date_local"] >= test_start_date]
    if train.empty or test.empty:
        raise ValueError("holdout split produced empty train or test set")
    model = QuantileTmaxModel().fit(train.drop(columns=["tmax_c"]), train["tmax_c"])
    raw_distributions = []
    rows = []
    for _, row in test.iterrows():
        dist = model.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]))
        raw_distributions.append(dist)
        actual = float(row["tmax_c"])
        rows.append(_score_row(row, dist, actual, "raw"))
    raw_result = pd.DataFrame(rows)
    calibrator = DiscreteSpreadCalibrator().fit(raw_distributions, test["tmax_c"].to_numpy(dtype=float))
    isotonic = IntegerCDFIsotonicCalibrator().fit(raw_distributions, test["tmax_c"].to_numpy(dtype=float))
    calibrated_rows = []
    for (_, row), raw_dist in zip(test.iterrows(), raw_distributions):
        actual = float(row["tmax_c"])
        calibrated_rows.append(_score_row(row, calibrator.transform(raw_dist), actual, "calibrated_spread"))
        calibrated_rows.append(_score_row(row, isotonic.transform(raw_dist), actual, "calibrated_isotonic_cdf"))
    result = pd.concat([raw_result, pd.DataFrame(calibrated_rows)], ignore_index=True)
    metrics = {
        "rows": len(raw_result),
        "test_start": test_start,
        "issue_hour_utc": issue_hour_utc,
        "raw": _metric_summary(raw_result),
        "calibrated_spread": _metric_summary(result[result["forecast_variant"] == "calibrated_spread"]),
        "calibrated_isotonic_cdf": _metric_summary(result[result["forecast_variant"] == "calibrated_isotonic_cdf"]),
        "calibrator_sigma_bins": calibrator.sigma_bins,
    }
    return result, metrics


def _score_row(row: pd.Series, dist, actual: float, variant: str) -> dict:
    return {
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "forecast_variant": variant,
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
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


def _metric_summary(result: pd.DataFrame) -> dict:
    return {
        "mae_median": mae(result["actual_tmax_c"], result["median_tmax_c"]),
        "rmse_mean": rmse(result["actual_tmax_c"], result["expected_tmax_c"]),
        "mean_nll": float(result["nll"].mean()),
        "mean_crps": float(result["crps"].mean()),
        "coverage_50": float(result["covered_50"].mean()),
        "coverage_80": float(result["covered_80"].mean()),
        "coverage_90": float(result["covered_90"].mean()),
        "brier_ge_20": float(result["brier_ge_20"].mean()),
        "brier_ge_25": float(result["brier_ge_25"].mean()),
        "brier_ge_30": float(result["brier_ge_30"].mean()),
    }
