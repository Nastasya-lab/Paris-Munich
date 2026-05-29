import pandas as pd

from weather_tmax_bot.evaluation.quantile_backtest import holdout_quantile_backtest


def test_quantile_backtest_returns_raw_and_calibrated():
    dates = pd.date_range("2023-01-01", "2025-03-31", freq="D")
    df = pd.DataFrame(
        {
            "target_date_local": dates.date.astype(str),
            "issue_hour_utc": 6,
            "doy_sin": 0.0,
            "doy_cos": 1.0,
            "month": dates.month,
            "tmax_c": 10 + dates.dayofyear.to_numpy() * 0.01,
        }
    )
    result, metrics = holdout_quantile_backtest(df, test_start="2025-01-01", issue_hour_utc=6)
    assert set(result["forecast_variant"]) == {"raw", "calibrated_spread", "calibrated_isotonic_cdf"}
    assert "calibrated_spread" in metrics
    assert "calibrated_isotonic_cdf" in metrics
