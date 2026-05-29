import pandas as pd

from weather_tmax_bot.evaluation.rolling_backtest import RollingWindow, expanding_quantile_backtest


def test_expanding_quantile_backtest_smoke():
    dates = pd.date_range("2022-01-01", "2025-03-31", freq="D")
    df = pd.DataFrame(
        {
            "target_date_local": dates.date.astype(str),
            "issue_hour_utc": 6,
            "doy_sin": 0.0,
            "doy_cos": 1.0,
            "month": dates.month,
            "tmax_c": 8 + dates.dayofyear.to_numpy() * 0.02,
        }
    )
    window = RollingWindow("2023-12-31", "2024-01-01", "2024-12-31", "2025-01-01", "2025-03-31")
    result, summary = expanding_quantile_backtest(df, issue_hours_utc=[6], windows=[window])
    assert not result.empty
    assert set(result["forecast_variant"]) == {"raw", "calibrated_spread", "calibrated_isotonic_cdf"}
    assert not summary.empty
