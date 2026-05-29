import argparse
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.rolling_backtest import FULL_WINDOWS, expanding_quantile_backtest, seasonal_breakdown


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    dataset = pd.read_parquet("data/processed/training_dataset.parquet")
    result, summary = expanding_quantile_backtest(
        dataset,
        issue_hours_utc=[0, 6, 12, 18] if args.full else [6, 18],
        windows=FULL_WINDOWS if args.full else None,
    )
    seasons = seasonal_breakdown(result)
    write_parquet(result, "data/reports/rolling_quantile_backtest.parquet")
    write_parquet(summary, "data/reports/rolling_quantile_summary.parquet")
    write_parquet(seasons, "data/reports/rolling_quantile_seasonal_summary.parquet")
    print(f"Wrote {len(result)} rolling forecast rows")


if __name__ == "__main__":
    main()
