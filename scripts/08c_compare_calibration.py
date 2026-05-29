import pandas as pd

from weather_tmax_bot.data.storage import write_parquet


def main():
    holdout = pd.read_parquet("data/reports/quantile_holdout_backtest.parquet")
    rows = []
    for variant, group in holdout.groupby("forecast_variant"):
        rows.append(
            {
                "forecast_variant": variant,
                "rows": len(group),
                "mean_nll": group["nll"].mean(),
                "mean_crps": group["crps"].mean(),
                "coverage_50": group["covered_50"].mean(),
                "coverage_80": group["covered_80"].mean(),
                "coverage_90": group["covered_90"].mean(),
                "brier_ge_20": group["brier_ge_20"].mean(),
                "brier_ge_25": group["brier_ge_25"].mean(),
                "brier_ge_30": group["brier_ge_30"].mean(),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["mean_nll", "mean_crps"]).reset_index(drop=True)
    summary["selected_for_production"] = summary["forecast_variant"].eq("calibrated_spread")
    write_parquet(summary, "data/reports/calibration_comparison.parquet")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
