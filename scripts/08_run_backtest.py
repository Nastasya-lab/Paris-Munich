from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.backtest import backtest_climatology
from weather_tmax_bot.evaluation.plots import save_pit_histogram, save_reliability_curve
from weather_tmax_bot.evaluation.quantile_backtest import holdout_quantile_backtest
from weather_tmax_bot.evaluation.reliability import reliability_table


def main():
    target = pd.read_parquet("data/processed/daily_target.parquet")
    target = target[target["quality_flags"] == "ok"].copy()
    result = backtest_climatology(target)
    write_parquet(result, "data/reports/climatology_backtest.parquet")
    quantile_text = ""
    try:
        dataset = pd.read_parquet("data/processed/training_dataset.parquet")
        q_result, q_metrics = holdout_quantile_backtest(dataset)
        write_parquet(q_result, "data/reports/quantile_holdout_backtest.parquet")
        calibrated = q_result[q_result["forecast_variant"] == "calibrated_spread"]
        raw = q_result[q_result["forecast_variant"] == "raw"]
        save_pit_histogram(raw["pit"], "data/reports/pit_raw.png")
        save_pit_histogram(calibrated["pit"], "data/reports/pit_calibrated_spread.png")
        rel25 = reliability_table(calibrated["prob_ge_25"], calibrated["actual_tmax_c"] >= 25)
        rel30 = reliability_table(calibrated["prob_ge_30"], calibrated["actual_tmax_c"] >= 30)
        write_parquet(rel25, "data/reports/reliability_ge_25_calibrated.parquet")
        write_parquet(rel30, "data/reports/reliability_ge_30_calibrated.parquet")
        save_reliability_curve(rel25, "data/reports/reliability_ge_25_calibrated.png", "P(Tmax >= 25C)")
        save_reliability_curve(rel30, "data/reports/reliability_ge_30_calibrated.png", "P(Tmax >= 30C)")
        raw_m = q_metrics["raw"]
        cal_m = q_metrics["calibrated_spread"]
        iso_m = q_metrics["calibrated_isotonic_cdf"]
        quantile_text = (
            "\n\nQuantile MVP holdout, issue hour 06 UTC, test from 2025-01-01:\n"
            f"Rows: {q_metrics['rows']}\n"
            f"Raw MAE/RMSE: {raw_m['mae_median']} / {raw_m['rmse_mean']}\n"
            f"Raw NLL/CRPS: {raw_m['mean_nll']} / {raw_m['mean_crps']}\n"
            f"Raw coverage 50/80/90: {raw_m['coverage_50']} / {raw_m['coverage_80']} / {raw_m['coverage_90']}\n"
            f"Calibrated spread sigma bins: {q_metrics['calibrator_sigma_bins']}\n"
            f"Calibrated MAE/RMSE: {cal_m['mae_median']} / {cal_m['rmse_mean']}\n"
            f"Calibrated NLL/CRPS: {cal_m['mean_nll']} / {cal_m['mean_crps']}\n"
            f"Calibrated coverage 50/80/90: {cal_m['coverage_50']} / {cal_m['coverage_80']} / {cal_m['coverage_90']}\n"
            f"Calibrated Brier ge20/ge25/ge30: {cal_m['brier_ge_20']} / {cal_m['brier_ge_25']} / {cal_m['brier_ge_30']}\n"
            f"Isotonic CDF MAE/RMSE: {iso_m['mae_median']} / {iso_m['rmse_mean']}\n"
            f"Isotonic CDF NLL/CRPS: {iso_m['mean_nll']} / {iso_m['mean_crps']}\n"
            f"Isotonic CDF coverage 50/80/90: {iso_m['coverage_50']} / {iso_m['coverage_80']} / {iso_m['coverage_90']}\n"
            f"Isotonic CDF Brier ge20/ge25/ge30: {iso_m['brier_ge_20']} / {iso_m['brier_ge_25']} / {iso_m['brier_ge_30']}\n"
            "\nGenerated plots:\n"
            "- `data/reports/pit_raw.png`\n"
            "- `data/reports/pit_calibrated_spread.png`\n"
            "- `data/reports/reliability_ge_25_calibrated.png`\n"
            "- `data/reports/reliability_ge_30_calibrated.png`\n"
        )
    except Exception as exc:
        quantile_text = f"\n\nQuantile MVP holdout was not run: {exc}\n"
    Path("docs/backtest_results.md").write_text(
        "# Backtest results\n\n"
        "Current report covers the climatology baseline on available DWD targets. "
        "NWP forecast-as-issued backtests remain separated until honest issued archives exist.\n\n"
        f"Rows: {len(result)}\n"
        f"MAE median: {result.attrs.get('mae_median')}\n"
        f"RMSE mean: {result.attrs.get('rmse_mean')}\n"
        f"{quantile_text}",
        encoding="utf-8",
    )
    print(f"Wrote backtest with {len(result)} rows")


if __name__ == "__main__":
    main()
