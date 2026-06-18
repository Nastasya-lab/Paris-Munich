from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_intraday_survival import apply_metar_intraday_survival_layer


def main() -> None:
    args = _parse_args()
    dataset = pd.read_parquet(args.dataset)
    metadata = _load_json(args.metadata)
    model = joblib.load(args.model)

    frame = _prepare_frame(dataset)
    split = metadata.get("split") or {}
    test_start = split.get("test_start")
    test_end = split.get("test_end")
    if not test_start or not test_end:
        raise ValueError("metadata split must include test_start and test_end")

    prior_history = frame[frame["target_date_local"] < test_start].copy()
    test = frame[(frame["target_date_local"] >= test_start) & (frame["target_date_local"] <= test_end)].copy()
    if test.empty:
        raise ValueError("no holdout rows found for requested split")

    rows = []
    for _, row in test.iterrows():
        base = model.predict_distribution(row)
        adjusted = apply_metar_intraday_survival_layer(
            base,
            row,
            historical_dataset=prior_history,
        ).distribution
        rows.append(_score("base_icon_d2_metar_tmax", row, base))
        rows.append(_score("base_plus_intraday_survival", row, adjusted))

    scored = pd.DataFrame(rows)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _group_summary(scored, ["model_variant", "season"])
    deltas = _deltas(summary)
    report = {
        "analysis": "LFPB historical holdout replay for Paris intraday survival layer.",
        "model": str(args.model),
        "dataset": str(args.dataset),
        "test_period": [test_start, test_end],
        "test_rows": int(len(test)),
        "test_days": int(test["target_date_local"].nunique()),
        "prior_history_rows": int(len(prior_history)),
        "prior_history_days": int(prior_history["target_date_local"].nunique()),
        "leakage_policy": "survival priors are fitted only on rows before test_start; test targets are used only for scoring",
        "summary": json.loads(summary.to_json(orient="records")),
        "deltas_base_plus_survival_minus_base": deltas,
        "created_outputs": {
            "rows": str(Path(args.output_dir) / "lfpb_intraday_survival_backtest_rows.parquet"),
            "summary": str(Path(args.output_dir) / "lfpb_intraday_survival_backtest_summary.csv"),
            "by_hour": str(Path(args.output_dir) / "lfpb_intraday_survival_backtest_by_hour.csv"),
            "by_season": str(Path(args.output_dir) / "lfpb_intraday_survival_backtest_by_season.csv"),
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output_dir / "lfpb_intraday_survival_backtest_rows.parquet", index=False)
    summary.to_csv(output_dir / "lfpb_intraday_survival_backtest_summary.csv", index=False)
    by_hour.to_csv(output_dir / "lfpb_intraday_survival_backtest_by_hour.csv", index=False)
    by_season.to_csv(output_dir / "lfpb_intraday_survival_backtest_by_season.csv", index=False)
    (output_dir / "lfpb_intraday_survival_backtest.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _prepare_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["target_date_local"] = frame["target_date_local"].astype(str)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["season"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.month.map(_season)
    frame = frame[frame["leakage_check_passed"].fillna(False).astype(bool)].copy()
    frame = frame.dropna(subset=["final_metar_tmax_c", "current_metar_max_c", "model_tmax_c"])
    return frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    actual_bin = int(round(actual))
    probability_actual_bin = _probability_for_bin(dist, actual_bin)
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": row.get("season"),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "latest_metar_temp_c": float(row["latest_metar_temp_c"]),
        "drop_from_current_max_c": float(row["drop_from_current_max_c"]),
        "has_rain_recent_metar": bool(row.get("has_rain_recent_metar", False)),
        "model_future_temp_max_c": _optional_float(row, "model_future_temp_max_c"),
        "nwp_future_minus_current_max_c": _optional_float(row, "nwp_future_minus_current_max_c"),
        "remaining_upside_c": float(row["remaining_upside_c"]),
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "probability_actual_integer_bin": probability_actual_bin,
        "probability_ge_current_plus_1c": dist.threshold_ge(int(np.ceil(current_max + 1))),
        "probability_ge_20c": dist.threshold_ge(20),
        "probability_ge_25c": dist.threshold_ge(25),
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current_max + 1))), bool(row["remaining_upside_c"] >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current_max + 2))), bool(row["remaining_upside_c"] >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current_max + 3))), bool(row["remaining_upside_c"] >= 3.0)),
        "coverage_80": _covered(dist, actual, 0.80),
    }


def _group_summary(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(columns, keys)),
                "rows": len(group),
                "distinct_days": int(group["target_date_local"].nunique()),
                "mae_expected": mae(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "rmse_expected": rmse(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "bias_expected": float(group["bias_expected"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "mean_probability_actual_integer_bin": float(group["probability_actual_integer_bin"].mean()),
                "mean_probability_ge_current_plus_1c": float(group["probability_ge_current_plus_1c"].mean()),
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _deltas(summary: pd.DataFrame) -> dict:
    base = summary[summary["model_variant"] == "base_icon_d2_metar_tmax"]
    adjusted = summary[summary["model_variant"] == "base_plus_intraday_survival"]
    if base.empty or adjusted.empty:
        return {}
    base_row = base.iloc[0]
    adjusted_row = adjusted.iloc[0]
    metrics = [
        "mae_expected",
        "rmse_expected",
        "bias_expected",
        "mean_nll",
        "mean_crps",
        "mean_probability_actual_integer_bin",
        "mean_probability_ge_current_plus_1c",
        "brier_upside_ge_1c",
        "brier_upside_ge_2c",
        "brier_upside_ge_3c",
        "coverage_80",
    ]
    return {metric: float(adjusted_row[metric] - base_row[metric]) for metric in metrics}


def _probability_for_bin(dist: TmaxDistribution, actual_bin: int) -> float:
    mask = dist.bins_c == actual_bin
    return float(dist.probabilities[mask].sum()) if mask.any() else 0.0


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _season(month: int | float | None) -> str:
    if month is None or pd.isna(month):
        return "unknown"
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _optional_float(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest LFPB intraday survival layer on historical holdout rows.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--model", default="data/models/lfpb_metar_tmax_icon_d2_v1.joblib")
    parser.add_argument("--metadata", default="data/reports/lfpb_icon_d2_metar_tmax_training.json")
    parser.add_argument("--output-dir", default="data/reports")
    return parser.parse_args()


if __name__ == "__main__":
    main()
