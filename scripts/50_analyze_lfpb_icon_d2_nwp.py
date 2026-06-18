from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import DEFAULT_METAR_TMAX_FEATURES, MetarTmaxUpsideModel


NWP_COLUMNS = [
    "model_tmax_c",
    "model_future_temp_max_c",
    "model_cloud_cover_mean",
    "model_future_cloud_cover_mean",
    "model_precip_sum",
    "model_future_precip_sum",
    "model_shortwave_radiation_sum",
    "model_future_shortwave_radiation_sum",
    "model_wind_speed_max",
    "model_future_wind_speed_max",
    "model_gust_max",
    "model_future_gust_max",
    "model_dewpoint_mean",
    "model_relative_humidity_mean",
    "forecast_horizon_hours",
    "nwp_model_minus_current_max_c",
    "nwp_future_minus_current_max_c",
]


def main() -> None:
    dataset = pd.read_parquet("data/processed/metar_upside_dataset_LFPB.parquet")
    nwp = pd.read_parquet("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    joined = _join_asof_nwp(dataset, nwp)
    if joined.empty:
        raise ValueError("No as-of LFPB ICON-D2 rows available")
    joined.to_parquet("data/processed/metar_upside_dataset_LFPB_icon_d2.parquet", index=False)

    scored, split = _stress_backtest(joined)
    summary = _summary(scored)
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source": "open_meteo.single_run.icon_d2",
        "rows_joined": len(joined),
        "days_joined": int(joined["target_date_local"].nunique()),
        "target_period": [str(joined["target_date_local"].min()), str(joined["target_date_local"].max())],
        "nwp_model_tmax_nonnull_rate": float(joined["model_tmax_c"].notna().mean()),
        "split": split,
        "summary": json.loads(summary.to_json(orient="records")),
        "limitations": [
            "This is a preliminary stress-test on the currently backfilled LFPB ICON-D2 window only.",
            "The full 2025-2026 NWP backfill is not complete yet.",
            "NWP-aware ML is uncalibrated here because the overlap period is still short.",
            "All NWP rows are selected as-of: knowledge_time_utc <= issue_time_utc.",
        ],
    }
    Path("data/reports/lfpb_icon_d2_nwp_stress_summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")
    by_hour.to_csv("data/reports/lfpb_icon_d2_nwp_stress_by_hour.csv", index=False)
    scored.to_parquet("data/reports/lfpb_icon_d2_nwp_stress_rows.parquet", index=False)
    Path("data/reports/lfpb_icon_d2_nwp_stress.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path("docs/lfpb_icon_d2_nwp_stress.md").write_text(_markdown(report, summary, by_hour), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


def _join_asof_nwp(dataset: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    ds = dataset.copy()
    nw = nwp.copy()
    ds["target_date_local"] = ds["target_date_local"].astype(str)
    nw["target_date_local"] = nw["target_date_local"].astype(str)
    ds["issue_time_utc"] = pd.to_datetime(ds["issue_time_utc"], utc=True)
    nw["knowledge_time_utc"] = pd.to_datetime(nw["knowledge_time_utc"], utc=True)
    nw = nw[nw["model_tmax_c"].notna()].sort_values("knowledge_time_utc")
    rows = []
    for _, row in ds.iterrows():
        candidates = nw[
            (nw["target_date_local"] == row["target_date_local"])
            & (nw["knowledge_time_utc"] <= row["issue_time_utc"])
        ]
        if candidates.empty:
            continue
        latest = candidates.iloc[-1]
        merged = row.to_dict()
        for column in NWP_COLUMNS:
            if column in latest:
                merged[column] = latest[column]
        merged["nwp_model_minus_current_max_c"] = float(latest["model_tmax_c"]) - float(row["current_metar_max_c"])
        future = latest.get("model_future_temp_max_c")
        merged["nwp_future_minus_current_max_c"] = np.nan if pd.isna(future) else float(future) - float(row["current_metar_max_c"])
        merged["nwp_knowledge_time_utc"] = latest["knowledge_time_utc"].isoformat()
        merged["nwp_model_run_time_utc"] = pd.Timestamp(latest["model_run_time_utc"]).isoformat()
        merged["nwp_source_id"] = latest["source_id"]
        merged["max_feature_knowledge_time_utc"] = max(
            pd.Timestamp(row["max_feature_knowledge_time_utc"]),
            latest["knowledge_time_utc"],
        ).isoformat()
        merged["leakage_check_passed"] = bool(pd.Timestamp(merged["max_feature_knowledge_time_utc"]) <= row["issue_time_utc"])
        rows.append(merged)
    out = pd.DataFrame(rows)
    return out[out["leakage_check_passed"].fillna(False).astype(bool)].reset_index(drop=True)


def _stress_backtest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    data = frame.copy()
    data["target_date_local"] = pd.to_datetime(data["target_date_local"]).dt.date
    dates = sorted(data["target_date_local"].unique())
    split_index = max(25, int(len(dates) * 0.65))
    split_date = dates[split_index]
    train = data[data["target_date_local"] < split_date].copy()
    test = data[data["target_date_local"] >= split_date].copy()
    if len(train) < 200 or test.empty:
        raise ValueError("Not enough LFPB ICON-D2 overlap for stress backtest")
    feature_columns = list(DEFAULT_METAR_TMAX_FEATURES) + NWP_COLUMNS
    model = MetarTmaxUpsideModel(min_rows=200, max_iter=45, feature_columns=feature_columns).fit(train)
    residuals = _residual_samples(train)
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("nwp_aware_ml_uncalibrated", row, model.predict_distribution(row)))
        rows.append(_score("raw_icon_d2_residual_distribution", row, _raw_icon_residual_distribution(row, residuals, train)))
        rows.append(_score("raw_icon_d2_point", row, TmaxDistribution(np.array([int(round(row["model_tmax_c"]))]), np.array([1.0]))))
        rows.append(_score("persistence_current_metar_max", row, TmaxDistribution(np.array([int(round(row["current_metar_max_c"]))]), np.array([1.0]))))
    return pd.DataFrame(rows), {
        "train_start": str(train["target_date_local"].min()),
        "train_end": str(train["target_date_local"].max()),
        "test_start": str(test["target_date_local"].min()),
        "test_end": str(test["target_date_local"].max()),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_days": int(train["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
    }


def _residual_samples(train: pd.DataFrame) -> dict[int, np.ndarray]:
    train = train.copy()
    train["residual"] = pd.to_numeric(train["final_metar_tmax_c"], errors="coerce") - pd.to_numeric(train["model_tmax_c"], errors="coerce")
    out = {}
    for hour, group in train.groupby("local_issue_hour"):
        values = group["residual"].dropna().to_numpy(dtype=float)
        if len(values):
            out[int(hour)] = values
    out[-1] = train["residual"].dropna().to_numpy(dtype=float)
    return out


def _raw_icon_residual_distribution(row: pd.Series, residuals: dict[int, np.ndarray], train: pd.DataFrame) -> TmaxDistribution:
    samples = residuals.get(int(row["local_issue_hour"]), residuals.get(-1))
    if samples is None or len(samples) < 20:
        samples = residuals.get(-1, np.array([0.0]))
    rounded = np.rint(float(row["model_tmax_c"]) + samples).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    dist = TmaxDistribution(bins, probabilities)
    return dist.truncate_below(float(row["current_metar_max_c"]))


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "actual_metar_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "coverage_80": _covered(dist, actual, 0.80),
    }


def _summary(scored: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or ["model_variant"]
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
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB ICON-D2 NWP stress test",
            "",
            f"- rows joined: `{report['rows_joined']}`",
            f"- days joined: `{report['days_joined']}`",
            f"- target period: `{report['target_period'][0]}` to `{report['target_period'][1]}`",
            f"- split: `{report['split']}`",
            "",
            "## Overall",
            "",
            _table(summary),
            "",
            "## By local issue hour",
            "",
            _table(by_hour),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
