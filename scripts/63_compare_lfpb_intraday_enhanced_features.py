from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)


ENHANCED_INTRADAY_FEATURES = [
    "temp_slope_since_sunrise",
    "temp_trend_last_2_metars",
    "latest_2_metar_temp_change_c",
    "cloud_cover_proxy_latest",
    "cloud_cover_proxy_trend_last_2_metars",
    "cloud_cover_proxy_trend_2h",
    "lowest_ceiling_ft_latest",
    "ceiling_trend_last_2_metars",
    "ceiling_trend_2h",
    "dewpoint_depression_latest",
    "dewpoint_depression_trend_2h",
    "pressure_tendency_1h",
    "pressure_tendency_3h",
    "wind_dir_shift_2h_deg",
    "wind_speed_trend_2h",
    "wind_direction_latest_deg",
    "wind_speed_latest_kt",
    "rain_started_after_current_max",
    "cb_tcu_appeared_after_current_max",
    "showers_appeared_after_current_max",
    "fog_or_br_recent_metar",
    "cavok_trend_last_2_metars",
    "metar_minutes_since_current_max",
    "metar_hours_since_sunrise",
    "temp_drop_after_rain_start_c",
    "temp_drop_after_cb_tcu_c",
    "wind_direction_valid_count_2h",
]


def main() -> None:
    dataset = pd.read_parquet("data/processed/metar_upside_dataset_LFPB_intraday_enhanced.parquet")
    frame = prepare_metar_tmax_dataset(dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)
    base_features = list(DEFAULT_METAR_TMAX_FEATURES)
    enhanced_features = base_features + ENHANCED_INTRADAY_FEATURES

    base_model = _fit_calibrated(train, calibration, base_features)
    enhanced_model = _fit_calibrated(train, calibration, enhanced_features)
    scored = _score_holdout(test, base_model, enhanced_model)
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _summary(scored, ["model_variant", "season"])
    by_regime = _summary(scored, ["model_variant", "rain_or_cb_after_max"])
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_intraday_enhanced_feature_comparison_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_intraday_enhanced_feature_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_intraday_enhanced_feature_comparison_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_intraday_enhanced_feature_comparison_by_season.csv", index=False)
    by_regime.to_csv(report_dir / "lfpb_intraday_enhanced_feature_comparison_by_regime.csv", index=False)
    base = _row(summary, "base_metar_intraday")
    enhanced = _row(summary, "enhanced_metar_intraday")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "experiment": "base METAR intraday features vs enhanced METAR intraday features",
        "rows": len(frame),
        "days": int(frame["target_date_local"].nunique()),
        "period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "base_feature_count": len(base_features),
        "enhanced_feature_count": len(enhanced_features),
        "enhanced_features": ENHANCED_INTRADAY_FEATURES,
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, enhanced),
    }
    (report_dir / "lfpb_intraday_enhanced_feature_comparison.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/lfpb_intraday_enhanced_feature_comparison.md").write_text(
        _markdown(report, summary, by_hour, by_regime),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _time_split(frame: pd.DataFrame):
    dates = sorted(frame["target_date_local"].unique())
    train_end = max(1, int(len(dates) * 0.60))
    calibration_end = max(train_end + 1, int(len(dates) * 0.80))
    train = frame[frame["target_date_local"].isin(dates[:train_end])].copy()
    calibration = frame[frame["target_date_local"].isin(dates[train_end:calibration_end])].copy()
    test = frame[frame["target_date_local"].isin(dates[calibration_end:])].copy()
    return train, calibration, test, {
        "method": "chronological_60_20_20_by_target_day",
        "train_start": str(train["target_date_local"].min()),
        "train_end": str(train["target_date_local"].max()),
        "calibration_start": str(calibration["target_date_local"].min()),
        "calibration_end": str(calibration["target_date_local"].max()),
        "test_start": str(test["target_date_local"].min()),
        "test_end": str(test["target_date_local"].max()),
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_days": int(train["target_date_local"].nunique()),
        "calibration_days": int(calibration["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
    }


def _fit_calibrated(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str]) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(min_rows=5000, max_iter=50, feature_columns=features).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _survival_calibration_rows(model: MetarTmaxUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_upside_survival_frame(frame)
    rows = []
    for index, row in frame.iterrows():
        out = {
            "target_date_local": str(row["target_date_local"]),
            "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
            "local_issue_hour": int(row["local_issue_hour"]),
            "season": _season(row["target_date_local"]),
            "remaining_upside_c": float(row["remaining_upside_c"]),
        }
        for threshold in range(1, model.max_upside_c + 1):
            out[f"raw_probability_upside_ge_{threshold}c"] = float(raw.loc[index, f"probability_upside_ge_{threshold}c"])
            out[f"actual_upside_ge_{threshold}c"] = float(row["remaining_upside_c"] >= threshold)
        rows.append(out)
    return pd.DataFrame(rows)


def _score_holdout(test: pd.DataFrame, base_model: MetarTmaxUpsideModel, enhanced_model: MetarTmaxUpsideModel) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("base_metar_intraday", row, base_model.predict_distribution(row)))
        rows.append(_score("enhanced_metar_intraday", row, enhanced_model.predict_distribution(row)))
        rows.append(
            _score(
                "persistence_current_metar_max",
                row,
                TmaxDistribution(np.array([int(round(row["current_metar_max_c"]))]), np.array([1.0])),
            )
        )
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": _season(row["target_date_local"]),
        "rain_or_cb_after_max": bool(row.get("rain_started_after_current_max", False) or row.get("cb_tcu_appeared_after_current_max", False)),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current_max + 1))), bool(row["remaining_upside_c"] >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current_max + 2))), bool(row["remaining_upside_c"] >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current_max + 3))), bool(row["remaining_upside_c"] >= 3.0)),
        "coverage_80": _covered(dist, actual, 0.80),
    }


def _summary(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
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
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _recommendation(base: dict, enhanced: dict) -> dict:
    mae_delta = float(enhanced["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(enhanced["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(enhanced["mean_crps"]) - float(base["mean_crps"])
    if mae_delta < -0.03 and nll_delta <= 0.03 and crps_delta <= 0.01:
        decision = "candidate_for_shadow"
        reason = "Enhanced intraday features improved point accuracy without a major probabilistic penalty."
    elif nll_delta < -0.05 and mae_delta <= 0.02:
        decision = "candidate_for_probabilistic_shadow"
        reason = "Enhanced intraday features mostly improve probability quality; keep as shadow before promotion."
    else:
        decision = "do_not_promote_yet"
        reason = "Enhanced intraday features did not pass the promotion gate."
    return {
        "decision": decision,
        "reason": reason,
        "enhanced_minus_base_mae": mae_delta,
        "enhanced_minus_base_nll": nll_delta,
        "enhanced_minus_base_crps": crps_delta,
    }


def _row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB enhanced intraday METAR feature comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- period: `{report['period'][0]}` to `{report['period'][1]}`",
            f"- rows: `{report['rows']}`",
            f"- days: `{report['days']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            f"- reason: {report['recommendation']['reason']}",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## By Hour",
            "",
            _table(by_hour),
            "",
            "## Rain/CB After Max Regime",
            "",
            _table(by_regime),
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
