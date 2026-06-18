from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.features.spatial_metar import DEFAULT_SPATIAL_STATIONS, add_spatial_metar_features_to_frame, spatial_feature_columns
from weather_tmax_bot.features.wind_advection import (
    DEFAULT_ADVECTION_STATIONS,
    add_wind_advection_features_to_frame,
    wind_advection_feature_columns,
)
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)


NWP_COLUMNS = [
    "model_tmax_c",
    "model_temp_at_08_local",
    "model_temp_at_11_local",
    "model_temp_at_14_local",
    "model_temp_at_17_local",
    "model_precip_sum_mm",
    "model_cloud_cover_mean_pct",
    "model_cloud_cover_max_pct",
    "model_shortwave_radiation_sum",
    "model_wind_speed_max_kmh",
    "model_wind_gust_max_kmh",
]

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
    args = _parse_args()
    dataset = _load_or_build_dataset(args)
    frame = prepare_metar_tmax_dataset(dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)

    feature_sets = {
        "icon_d2_intraday": list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS),
        "icon_d2_spatial": list(DEFAULT_METAR_TMAX_FEATURES)
        + list(ENHANCED_INTRADAY_FEATURES)
        + list(NWP_COLUMNS)
        + spatial_feature_columns(args.neighbor_station),
        "icon_d2_spatial_wind_advection": list(DEFAULT_METAR_TMAX_FEATURES)
        + list(ENHANCED_INTRADAY_FEATURES)
        + list(NWP_COLUMNS)
        + spatial_feature_columns(args.neighbor_station)
        + wind_advection_feature_columns(args.advection_station),
    }

    residual_ensemble = _fit_residual_ensemble(train)
    models = {
        name: _fit_ensemble(train, calibration, columns, residual_ensemble, args)
        for name, columns in feature_sets.items()
    }
    scored = _score_holdout(test, models)
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _summary(scored, ["model_variant", "season"])
    by_regime = _summary(scored, ["model_variant", "advection_regime"])
    by_availability = _summary(scored, ["model_variant", "adv_available_station_count"])

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_wind_advection_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_wind_advection_backtest_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_wind_advection_backtest_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_wind_advection_backtest_by_season.csv", index=False)
    by_regime.to_csv(report_dir / "lfpb_wind_advection_backtest_by_regime.csv", index=False)
    by_availability.to_csv(report_dir / "lfpb_wind_advection_backtest_by_availability.csv", index=False)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "experiment": "wind/advection METAR features for Paris METAR Tmax",
        "rows": len(frame),
        "days": int(frame["target_date_local"].nunique()),
        "period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "feature_counts": {name: len(columns) for name, columns in feature_sets.items()},
        "wind_advection_feature_columns": wind_advection_feature_columns(args.advection_station),
        "availability": _availability_report(frame, args.advection_station),
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(summary, by_hour, by_regime),
    }
    (report_dir / "lfpb_wind_advection_backtest.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path(args.doc_path).write_text(_markdown(report, summary, by_hour, by_regime, by_availability), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


def _load_or_build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dataset)
    if output.exists() and not args.rebuild:
        return pd.read_parquet(output)

    base = pd.read_parquet(args.dataset)
    if not set(spatial_feature_columns(args.neighbor_station)).issubset(base.columns):
        neighbors = {
            station: pd.read_parquet(f"data/interim/metar_iem_{station}.parquet")
            for station in args.neighbor_station
        }
        base = add_spatial_metar_features_to_frame(
            base,
            neighbors,
            timezone_name=args.timezone,
            stations=args.neighbor_station,
        )
    station_metars = {
        station: pd.read_parquet(f"data/interim/metar_iem_{station}.parquet")
        for station in args.advection_station
    }
    out = add_wind_advection_features_to_frame(
        base,
        station_metars,
        timezone_name=args.timezone,
        stations=args.advection_station,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    return out


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


def _fit_ensemble(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    residual_ensemble: IconD2MetarTmaxEnsemble,
    args: argparse.Namespace,
) -> IconD2MetarTmaxEnsemble:
    model = MetarTmaxUpsideModel(
        min_rows=args.min_train_rows,
        max_iter=args.max_iter,
        feature_columns=features,
    ).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    return IconD2MetarTmaxEnsemble(
        ml_model=model,
        residuals_by_hour=residual_ensemble.residuals_by_hour,
        ml_weight=args.ml_weight,
        model_version="lfpb_wind_advection_backtest",
    )


def _fit_residual_ensemble(frame: pd.DataFrame) -> IconD2MetarTmaxEnsemble:
    data = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    data["residual"] = data["final_metar_tmax_c"].astype(float) - data["model_tmax_c"].astype(float)
    residuals_by_hour = {
        int(hour): group["residual"].to_numpy(dtype=float)
        for hour, group in data.groupby("local_issue_hour")
    }
    residuals_by_hour[-1] = data["residual"].to_numpy(dtype=float)
    dummy = MetarTmaxUpsideModel(min_rows=1)
    dummy.fitted = True
    dummy.training_rows = len(frame)
    return IconD2MetarTmaxEnsemble(dummy, residuals_by_hour, ml_weight=0.0)


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


def _score_holdout(test: pd.DataFrame, models: dict[str, IconD2MetarTmaxEnsemble]) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        for name, model in models.items():
            rows.append(_score(name, row, model.predict_distribution(row)))
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
        "advection_regime": _advection_regime(row),
        "adv_available_station_count": int(row.get("adv_available_station_count", 0)),
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


def _recommendation(summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame) -> dict:
    base = _row(summary, "icon_d2_spatial")
    candidate = _row(summary, "icon_d2_spatial_wind_advection")
    if not base or not candidate:
        return {"decision": "invalid", "reason": "Missing comparison rows."}
    mae_delta = float(candidate["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(base["mean_crps"])
    max_hour_nll_regression = _max_group_regression(by_hour, "icon_d2_spatial_wind_advection", "icon_d2_spatial", "mean_nll", "local_issue_hour")
    max_regime_nll_regression = _max_group_regression(by_regime, "icon_d2_spatial_wind_advection", "icon_d2_spatial", "mean_nll", "advection_regime")
    if mae_delta < -0.02 and nll_delta <= 0.02 and crps_delta <= 0.01 and max_hour_nll_regression <= 0.12:
        decision = "candidate_for_shadow"
        reason = "Wind/advection features improved point accuracy over spatial model without material probabilistic regression."
    elif nll_delta < -0.04 and mae_delta <= 0.03 and max_regime_nll_regression <= 0.15:
        decision = "candidate_for_probabilistic_shadow"
        reason = "Wind/advection features improved probability quality, but point benefit is modest."
    else:
        decision = "do_not_promote_yet"
        reason = "Wind/advection features did not pass the offline gate over the current spatial model."
    return {
        "decision": decision,
        "reason": reason,
        "candidate_minus_spatial_mae": mae_delta,
        "candidate_minus_spatial_nll": nll_delta,
        "candidate_minus_spatial_crps": crps_delta,
        "max_hour_nll_regression": max_hour_nll_regression,
        "max_regime_nll_regression": max_regime_nll_regression,
    }


def _availability_report(frame: pd.DataFrame, stations: list[str]) -> dict:
    out = {
        "any_station_available_rate": float((frame["adv_available_station_count"] > 0).mean()),
        "all_stations_available_rate": float((frame["adv_available_station_count"] >= len(stations)).mean()),
    }
    for station in stations:
        col = f"adv_{station.lower()}_available"
        out[f"{station}_available_rate"] = float(frame[col].mean()) if col in frame.columns else 0.0
    return out


def _advection_regime(row: pd.Series) -> str:
    if bool(row.get("adv_any_frontal_passage_signal", False)):
        return "frontal_passage"
    if bool(row.get("adv_any_cold_advection_signal", False)):
        return "cold_advection"
    if bool(row.get("adv_any_warm_advection_signal", False)):
        return "warm_advection"
    if bool(row.get("adv_any_north_sector", False)):
        return "north_sector"
    if bool(row.get("adv_any_south_sector", False)):
        return "south_sector"
    return "neutral_or_missing"


def _row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _max_group_regression(grouped: pd.DataFrame, candidate: str, base: str, metric: str, key: str) -> float:
    candidate_rows = grouped[grouped["model_variant"] == candidate][[key, metric]]
    base_rows = grouped[grouped["model_variant"] == base][[key, metric]]
    merged = candidate_rows.merge(base_rows, on=key, suffixes=("_candidate", "_base"))
    if merged.empty:
        return 0.0
    return float((merged[f"{metric}_candidate"] - merged[f"{metric}_base"]).max())


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


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame, by_availability: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB wind/advection feature backtest",
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
            "## By Advection Regime",
            "",
            _table(by_regime),
            "",
            "## By Station Availability",
            "",
            _table(by_availability),
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest wind/advection METAR features for LFPB.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2_spatial.parquet")
    parser.add_argument("--output-dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2_spatial_advection.parquet")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--neighbor-station", action="append", default=list(DEFAULT_SPATIAL_STATIONS))
    parser.add_argument("--advection-station", action="append", default=list(DEFAULT_ADVECTION_STATIONS))
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--doc-path", default="docs/lfpb_wind_advection_backtest.md")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=70)
    parser.add_argument("--ml-weight", type=float, default=0.5)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
