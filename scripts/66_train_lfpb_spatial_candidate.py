from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.features.spatial_metar import DEFAULT_SPATIAL_STATIONS, add_spatial_metar_features_to_frame, spatial_feature_columns
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)
from weather_tmax_bot.models.model_registry import register_artifact
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_metar_tmax_icon_d2_spatial_candidate_v1"

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
    args = _parse_args()
    dataset = _load_or_build_spatial_dataset(args)
    frame = prepare_metar_tmax_dataset(dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["model_tmax_c"].notna()].sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)
    features = list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS) + spatial_feature_columns(args.neighbor_station)

    model = MetarTmaxUpsideModel(min_rows=args.min_train_rows, max_iter=args.max_iter, feature_columns=features).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    residuals_cal = _residual_samples_by_hour(train)
    ml_weight = _optimize_ml_weight(calibration, model, residuals_cal)
    residuals = _residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True))
    ensemble = IconD2MetarTmaxEnsemble(
        ml_model=model,
        residuals_by_hour=residuals,
        ml_weight=ml_weight,
        model_version=MODEL_VERSION,
    )
    scored = _score_holdout(test, ensemble)
    summary = _summary(scored, ["model_variant"])
    candidate_metrics = summary.iloc[0].to_dict()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_VERSION}.joblib"
    metadata_path = model_dir / f"{MODEL_VERSION}.metadata.json"
    joblib.dump(ensemble, model_path)
    metadata = {
        "model_name": "lfpb_spatial_metar_icon_d2_shadow_candidate",
        "model_version": MODEL_VERSION,
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "role": "shadow_candidate",
        "active_local_hour_window": [12, 18],
        "neighbor_stations": list(args.neighbor_station),
        "feature_set_version": "lfpb.metar_tmax.icon_d2.spatial_metar_candidate.v1",
        "feature_columns": features,
        "spatial_feature_columns": spatial_feature_columns(args.neighbor_station),
        "usable_rows": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "selected_ml_weight": ml_weight,
        "calibration_metadata": calibrator.to_metadata(),
        "holdout_metrics": candidate_metrics,
        "data_snapshot_hash": stable_hash(
            {
                "rows": len(frame),
                "target_sum": float(frame["final_metar_tmax_c"].sum()),
                "model_tmax_sum": float(frame["model_tmax_c"].sum()),
                "spatial_available_sum": float(frame["spatial_available_station_count"].sum()),
            }
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "promotion_status": "shadow_candidate_only",
        "limitations": [
            "Used only as a parallel signal from 12:00 to 18:00 Europe/Paris.",
            "Not used for production forecast acceptance or forecast log outcome scoring.",
            "Backtest showed better NLL/coverage but worse expected-value MAE overall.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    register_artifact(
        version=MODEL_VERSION,
        artifact_type="model",
        path=model_path,
        metadata_path=metadata_path,
        metrics=candidate_metrics,
        model_dir=model_dir,
    )
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_spatial_candidate_holdout_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_spatial_candidate_holdout_summary.csv", index=False)
    (report_dir / "lfpb_spatial_candidate_training.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


def _load_or_build_spatial_dataset(args: argparse.Namespace) -> pd.DataFrame:
    output = Path(args.output_dataset)
    if output.exists() and not args.rebuild_dataset:
        return pd.read_parquet(output)
    base = pd.read_parquet(args.dataset)
    neighbors = {
        station: pd.read_parquet(Path(args.neighbor_dir) / f"metar_iem_{station}.parquet")
        for station in args.neighbor_station
    }
    spatial = add_spatial_metar_features_to_frame(base, neighbors, timezone_name=args.timezone, stations=args.neighbor_station)
    output.parent.mkdir(parents=True, exist_ok=True)
    spatial.to_parquet(output, index=False)
    return spatial


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


def _residual_samples_by_hour(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    data = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    data["residual"] = data["final_metar_tmax_c"].astype(float) - data["model_tmax_c"].astype(float)
    out = {-1: data["residual"].to_numpy(dtype=float)}
    for hour, group in data.groupby("local_issue_hour"):
        out[int(hour)] = group["residual"].to_numpy(dtype=float)
    return out


def _optimize_ml_weight(calibration: pd.DataFrame, model: MetarTmaxUpsideModel, residuals: dict[int, np.ndarray]) -> float:
    calibration_ensemble = IconD2MetarTmaxEnsemble(model, residuals, 0.0, "calibration")
    cached = [
        (calibration_ensemble.residual_distribution(row), model.predict_distribution(row), float(row["final_metar_tmax_c"]))
        for _, row in calibration.iterrows()
    ]
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        losses = [_nll_mixed(residual_dist, ml_dist, actual, weight) for residual_dist, ml_dist, actual in cached]
        score = float(np.mean(losses))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _nll_mixed(left, right, actual: float, right_weight: float) -> float:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    bins = np.arange(min(left.bins_c.min(), right.bins_c.min()), max(left.bins_c.max(), right.bins_c.max()) + 1)
    left_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(left.bins_c, left.probabilities)}
    right_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(right.bins_c, right.probabilities)}
    probability = 0.0
    actual_bin = int(round(actual))
    for bin_c in bins:
        if int(bin_c) == actual_bin:
            probability = (1.0 - weight) * left_lookup.get(actual_bin, 0.0) + weight * right_lookup.get(actual_bin, 0.0)
            break
    return float(-np.log(max(probability, 1e-12)))


def _score_holdout(test: pd.DataFrame, model: IconD2MetarTmaxEnsemble) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        dist = model.predict_distribution(row)
        actual = float(row["final_metar_tmax_c"])
        current_max = float(row["current_metar_max_c"])
        rows.append(
            {
                "model_variant": "spatial_metar_icon_d2",
                "target_date_local": str(row["target_date_local"]),
                "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
                "local_issue_hour": int(row["local_issue_hour"]),
                "season": str(row.get("season", "unknown")),
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
        )
    return pd.DataFrame(rows)


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


def _covered(dist, actual: float, mass: float) -> bool:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFPB spatial METAR shadow candidate.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--output-dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2_spatial.parquet")
    parser.add_argument("--neighbor-dir", default="data/interim")
    parser.add_argument("--neighbor-station", action="append", default=DEFAULT_SPATIAL_STATIONS)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--rebuild-dataset", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
