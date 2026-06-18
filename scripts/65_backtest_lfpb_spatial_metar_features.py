from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)
from weather_tmax_bot.utils.time import local_day_bounds_utc


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

NEIGHBOR_STATIONS = ["LFPG", "LFPO"]


def main() -> None:
    args = _parse_args()
    base = pd.read_parquet(args.dataset)
    neighbors = {station: pd.read_parquet(Path(args.neighbor_dir) / f"metar_iem_{station}.parquet") for station in args.neighbor_station}
    spatial_dataset = add_spatial_metar_features(base, neighbors, timezone_name=args.timezone)
    output_dataset = Path(args.output_dataset)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    spatial_dataset.to_parquet(output_dataset, index=False)

    frame = prepare_metar_tmax_dataset(spatial_dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["model_tmax_c"].notna()].copy()
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)

    base_features = list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS)
    spatial_features = base_features + _spatial_feature_columns(args.neighbor_station)

    base_model = _fit_model(train, calibration, base_features, args.min_train_rows, args.max_iter)
    spatial_model = _fit_model(train, calibration, spatial_features, args.min_train_rows, args.max_iter)

    residuals_for_calibration = _residual_samples_by_hour(train)
    residuals_for_test = _residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True))
    base_weight = _optimize_ml_weight(calibration, base_model, residuals_for_calibration)
    spatial_weight = _optimize_ml_weight(calibration, spatial_model, residuals_for_calibration)
    base_ensemble = IconD2MetarTmaxEnsemble(base_model, residuals_for_test, base_weight, "lfpb_spatial_backtest_base")
    spatial_ensemble = IconD2MetarTmaxEnsemble(spatial_model, residuals_for_test, spatial_weight, "lfpb_spatial_backtest_candidate")

    scored = _score_holdout(test, base_ensemble, spatial_ensemble)
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _summary(scored, ["model_variant", "season"])
    by_neighbor_availability = _summary(scored, ["model_variant", "spatial_available_station_count"])
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "experiment": "offline spatial METAR context backtest",
        "dataset": args.dataset,
        "neighbor_stations": list(args.neighbor_station),
        "rows": len(frame),
        "days": int(frame["target_date_local"].nunique()),
        "period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "base_feature_count": len(base_features),
        "spatial_feature_count": len(spatial_features),
        "spatial_feature_columns": _spatial_feature_columns(args.neighbor_station),
        "base_ml_weight": base_weight,
        "spatial_ml_weight": spatial_weight,
        "neighbor_coverage": _neighbor_coverage(frame, args.neighbor_station),
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(summary, by_hour, by_neighbor_availability),
        "limitations": [
            "Offline replay only; production artifacts are not changed.",
            "Neighbor METAR records are historical IEM archive records and are used only when knowledge_time_utc <= issue_time_utc.",
            "Spatial features use neighbor observations as context, not as the LFPB target.",
        ],
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_spatial_metar_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_spatial_metar_backtest_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_spatial_metar_backtest_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_spatial_metar_backtest_by_season.csv", index=False)
    by_neighbor_availability.to_csv(report_dir / "lfpb_spatial_metar_backtest_by_neighbor_availability.csv", index=False)
    (report_dir / "lfpb_spatial_metar_backtest.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path(args.doc_path).write_text(_markdown(report, summary, by_hour, by_neighbor_availability), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


def add_spatial_metar_features(
    base: pd.DataFrame,
    neighbors: dict[str, pd.DataFrame],
    *,
    timezone_name: str,
) -> pd.DataFrame:
    out = base.copy()
    prepared = {station: _prepare_neighbor_metar(frame) for station, frame in neighbors.items()}
    rows = []
    for _, row in out.iterrows():
        issue = pd.Timestamp(row["issue_time_utc"]).tz_convert("UTC")
        target_date = pd.Timestamp(str(row["target_date_local"])).date()
        features: dict[str, float | int | bool] = {}
        latest_values = []
        max_values = []
        available = 0
        max_knowledge_time = pd.Timestamp(row.get("max_feature_knowledge_time_utc", row["issue_time_utc"])).tz_convert("UTC")
        for station, frame in prepared.items():
            station_features, station_latest, station_max, station_max_knowledge = _neighbor_features_for_row(
                frame,
                issue_time_utc=issue,
                target_date_local=target_date,
                timezone_name=timezone_name,
                prefix=f"spatial_{station.lower()}",
            )
            features.update(station_features)
            if station_latest is not None:
                latest_values.append(station_latest)
                available += 1
            if station_max is not None:
                max_values.append(station_max)
            if station_max_knowledge is not None:
                max_knowledge_time = max(max_knowledge_time, station_max_knowledge)
        lfpb_latest = _optional_float(row.get("latest_metar_temp_c"))
        lfpb_current_max = _optional_float(row.get("current_metar_max_c"))
        features["spatial_available_station_count"] = available
        features["spatial_latest_temp_mean_c"] = _mean_or_nan(latest_values)
        features["spatial_latest_temp_max_c"] = _max_or_nan(latest_values)
        features["spatial_latest_temp_min_c"] = _min_or_nan(latest_values)
        features["spatial_latest_temp_spread_c"] = _spread_or_nan(latest_values)
        features["spatial_current_max_mean_c"] = _mean_or_nan(max_values)
        features["spatial_current_max_max_c"] = _max_or_nan(max_values)
        features["spatial_current_max_min_c"] = _min_or_nan(max_values)
        features["spatial_current_max_spread_c"] = _spread_or_nan(max_values)
        features["spatial_latest_minus_lfpb_latest_mean_c"] = _maybe_diff(features["spatial_latest_temp_mean_c"], lfpb_latest)
        features["spatial_max_minus_lfpb_current_max_mean_c"] = _maybe_diff(features["spatial_current_max_mean_c"], lfpb_current_max)
        features["spatial_any_neighbor_above_lfpb_latest"] = bool(
            lfpb_latest is not None and any(value > lfpb_latest for value in latest_values)
        )
        features["spatial_any_neighbor_above_lfpb_current_max"] = bool(
            lfpb_current_max is not None and any(value > lfpb_current_max for value in max_values)
        )
        features["spatial_max_feature_knowledge_time_utc"] = max_knowledge_time.isoformat()
        features["spatial_leakage_check_passed"] = bool(max_knowledge_time <= issue)
        rows.append(features)
    spatial = pd.DataFrame(rows, index=out.index)
    for column in spatial.columns:
        out[column] = spatial[column]
    out["max_feature_knowledge_time_utc"] = out["spatial_max_feature_knowledge_time_utc"]
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool) & out["spatial_leakage_check_passed"].fillna(False).astype(bool)
    return out


def _neighbor_features_for_row(
    frame: pd.DataFrame,
    *,
    issue_time_utc: pd.Timestamp,
    target_date_local,
    timezone_name: str,
    prefix: str,
) -> tuple[dict, float | None, float | None, pd.Timestamp | None]:
    columns = {
        f"{prefix}_available": False,
        f"{prefix}_latest_temp_c": np.nan,
        f"{prefix}_current_max_c": np.nan,
        f"{prefix}_latest_minus_lfpb_latest_c": np.nan,
        f"{prefix}_max_minus_lfpb_current_max_c": np.nan,
        f"{prefix}_drop_from_current_max_c": np.nan,
        f"{prefix}_temp_trend_1h": np.nan,
        f"{prefix}_temp_trend_3h": np.nan,
        f"{prefix}_temp_trend_last_2_metars": np.nan,
        f"{prefix}_dewpoint_depression_latest": np.nan,
        f"{prefix}_cloud_cover_proxy_latest": np.nan,
        f"{prefix}_has_rain_recent": False,
        f"{prefix}_has_thunder_recent": False,
        f"{prefix}_is_cavok_latest": False,
        f"{prefix}_age_minutes": np.nan,
        f"{prefix}_count_so_far": 0,
        f"{prefix}_count_last_1h": 0,
        f"{prefix}_count_last_3h": 0,
    }
    if frame.empty:
        return columns, None, None, None
    day_start, day_end = local_day_bounds_utc(target_date_local, timezone_name)
    so_far = frame[
        (frame["observation_time_utc"] >= pd.Timestamp(day_start))
        & (frame["observation_time_utc"] < pd.Timestamp(day_end))
        & (frame["knowledge_time_utc"] <= issue_time_utc)
    ].copy()
    if so_far.empty:
        return columns, None, None, None
    latest = so_far.iloc[-1]
    latest_temp = float(latest["temperature_c"])
    current_max = float(so_far["temperature_c"].max())
    columns.update(
        {
            f"{prefix}_available": True,
            f"{prefix}_latest_temp_c": latest_temp,
            f"{prefix}_current_max_c": current_max,
            f"{prefix}_drop_from_current_max_c": float(current_max - latest_temp),
            f"{prefix}_temp_trend_1h": _trend(_window(so_far, issue_time_utc, 1), "temperature_c"),
            f"{prefix}_temp_trend_3h": _trend(_window(so_far, issue_time_utc, 3), "temperature_c"),
            f"{prefix}_temp_trend_last_2_metars": _trend(so_far.tail(2), "temperature_c"),
            f"{prefix}_dewpoint_depression_latest": _finite_or_nan(latest_temp - _finite_or_nan(latest.get("dewpoint_c"))),
            f"{prefix}_cloud_cover_proxy_latest": _cloud_cover_proxy(latest),
            f"{prefix}_has_rain_recent": _has_weather(_window(so_far, issue_time_utc, 3), ["RA", "SHRA", "TSRA"]),
            f"{prefix}_has_thunder_recent": _has_weather(_window(so_far, issue_time_utc, 6), ["TS"]),
            f"{prefix}_is_cavok_latest": bool(latest.get("cavok", False)),
            f"{prefix}_age_minutes": float((issue_time_utc - latest["knowledge_time_utc"]).total_seconds() / 60.0),
            f"{prefix}_count_so_far": int(len(so_far)),
            f"{prefix}_count_last_1h": int(_window(so_far, issue_time_utc, 1).shape[0]),
            f"{prefix}_count_last_3h": int(_window(so_far, issue_time_utc, 3).shape[0]),
        }
    )
    return columns, latest_temp, current_max, so_far["knowledge_time_utc"].max()


def _prepare_neighbor_metar(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["knowledge_time_utc"] = pd.to_datetime(df.get("knowledge_time_utc", df["observation_time_utc"]), utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    for column in ["dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt", "ceiling_ft"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["observation_time_utc", "knowledge_time_utc", "temperature_c"]).sort_values("observation_time_utc")


def _fit_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    min_train_rows: int,
    max_iter: int,
) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(min_rows=min_train_rows, max_iter=max_iter, feature_columns=feature_columns).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    return model


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
    cached = []
    for _, row in calibration.iterrows():
        residual_dist = calibration_ensemble.residual_distribution(row)
        ml_dist = model.predict_distribution(row)
        actual = float(row["final_metar_tmax_c"])
        cached.append((residual_dist, ml_dist, actual))
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        losses = [nll_integer_bin(_mix(residual_dist, ml_dist, weight), actual) for residual_dist, ml_dist, actual in cached]
        score = float(np.mean(losses))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _score_holdout(test: pd.DataFrame, base_ensemble: IconD2MetarTmaxEnsemble, spatial_ensemble: IconD2MetarTmaxEnsemble) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("production_like_icon_d2", row, base_ensemble.predict_distribution(row)))
        rows.append(_score("spatial_metar_icon_d2", row, spatial_ensemble.predict_distribution(row)))
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": str(row.get("season", "unknown")),
        "spatial_available_station_count": int(row.get("spatial_available_station_count", 0)),
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


def _spatial_feature_columns(stations: list[str]) -> list[str]:
    columns = []
    for station in stations:
        prefix = f"spatial_{station.lower()}"
        columns.extend(
            [
                f"{prefix}_available",
                f"{prefix}_latest_temp_c",
                f"{prefix}_current_max_c",
                f"{prefix}_drop_from_current_max_c",
                f"{prefix}_temp_trend_1h",
                f"{prefix}_temp_trend_3h",
                f"{prefix}_temp_trend_last_2_metars",
                f"{prefix}_dewpoint_depression_latest",
                f"{prefix}_cloud_cover_proxy_latest",
                f"{prefix}_has_rain_recent",
                f"{prefix}_has_thunder_recent",
                f"{prefix}_is_cavok_latest",
                f"{prefix}_age_minutes",
                f"{prefix}_count_so_far",
                f"{prefix}_count_last_1h",
                f"{prefix}_count_last_3h",
            ]
        )
    columns.extend(
        [
            "spatial_available_station_count",
            "spatial_latest_temp_mean_c",
            "spatial_latest_temp_max_c",
            "spatial_latest_temp_min_c",
            "spatial_latest_temp_spread_c",
            "spatial_current_max_mean_c",
            "spatial_current_max_max_c",
            "spatial_current_max_min_c",
            "spatial_current_max_spread_c",
            "spatial_latest_minus_lfpb_latest_mean_c",
            "spatial_max_minus_lfpb_current_max_mean_c",
            "spatial_any_neighbor_above_lfpb_latest",
            "spatial_any_neighbor_above_lfpb_current_max",
        ]
    )
    return columns


def _neighbor_coverage(frame: pd.DataFrame, stations: list[str]) -> dict:
    out = {
        "rows": len(frame),
        "any_neighbor_available_rate": float((frame["spatial_available_station_count"] > 0).mean()),
        "all_neighbors_available_rate": float((frame["spatial_available_station_count"] >= len(stations)).mean()),
    }
    for station in stations:
        out[f"{station}_available_rate"] = float(frame[f"spatial_{station.lower()}_available"].mean())
        out[f"{station}_median_age_minutes"] = float(pd.to_numeric(frame[f"spatial_{station.lower()}_age_minutes"], errors="coerce").median())
    return out


def _recommendation(summary: pd.DataFrame, by_hour: pd.DataFrame, by_availability: pd.DataFrame) -> dict:
    base = _variant_row(summary, "production_like_icon_d2")
    spatial = _variant_row(summary, "spatial_metar_icon_d2")
    mae_delta = float(spatial["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(spatial["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(spatial["mean_crps"]) - float(base["mean_crps"])
    max_hour_nll_regression = _max_group_regression(by_hour, "spatial_metar_icon_d2", "mean_nll", "local_issue_hour")
    max_availability_nll_regression = _max_group_regression(
        by_availability,
        "spatial_metar_icon_d2",
        "mean_nll",
        "spatial_available_station_count",
    )
    promote = mae_delta <= -0.02 and nll_delta <= 0.02 and crps_delta <= 0.005 and max_hour_nll_regression <= 0.20
    return {
        "decision": "candidate_for_shadow" if promote else "do_not_promote_yet",
        "mae_delta_spatial_minus_base": mae_delta,
        "nll_delta_spatial_minus_base": nll_delta,
        "crps_delta_spatial_minus_base": crps_delta,
        "max_hour_nll_regression": max_hour_nll_regression,
        "max_availability_nll_regression": max_availability_nll_regression,
        "reason": (
            "Spatial METAR features improved point accuracy without meaningful probabilistic regression."
            if promote
            else "Spatial METAR features did not pass the offline promotion gate."
        ),
    }


def _max_group_regression(frame: pd.DataFrame, variant: str, metric: str, group_column: str) -> float:
    base = frame[frame["model_variant"] == "production_like_icon_d2"][[group_column, metric]]
    cand = frame[frame["model_variant"] == variant][[group_column, metric]]
    merged = base.merge(cand, on=group_column, suffixes=("_base", "_candidate"))
    if merged.empty:
        return 0.0
    return float((merged[f"{metric}_candidate"] - merged[f"{metric}_base"]).max())


def _variant_row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _mix(left: TmaxDistribution, right: TmaxDistribution, right_weight: float) -> TmaxDistribution:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    bins = np.arange(min(left.bins_c.min(), right.bins_c.min()), max(left.bins_c.max(), right.bins_c.max()) + 1)
    left_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(left.bins_c, left.probabilities)}
    right_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(right.bins_c, right.probabilities)}
    probs = np.array([(1.0 - weight) * left_lookup.get(int(bin_c), 0.0) + weight * right_lookup.get(int(bin_c), 0.0) for bin_c in bins])
    return TmaxDistribution(bins, probs)


def _window(df: pd.DataFrame, issue_utc: pd.Timestamp, hours: float) -> pd.DataFrame:
    return df[df["observation_time_utc"] >= issue_utc - pd.Timedelta(hours=hours)]


def _trend(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.iloc[-1] - values.iloc[0])


def _has_weather(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    return any(code in text for code in codes)


def _cloud_cover_proxy(row: pd.Series) -> float:
    if bool(row.get("cavok", False)):
        return 0.0
    text = " ".join(str(row.get(column, "") or "") for column in ["cloud_layers", "raw_metar"])
    if "OVC" in text:
        return 8.0
    if "BKN" in text:
        return 6.0
    if "SCT" in text:
        return 4.0
    if "FEW" in text:
        return 2.0
    if "NSC" in text or "SKC" in text or "CLR" in text:
        return 0.0
    return float("nan")


def _finite_or_nan(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _optional_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


def _min_or_nan(values: list[float]) -> float:
    return float(np.min(values)) if values else float("nan")


def _spread_or_nan(values: list[float]) -> float:
    return float(np.max(values) - np.min(values)) if len(values) >= 2 else float("nan")


def _maybe_diff(left, right: float | None) -> float:
    if right is None or pd.isna(left):
        return float("nan")
    return float(left) - float(right)


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


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_neighbor_availability: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB spatial METAR feature backtest",
            "",
            "Offline replay only. Production artifacts were not changed.",
            "",
            "## Summary",
            "",
            "```csv",
            summary.to_csv(index=False).strip(),
            "```",
            "",
            "## By hour",
            "",
            "```csv",
            by_hour.to_csv(index=False).strip(),
            "```",
            "",
            "## By neighbor availability",
            "",
            "```csv",
            by_neighbor_availability.to_csv(index=False).strip(),
            "```",
            "",
            "## Recommendation",
            "",
            "```json",
            json.dumps(report["recommendation"], indent=2, ensure_ascii=False),
            "```",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest spatial METAR context features for LFPB.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--neighbor-dir", default="data/interim")
    parser.add_argument("--neighbor-station", action="append", default=NEIGHBOR_STATIONS)
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--output-dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2_spatial.parquet")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--doc-path", default="docs/lfpb_spatial_metar_backtest.md")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    parser.add_argument("--max-iter", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    main()
