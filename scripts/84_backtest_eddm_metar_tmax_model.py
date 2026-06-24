from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.features.metar_upside_dataset import build_metar_remaining_upside_dataset
from weather_tmax_bot.features.spatial_metar import EDDM_SPATIAL_STATIONS, add_spatial_metar_features_to_frame, spatial_feature_columns
from weather_tmax_bot.features.wind_advection import (
    EDDM_ADVECTION_STATIONS,
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
from weather_tmax_bot.models.model_registry import promote_model, register_artifact
from weather_tmax_bot.utils.hashing import stable_hash


AIRPORT = "EDDM"
TIMEZONE = "Europe/Berlin"
MODEL_VERSION = "eddm_metar_tmax_icon_d2_candidate_v1"
SPATIAL_MODEL_VERSION = "eddm_metar_tmax_icon_d2_spatial_v1"
WIND_ADVECTION_MODEL_VERSION = "eddm_metar_tmax_icon_d2_spatial_wind_advection_v1"
LOCAL_10_17_HOURS = {10, 12, 14, 16}

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
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    metar = pd.read_parquet(args.metar_path)
    nwp = pd.read_parquet(args.nwp_archive)
    target = _build_metar_tmax_target(metar)
    write_parquet(target, args.target_output)
    target = _restrict_target_to_nwp_overlap(target, nwp)

    dataset = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao=AIRPORT,
        timezone_name=TIMEZONE,
        local_issue_hours=args.local_issue_hours,
    )
    joined = _join_asof_nwp(dataset, nwp)
    if joined.empty:
        raise ValueError("No leakage-safe EDDM METAR Tmax rows joined to ICON-D2 NWP")
    write_parquet(joined, args.dataset_output)

    modeling_frame = joined
    spatial_enabled = bool(args.include_spatial)
    neighbor_stations = list(args.neighbor_station)
    missing_neighbors = [
        station
        for station in neighbor_stations
        if not (Path(args.neighbor_dir) / f"metar_iem_{station}.parquet").exists()
    ]
    if spatial_enabled and missing_neighbors:
        spatial_enabled = False
    if spatial_enabled:
        spatial_path = Path(args.spatial_dataset_output)
        if args.reuse_spatial_dataset and spatial_path.exists():
            candidate_frame = pd.read_parquet(spatial_path)
            if _same_modeling_keys(joined, candidate_frame):
                modeling_frame = candidate_frame
            else:
                modeling_frame = _build_spatial_frame(joined, args.neighbor_dir, neighbor_stations)
                write_parquet(modeling_frame, args.spatial_dataset_output)
        else:
            modeling_frame = _build_spatial_frame(joined, args.neighbor_dir, neighbor_stations)
            write_parquet(modeling_frame, args.spatial_dataset_output)

    wind_advection_enabled = bool(args.include_wind_advection) and spatial_enabled
    advection_stations = list(args.advection_station)
    missing_advection_stations = [
        station
        for station in advection_stations
        if not (Path(args.neighbor_dir) / f"metar_iem_{station}.parquet").exists()
    ]
    if wind_advection_enabled and missing_advection_stations:
        wind_advection_enabled = False
    if wind_advection_enabled:
        wind_path = Path(args.wind_advection_dataset_output)
        if args.reuse_wind_advection_dataset and wind_path.exists():
            candidate_frame = pd.read_parquet(wind_path)
            if _same_modeling_keys(modeling_frame, candidate_frame):
                modeling_frame = candidate_frame
            else:
                modeling_frame = _build_wind_advection_frame(modeling_frame, args.neighbor_dir, advection_stations)
                write_parquet(modeling_frame, args.wind_advection_dataset_output)
        else:
            modeling_frame = _build_wind_advection_frame(modeling_frame, args.neighbor_dir, advection_stations)
            write_parquet(modeling_frame, args.wind_advection_dataset_output)

    frame = prepare_metar_tmax_dataset(modeling_frame)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["model_tmax_c"].notna()].copy()
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(
        frame,
        min_train_rows=args.min_train_rows,
        min_calibration_rows=args.min_calibration_rows,
        min_test_rows=args.min_test_rows,
    )

    base_features = list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS)
    base_model = _fit_model(train, calibration, base_features, args.min_train_rows, args.max_iter)
    residuals_for_calibration = _residual_samples_by_hour(train)
    residuals_for_test = _residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True))
    base_weight = _optimize_ml_weight(calibration, base_model, residuals_for_calibration)
    base_ensemble = IconD2MetarTmaxEnsemble(base_model, residuals_for_test, base_weight, MODEL_VERSION)

    spatial_ensemble = None
    spatial_weight = None
    spatial_features = []
    wind_advection_ensemble = None
    wind_advection_weight = None
    wind_advection_features = []
    if spatial_enabled:
        spatial_features = base_features + spatial_feature_columns(neighbor_stations)
        spatial_model = _fit_model(train, calibration, spatial_features, args.min_train_rows, args.max_iter)
        spatial_weight = _optimize_ml_weight(calibration, spatial_model, residuals_for_calibration)
        spatial_ensemble = IconD2MetarTmaxEnsemble(
            spatial_model,
            residuals_for_test,
            spatial_weight,
            SPATIAL_MODEL_VERSION,
        )
    if wind_advection_enabled:
        wind_advection_features = spatial_features + wind_advection_feature_columns(advection_stations, target_station=AIRPORT)
        wind_advection_model = _fit_model(train, calibration, wind_advection_features, args.min_train_rows, args.max_iter)
        wind_advection_weight = _optimize_ml_weight(calibration, wind_advection_model, residuals_for_calibration)
        wind_advection_ensemble = IconD2MetarTmaxEnsemble(
            wind_advection_model,
            residuals_for_test,
            wind_advection_weight,
            WIND_ADVECTION_MODEL_VERSION,
        )

    current_production_model = joblib.load(args.current_production_model)
    scored = _score_holdout(
        test,
        current_production_model=current_production_model,
        base_ensemble=base_ensemble,
        spatial_ensemble=spatial_ensemble,
        wind_advection_ensemble=wind_advection_ensemble,
    )
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    by_hour_10_17 = _summary(
        scored[scored["local_issue_hour"].isin(LOCAL_10_17_HOURS)].copy(),
        ["model_variant"],
    )
    by_season = _summary(scored, ["model_variant", "season"])
    recommendation = _recommendation(summary, by_hour_10_17)
    production_artifact = _maybe_save_production_artifact(
        args,
        summary=summary,
        by_hour_10_17=by_hour_10_17,
        frame=frame,
        split=split,
        base_features=base_features,
        spatial_features=spatial_features,
        wind_advection_features=wind_advection_features,
        base_ensemble=base_ensemble,
        spatial_ensemble=spatial_ensemble,
        wind_advection_ensemble=wind_advection_ensemble,
        base_weight=base_weight,
        spatial_weight=spatial_weight,
        wind_advection_weight=wind_advection_weight,
        neighbor_stations=neighbor_stations,
        advection_stations=advection_stations,
    )

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": AIRPORT,
        "experiment": "EDDM METAR Tmax target backtest",
        "production_changed": bool(production_artifact.get("promoted")),
        "production_artifact": production_artifact,
        "target": "daily maximum temperature reported by EDDM METAR",
        "current_production_comparator": {
            "model_path": args.current_production_model,
            "target": "scored against EDDM METAR Tmax using the current configured production artifact",
        },
        "target_rows": int(len(target)),
        "dataset_rows": int(len(dataset)),
        "joined_rows": int(len(joined)),
        "usable_rows": int(len(frame)),
        "days": int(frame["target_date_local"].nunique()),
        "period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "base_feature_count": len(base_features),
        "spatial_enabled": spatial_enabled,
        "missing_neighbor_stations": missing_neighbors,
        "neighbor_stations": neighbor_stations,
        "spatial_feature_count": len(spatial_features) if spatial_enabled else 0,
        "wind_advection_enabled": wind_advection_enabled,
        "missing_advection_stations": missing_advection_stations,
        "advection_stations": advection_stations,
        "wind_advection_feature_count": len(wind_advection_features) if wind_advection_enabled else 0,
        "base_ml_weight": base_weight,
        "spatial_ml_weight": spatial_weight,
        "wind_advection_ml_weight": wind_advection_weight,
        "summary": json.loads(summary.to_json(orient="records")),
        "summary_10_17_local": json.loads(by_hour_10_17.to_json(orient="records")),
        "recommendation": recommendation,
        "limitations": [
            (
                "Offline replay only; registry and production forecast are not changed."
                if not production_artifact
                else "Production artifact was saved because --save-production-model was requested."
            ),
            "Current Munich production comparator is evaluated on the same METAR Tmax target.",
            "Issue-hour window 10-17 local maps to configured local issue hours 10, 12, 14, 16.",
            "TAF is not used in the EDDM METAR-target candidate, matching the current Paris METAR-target approach.",
        ],
    }

    scored.to_parquet(report_dir / "eddm_metar_tmax_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "eddm_metar_tmax_backtest_summary.csv", index=False)
    by_hour.to_csv(report_dir / "eddm_metar_tmax_backtest_by_hour.csv", index=False)
    by_hour_10_17.to_csv(report_dir / "eddm_metar_tmax_backtest_10_17_summary.csv", index=False)
    by_season.to_csv(report_dir / "eddm_metar_tmax_backtest_by_season.csv", index=False)
    (report_dir / "eddm_metar_tmax_backtest.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path(args.doc_path).write_text(_markdown(report, summary, by_hour, by_hour_10_17), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


def _maybe_save_production_artifact(
    args: argparse.Namespace,
    *,
    summary: pd.DataFrame,
    by_hour_10_17: pd.DataFrame,
    frame: pd.DataFrame,
    split: dict,
    base_features: list[str],
    spatial_features: list[str],
    wind_advection_features: list[str],
    base_ensemble: IconD2MetarTmaxEnsemble,
    spatial_ensemble: IconD2MetarTmaxEnsemble | None,
    wind_advection_ensemble: IconD2MetarTmaxEnsemble | None,
    base_weight: float,
    spatial_weight: float | None,
    wind_advection_weight: float | None,
    neighbor_stations: list[str],
    advection_stations: list[str],
) -> dict:
    if not args.save_production_model:
        return {}
    if args.production_variant == "spatial_wind_advection":
        if wind_advection_ensemble is None:
            raise ValueError("--production-variant spatial_wind_advection requires --include-wind-advection with available advection data")
        version = WIND_ADVECTION_MODEL_VERSION
        ensemble = wind_advection_ensemble
        feature_columns = wind_advection_features
        ml_weight = wind_advection_weight
        variant_name = "eddm_metar_icon_d2_spatial_wind_advection"
    elif args.production_variant == "spatial":
        if spatial_ensemble is None:
            raise ValueError("--production-variant spatial requires --include-spatial with available neighbor data")
        version = SPATIAL_MODEL_VERSION
        ensemble = spatial_ensemble
        feature_columns = spatial_features
        ml_weight = spatial_weight
        variant_name = "eddm_metar_icon_d2_spatial"
    else:
        version = MODEL_VERSION
        ensemble = base_ensemble
        feature_columns = base_features
        ml_weight = base_weight
        variant_name = "eddm_metar_icon_d2"

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{version}.joblib"
    metadata_path = model_dir / f"{version}.metadata.json"
    joblib.dump(ensemble, model_path)

    holdout_metrics = _row(summary, variant_name) or {}
    holdout_10_17_metrics = _row(by_hour_10_17, variant_name) or {}
    metadata = {
        "model_name": "eddm_metar_tmax_icon_d2_remaining_upside",
        "model_version": version,
        "airport": AIRPORT,
        "target": "daily maximum temperature reported by EDDM METAR",
        "role": "production_champion" if args.promote_production else "production_candidate",
        "training_source": "IEM METAR historical observations + Open-Meteo forecast-as-issued ICON-D2 single runs",
        "feature_set_version": (
            "eddm.metar_tmax.icon_d2.spatial_wind_advection.v1"
            if args.production_variant == "spatial_wind_advection"
            else "eddm.metar_tmax.icon_d2.spatial_metar.v1"
            if args.production_variant == "spatial"
            else "eddm.metar_tmax.icon_d2.intraday_enhanced.v1"
        ),
        "feature_columns": feature_columns,
        "enhanced_intraday_feature_columns": ENHANCED_INTRADAY_FEATURES,
        "nwp_feature_columns": NWP_COLUMNS,
        "spatial_feature_columns": spatial_feature_columns(neighbor_stations) if args.production_variant in {"spatial", "spatial_wind_advection"} else [],
        "wind_advection_feature_columns": (
            wind_advection_feature_columns(advection_stations, target_station=AIRPORT)
            if args.production_variant == "spatial_wind_advection"
            else []
        ),
        "neighbor_stations": neighbor_stations if args.production_variant in {"spatial", "spatial_wind_advection"} else [],
        "advection_stations": advection_stations if args.production_variant == "spatial_wind_advection" else [],
        "usable_rows": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "selected_ml_weight": ml_weight,
        "ensemble_metadata": ensemble.to_metadata(),
        "calibration_metadata": ensemble.ml_model.calibrator.to_metadata() if ensemble.ml_model.calibrator else {},
        "holdout_metrics": holdout_metrics,
        "holdout_10_17_local_metrics": holdout_10_17_metrics,
        "comparison_summary": json.loads(summary.to_json(orient="records")),
        "data_snapshot_hash": stable_hash(
            {
                "rows": len(frame),
                "target_sum": float(frame["final_metar_tmax_c"].sum()),
                "model_tmax_sum": float(frame["model_tmax_c"].sum()),
                "target_start": str(frame["target_date_local"].min()),
                "target_end": str(frame["target_date_local"].max()),
                "variant": args.production_variant,
                "advection_available_sum": float(frame.get("adv_available_station_count", pd.Series(dtype=float)).sum()),
            }
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "promotion_status": "production_artifact_promoted" if args.promote_production else "production_artifact_saved_not_promoted",
        "limitations": [
            "Target is EDDM METAR Tmax, not DWD official Tmax.",
            "TAF is not used.",
            "Candidate promotion should be based on METAR-target live monitoring, not only offline replay.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    register_artifact(
        version=version,
        artifact_type="model",
        path=model_path,
        metadata_path=metadata_path,
        metrics=holdout_metrics,
        model_dir=model_dir,
    )
    promoted = False
    if args.promote_production:
        promote_model(
            model_version=version,
            reason=f"manual_promote_eddm_metar_target_{args.production_variant}",
            metrics=holdout_metrics,
            model_dir=model_dir,
        )
        promoted = True
    return {
        "version": version,
        "variant": variant_name,
        "path": str(model_path),
        "metadata_path": str(metadata_path),
        "promoted": promoted,
    }


def _build_metar_tmax_target(metar: pd.DataFrame) -> pd.DataFrame:
    df = metar.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["knowledge_time_utc"] = pd.to_datetime(df.get("knowledge_time_utc", df["observation_time_utc"]), utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["observation_time_utc", "knowledge_time_utc", "temperature_c"]).copy()
    df["target_date_local"] = df["observation_time_utc"].dt.tz_convert(TIMEZONE).dt.date.astype(str)
    rows = []
    for day, group in df.groupby("target_date_local", sort=True):
        group = group.sort_values("observation_time_utc")
        max_idx = group["temperature_c"].idxmax()
        max_row = group.loc[max_idx]
        obs_count = int(len(group))
        rows.append(
            {
                "airport_icao": AIRPORT,
                "target_date_local": day,
                "timezone": TIMEZONE,
                "metar_tmax_c": float(max_row["temperature_c"]),
                "metar_tmax_time_utc": pd.Timestamp(max_row["observation_time_utc"]).isoformat(),
                "obs_count": obs_count,
                "quality_flags": "ok" if obs_count >= 8 else "low_coverage",
                "source_id": "iem.metar.archive.EDDM",
                "source_version": str(max_row.get("source_version", "iem.asos.csv")),
                "truth_data_release_time_utc": group["knowledge_time_utc"].max().isoformat(),
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def _build_spatial_frame(joined: pd.DataFrame, neighbor_dir: str, neighbor_stations: list[str]) -> pd.DataFrame:
    neighbor_metars = {
        station: pd.read_parquet(Path(neighbor_dir) / f"metar_iem_{station}.parquet")
        for station in neighbor_stations
    }
    return add_spatial_metar_features_to_frame(
        joined,
        neighbor_metars,
        timezone_name=TIMEZONE,
        stations=neighbor_stations,
    )


def _build_wind_advection_frame(frame: pd.DataFrame, neighbor_dir: str, advection_stations: list[str]) -> pd.DataFrame:
    station_metars = {
        station: pd.read_parquet(Path(neighbor_dir) / f"metar_iem_{station}.parquet")
        for station in advection_stations
    }
    return add_wind_advection_features_to_frame(
        frame,
        station_metars,
        timezone_name=TIMEZONE,
        stations=advection_stations,
        target_station=AIRPORT,
    )


def _same_modeling_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    key_columns = ["target_date_local", "issue_time_utc"]
    if any(column not in right.columns for column in key_columns):
        return False
    left_keys = left[key_columns].copy()
    right_keys = right[key_columns].copy()
    left_keys["target_date_local"] = left_keys["target_date_local"].astype(str)
    right_keys["target_date_local"] = right_keys["target_date_local"].astype(str)
    left_keys["issue_time_utc"] = pd.to_datetime(left_keys["issue_time_utc"], utc=True, errors="coerce")
    right_keys["issue_time_utc"] = pd.to_datetime(right_keys["issue_time_utc"], utc=True, errors="coerce")
    left_keys = left_keys.sort_values(key_columns).reset_index(drop=True)
    right_keys = right_keys.sort_values(key_columns).reset_index(drop=True)
    return left_keys.equals(right_keys)


def _restrict_target_to_nwp_overlap(target: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    nw = nwp.copy()
    nw = nw[nw["model_tmax_c"].notna()].copy()
    dates = set(nw["target_date_local"].astype(str).unique().tolist())
    return target[target["target_date_local"].astype(str).isin(dates)].copy()


def _join_asof_nwp(dataset: pd.DataFrame, nwp: pd.DataFrame) -> pd.DataFrame:
    ds = dataset.copy()
    nw = nwp.copy()
    ds["target_date_local"] = ds["target_date_local"].astype(str)
    nw["target_date_local"] = nw["target_date_local"].astype(str)
    ds["issue_time_utc"] = pd.to_datetime(ds["issue_time_utc"], utc=True)
    nw["knowledge_time_utc"] = pd.to_datetime(nw["knowledge_time_utc"], utc=True)
    nw["model_run_time_utc"] = pd.to_datetime(nw["model_run_time_utc"], utc=True)
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
            merged[column] = latest.get(column, np.nan)
        merged["nwp_model_minus_current_max_c"] = float(latest["model_tmax_c"]) - float(row["current_metar_max_c"])
        future = latest.get("model_future_temp_max_c", np.nan)
        merged["nwp_future_minus_current_max_c"] = (
            np.nan if pd.isna(future) else float(future) - float(row["current_metar_max_c"])
        )
        merged["nwp_knowledge_time_utc"] = latest["knowledge_time_utc"].isoformat()
        merged["nwp_model_run_time_utc"] = latest["model_run_time_utc"].isoformat()
        merged["nwp_source_id"] = latest["source_id"]
        merged["max_feature_knowledge_time_utc"] = max(
            pd.Timestamp(row["max_feature_knowledge_time_utc"]),
            latest["knowledge_time_utc"],
        ).isoformat()
        merged["leakage_check_passed"] = bool(
            pd.Timestamp(merged["max_feature_knowledge_time_utc"]) <= row["issue_time_utc"]
        )
        rows.append(merged)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out[out["leakage_check_passed"].fillna(False).astype(bool)].reset_index(drop=True)


def _fit_model(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    min_rows: int,
    max_iter: int,
) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(
        min_rows=min_rows,
        max_iter=max_iter,
        feature_columns=feature_columns,
    ).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(
        _survival_calibration_rows(model, calibration)
    )
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


def _residual_samples_by_hour(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    df = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    df["residual_c"] = df["final_metar_tmax_c"].astype(float) - df["model_tmax_c"].astype(float)
    residuals = {-1: df["residual_c"].to_numpy(dtype=float)}
    for hour, group in df.groupby("local_issue_hour"):
        if len(group) >= 20:
            residuals[int(hour)] = group["residual_c"].to_numpy(dtype=float)
    return residuals


def _optimize_ml_weight(
    calibration: pd.DataFrame,
    model: MetarTmaxUpsideModel,
    residuals: dict[int, np.ndarray],
) -> float:
    calibration_ensemble = IconD2MetarTmaxEnsemble(model, residuals, 0.0, "calibration")
    cached = [
        (calibration_ensemble.residual_distribution(row), model.predict_distribution(row), float(row["final_metar_tmax_c"]))
        for _, row in calibration.iterrows()
    ]
    best_weight = 0.0
    best_nll = np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        score = float(np.mean([_nll_mixed(residual_dist, ml_dist, actual, weight) for residual_dist, ml_dist, actual in cached]))
        if score < best_nll:
            best_nll = score
            best_weight = float(weight)
    return best_weight


def _nll_mixed(left: TmaxDistribution, right: TmaxDistribution, actual: float, right_weight: float) -> float:
    weight = float(np.clip(right_weight, 0.0, 1.0))
    left_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(left.bins_c, left.probabilities)}
    right_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(right.bins_c, right.probabilities)}
    actual_bin = int(round(actual))
    probability = (1.0 - weight) * left_lookup.get(actual_bin, 0.0) + weight * right_lookup.get(actual_bin, 0.0)
    return float(-np.log(max(probability, 1e-12)))


def _score_holdout(
    test: pd.DataFrame,
    *,
    current_production_model,
    base_ensemble: IconD2MetarTmaxEnsemble,
    spatial_ensemble: IconD2MetarTmaxEnsemble | None,
    wind_advection_ensemble: IconD2MetarTmaxEnsemble | None,
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        actual = float(row["final_metar_tmax_c"])
        variants = {
            "current_munich_production_core_on_metar": _predict_current_production(current_production_model, row),
            "eddm_metar_icon_d2": base_ensemble.predict_distribution(row),
        }
        if spatial_ensemble is not None:
            variants["eddm_metar_icon_d2_spatial"] = spatial_ensemble.predict_distribution(row)
        if wind_advection_ensemble is not None:
            variants["eddm_metar_icon_d2_spatial_wind_advection"] = wind_advection_ensemble.predict_distribution(row)
        for name, dist in variants.items():
            rows.append(_score_row(row, name, dist, actual))
    return pd.DataFrame(rows)


def _predict_current_production(model, row: pd.Series) -> TmaxDistribution:
    feature_row = row.to_dict()
    if hasattr(model, "ml_model") and hasattr(model, "residuals_by_hour"):
        return model.predict_distribution(feature_row)
    feature_row["month"] = pd.Timestamp(str(row["target_date_local"])).month
    feature_row["issue_hour_utc"] = pd.Timestamp(row["issue_time_utc"]).hour
    feature_row["nwp_missing"] = False
    observed = row.get("current_metar_max_c")
    return model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed)


def _score_row(row: pd.Series, variant: str, dist: TmaxDistribution, actual: float) -> dict:
    current_max = float(row["current_metar_max_c"])
    return {
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": row.get("season"),
        "model_variant": variant,
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "model_tmax_c": float(row["model_tmax_c"]),
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current_max + 1))), bool(actual - current_max >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current_max + 2))), bool(actual - current_max >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current_max + 3))), bool(actual - current_max >= 3.0)),
        "probabilities_by_integer_c": dist.to_payload()["probabilities_by_integer_c"],
    }


def _summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for key, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = {column: value for column, value in zip(group_cols, key, strict=True)}
        row.update(
            {
                "rows": int(len(group)),
                "distinct_days": int(group["target_date_local"].nunique()),
                "mae_expected": mae(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "rmse_expected": rmse(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "bias_expected": float(group["bias_expected"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _recommendation(summary: pd.DataFrame, summary_10_17: pd.DataFrame) -> dict:
    best_overall = _best_variant(summary)
    best_10_17 = _best_variant(summary_10_17)
    best_metar_overall = _best_variant(summary, exclude_variants={"current_munich_production_core_on_metar"})
    best_metar_10_17 = _best_variant(summary_10_17, exclude_variants={"current_munich_production_core_on_metar"})
    current_overall = _row(summary, "current_munich_production_core_on_metar")
    current_10_17 = _row(summary_10_17, "current_munich_production_core_on_metar")
    candidate = best_metar_overall
    if current_overall is None or current_10_17 is None or candidate is None or best_10_17 is None:
        return {"decision": "insufficient_comparison", "reason": "missing current or candidate row"}
    candidate_10_17 = _row(summary_10_17, str(candidate["model_variant"]))
    if candidate_10_17 is None:
        return {"decision": "insufficient_comparison", "reason": "missing 10-17 candidate row"}
    mae_delta = float(candidate["mae_expected"]) - float(current_overall["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(current_overall["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(current_overall["mean_crps"])
    mae_delta_10_17 = float(candidate_10_17["mae_expected"]) - float(current_10_17["mae_expected"])
    nll_delta_10_17 = float(candidate_10_17["mean_nll"]) - float(current_10_17["mean_nll"])
    crps_delta_10_17 = float(candidate_10_17["mean_crps"]) - float(current_10_17["mean_crps"])
    checks = {
        "candidate_is_not_current_production": candidate["model_variant"] != "current_munich_production_core_on_metar",
        "full_day_mae_improves_at_least_0_05c": mae_delta <= -0.05,
        "full_day_nll_not_worse": nll_delta <= 0.0,
        "full_day_crps_not_worse_by_0_003": crps_delta <= 0.003,
        "ten_to_seventeen_mae_not_worse_by_0_02c": mae_delta_10_17 <= 0.02,
        "enough_full_day_rows": int(candidate["rows"]) >= 300,
        "enough_10_17_rows": int(candidate_10_17["rows"]) >= 120,
    }
    decision = "promote_metar_target_candidate_to_shadow" if all(checks.values()) else "keep_as_research_candidate"
    return {
        "decision": decision,
        "best_overall_variant": None if best_overall is None else best_overall["model_variant"],
        "best_10_17_variant": best_10_17["model_variant"],
        "best_metar_overall_variant": candidate["model_variant"],
        "best_metar_10_17_variant": None if best_metar_10_17 is None else best_metar_10_17["model_variant"],
        "current_full_day_mae": float(current_overall["mae_expected"]),
        "candidate_full_day_mae": float(candidate["mae_expected"]),
        "candidate_minus_current_mae_full_day": mae_delta,
        "candidate_minus_current_nll_full_day": nll_delta,
        "candidate_minus_current_crps_full_day": crps_delta,
        "current_10_17_mae": float(current_10_17["mae_expected"]),
        "candidate_10_17_mae": float(candidate_10_17["mae_expected"]),
        "candidate_minus_current_mae_10_17": mae_delta_10_17,
        "candidate_minus_current_nll_10_17": nll_delta_10_17,
        "candidate_minus_current_crps_10_17": crps_delta_10_17,
        "checks": checks,
    }


def _best_variant(summary: pd.DataFrame, exclude_variants: set[str] | None = None) -> dict | None:
    if summary.empty:
        return None
    frame = summary.copy()
    if exclude_variants:
        frame = frame[~frame["model_variant"].isin(exclude_variants)].copy()
    if frame.empty:
        return None
    ordered = frame.sort_values(["mae_expected", "mean_nll", "mean_crps"]).reset_index(drop=True)
    return ordered.iloc[0].to_dict()


def _row(summary: pd.DataFrame, variant: str) -> dict | None:
    rows = summary[summary["model_variant"] == variant]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _time_split(
    frame: pd.DataFrame,
    *,
    min_train_rows: int,
    min_calibration_rows: int,
    min_test_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(frame["target_date_local"].unique())
    train_end_idx = max(1, int(len(dates) * 0.60))
    calibration_end_idx = max(train_end_idx + 1, int(len(dates) * 0.80))
    train = frame[frame["target_date_local"].isin(dates[:train_end_idx])].copy()
    calibration = frame[frame["target_date_local"].isin(dates[train_end_idx:calibration_end_idx])].copy()
    test = frame[frame["target_date_local"].isin(dates[calibration_end_idx:])].copy()
    while len(train) < min_train_rows and calibration_end_idx < len(dates) - 1:
        train_end_idx += 1
        calibration_end_idx += 1
        train = frame[frame["target_date_local"].isin(dates[:train_end_idx])].copy()
        calibration = frame[frame["target_date_local"].isin(dates[train_end_idx:calibration_end_idx])].copy()
        test = frame[frame["target_date_local"].isin(dates[calibration_end_idx:])].copy()
    if len(calibration) < min_calibration_rows or len(test) < min_test_rows:
        raise ValueError(
            f"Insufficient split rows: train={len(train)}, calibration={len(calibration)}, test={len(test)}"
        )
    return train, calibration, test, {
        "method": "chronological_60_20_20_by_target_day",
        "train_start": str(train["target_date_local"].min()),
        "train_end": str(train["target_date_local"].max()),
        "calibration_start": str(calibration["target_date_local"].min()),
        "calibration_end": str(calibration["target_date_local"].max()),
        "test_start": str(test["target_date_local"].min()),
        "test_end": str(test["target_date_local"].max()),
        "train_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "test_rows": int(len(test)),
        "train_days": int(train["target_date_local"].nunique()),
        "calibration_days": int(calibration["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
    }


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_hour_10_17: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# EDDM METAR Tmax target backtest",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- production changed: `{report['production_changed']}`",
            f"- target: {report['target']}",
            f"- period: `{report['period'][0]}` to `{report['period'][1]}`",
            f"- rows: `{report['usable_rows']}`",
            f"- days: `{report['days']}`",
            f"- spatial enabled: `{report['spatial_enabled']}`",
            f"- wind/advection enabled: `{report['wind_advection_enabled']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## 10-17 Local Summary",
            "",
            _table(by_hour_10_17),
            "",
            "## By Local Issue Hour",
            "",
            _table(by_hour),
            "",
        ]
    )


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if pd.isna(value):
        return ""
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest an EDDM METAR Tmax target model.")
    parser.add_argument("--metar-path", default="data/interim/metar_iem_EDDM.parquet")
    parser.add_argument("--nwp-archive", default="data/forecasts/open_meteo_single_runs_icon_d2.parquet")
    parser.add_argument("--current-production-model", default="data/models/eddm_metar_tmax_icon_d2_spatial_v1.joblib")
    parser.add_argument("--neighbor-dir", default="data/interim")
    parser.add_argument("--neighbor-station", nargs="*", default=EDDM_SPATIAL_STATIONS)
    parser.add_argument("--include-spatial", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-spatial-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advection-station", nargs="*", default=EDDM_ADVECTION_STATIONS)
    parser.add_argument("--include-wind-advection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-wind-advection-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--local-issue-hours", nargs="*", type=int, default=[6, 8, 10, 12, 14, 16, 18, 20])
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-calibration-rows", type=int, default=120)
    parser.add_argument("--min-test-rows", type=int, default=120)
    parser.add_argument("--max-iter", type=int, default=70)
    parser.add_argument("--save-production-model", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--promote-production", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--production-variant", choices=["base", "spatial", "spatial_wind_advection"], default="spatial_wind_advection")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--target-output", default="data/processed/metar_tmax_target_EDDM.parquet")
    parser.add_argument("--dataset-output", default="data/processed/metar_upside_dataset_EDDM_icon_d2.parquet")
    parser.add_argument("--spatial-dataset-output", default="data/processed/metar_upside_dataset_EDDM_icon_d2_spatial.parquet")
    parser.add_argument("--wind-advection-dataset-output", default="data/processed/metar_upside_dataset_EDDM_icon_d2_spatial_wind_advection.parquet")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--doc-path", default="docs/eddm_metar_tmax_backtest.md")
    return parser.parse_args()


if __name__ == "__main__":
    main()
