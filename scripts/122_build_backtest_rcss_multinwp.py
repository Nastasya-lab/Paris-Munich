from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from weather_tmax_bot.data.iem import IEMAdapter
from weather_tmax_bot.data.previous_runs import (
    build_previous_day1_snapshots,
    fetch_previous_day1_hourly,
)
from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.features.build_metar_target import build_daily_metar_tmax
from weather_tmax_bot.features.metar_upside_dataset import (
    ENHANCED_METAR_INTRADAY_FEATURES,
    build_metar_remaining_upside_dataset,
)
from weather_tmax_bot.features.spatial_metar import (
    add_spatial_metar_features_to_frame,
    spatial_feature_columns,
)
from weather_tmax_bot.features.wind_advection import (
    add_wind_advection_features_to_frame,
    wind_advection_feature_columns,
)
from weather_tmax_bot.models.distribution import (
    TmaxDistribution,
    project_unimodal_distribution,
    temperature_scale_distribution,
    unimodal_violation_count,
)
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)
from weather_tmax_bot.models.multinwp_tmax import (
    NWP_AGGREGATES,
    RCSS_NWP_MODELS,
    MultiNwpMetarTmaxModel,
    blend_nwp_features,
    prefixed_nwp_feature_columns,
)
from weather_tmax_bot.models.regression_tmax import RegressionResidualTmaxModel


AIRPORT = "RCSS"
TIMEZONE = "Asia/Taipei"
LATITUDE = 25.069722
LONGITUDE = 121.5525
NEIGHBORS = ["RCTP"]
ADVECTION_STATIONS = ["RCSS", "RCTP"]
ISSUE_HOURS = list(range(6, 21))
MODEL_VERSION = "rcss_metar_tmax_multinwp_d1_v1"
BASE_FEATURES = list(dict.fromkeys(DEFAULT_METAR_TMAX_FEATURES + list(ENHANCED_METAR_INTRADAY_FEATURES)))
PREFIXES = list(RCSS_NWP_MODELS.values())
AGGREGATE_FEATURES = [
    *[f"nwp_blend_{name}" for name in NWP_AGGREGATES],
    "nwp_tmax_spread_c",
    "nwp_tmax_std_c",
    "model_tmax_c",
    "model_future_temp_max_c",
    "nwp_model_minus_current_max_c",
    "nwp_future_minus_current_max_c",
]


def main() -> None:
    args = _parse_args()
    if args.command == "prepare-metar":
        _prepare_metar(args)
    elif args.command == "prepare-nwp":
        _prepare_nwp(args)
    elif args.command == "build-dataset":
        _build_dataset(args)
    elif args.command == "backtest":
        _backtest(args)
    elif args.command == "train":
        _train(args)
    else:
        _run_all(args)


def _run_all(args: argparse.Namespace) -> None:
    _prepare_metar(args)
    _prepare_nwp(args)
    _build_dataset(args)
    report = _backtest(args)
    _train(args, selected_variant=report["selection"]["selected_variant"])


def _prepare_metar(args: argparse.Namespace) -> None:
    output = Path(args.metar_dir)
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    for station in ADVECTION_STATIONS:
        path = output / f"metar_iem_{station}.parquet"
        if path.exists() and not args.force:
            print(json.dumps({"station": station, "status": "cached", "rows": len(pd.read_parquet(path))}))
            continue
        frame = _fetch_metar_with_retries(
            station,
            pd.Timestamp(args.start_date, tz="UTC").to_pydatetime(),
            end.tz_localize("UTC").to_pydatetime(),
        )
        frame.to_parquet(path, index=False)
        print(json.dumps({"station": station, "status": "downloaded", "rows": len(frame)}))
        time.sleep(2)


def _prepare_nwp(args: argparse.Namespace) -> None:
    cache = Path(args.nwp_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    availability = []
    for model, prefix in RCSS_NWP_MODELS.items():
        hourly = fetch_previous_day1_hourly(
            latitude=LATITUDE,
            longitude=LONGITUDE,
            model=model,
            start_date=args.start_date,
            end_date=args.end_date,
            cache_dir=cache / "chunks",
            chunk_days=args.chunk_days,
        )
        hourly_path = cache / f"rcss_{model}_previous_day1_hourly.parquet"
        hourly.to_parquet(hourly_path, index=False)
        snapshots = build_previous_day1_snapshots(
            hourly,
            timezone_name=TIMEZONE,
            issue_hours=ISSUE_HOURS,
            prefix=prefix,
        )
        snapshot_path = cache / f"rcss_{model}_previous_day1_snapshots.parquet"
        snapshots.to_parquet(snapshot_path, index=False)
        availability.append(
            {
                "model": model,
                "prefix": prefix,
                "hourly_rows": len(hourly),
                "valid_temperature_hours": int(pd.to_numeric(hourly.get("temperature_2m"), errors="coerce").notna().sum()),
                "snapshot_rows": len(snapshots),
                "snapshot_days": int(snapshots["target_date_local"].nunique()) if not snapshots.empty else 0,
                "first_day": None if snapshots.empty else snapshots["target_date_local"].min(),
                "last_day": None if snapshots.empty else snapshots["target_date_local"].max(),
            }
        )
        print(json.dumps(availability[-1], default=str))
    report = {"airport": AIRPORT, "period": [args.start_date, args.end_date], "models": availability}
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir, "rcss_multinwp_availability.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


def _build_dataset(args: argparse.Namespace) -> pd.DataFrame:
    metar_dir = Path(args.metar_dir)
    checkpoint_dir = Path(args.report_dir) / "dataset_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = checkpoint_dir / "base_metar_nwp.parquet"
    spatial_checkpoint = checkpoint_dir / "base_metar_nwp_spatial.parquet"
    metar = pd.read_parquet(metar_dir / "metar_iem_RCSS.parquet")
    target = build_daily_metar_tmax(
        metar,
        airport_icao=AIRPORT,
        timezone_name=TIMEZONE,
        source_id="iem.metar.archive.RCSS",
        expected_reports_per_day=48,
    )
    target.to_parquet(args.target_path, index=False)
    keys = ["target_date_local", "issue_time_utc", "local_issue_hour"]
    if base_checkpoint.exists() and not args.force:
        out = pd.read_parquet(base_checkpoint)
        print(json.dumps({"dataset_stage": "base_metar_nwp", "status": "cached", "rows": len(out)}))
    else:
        base = build_metar_remaining_upside_dataset(
            metar,
            target,
            airport_icao=AIRPORT,
            timezone_name=TIMEZONE,
            local_issue_hours=ISSUE_HOURS,
        )
        out = base.copy()
        out["target_date_local"] = out["target_date_local"].astype(str)
        out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True)
        common_days = set(out["target_date_local"])
        for model, prefix in RCSS_NWP_MODELS.items():
            nwp = pd.read_parquet(Path(args.nwp_cache_dir) / f"rcss_{model}_previous_day1_snapshots.parquet")
            nwp["target_date_local"] = nwp["target_date_local"].astype(str)
            nwp["issue_time_utc"] = pd.to_datetime(nwp["issue_time_utc"], utc=True)
            required = [f"{prefix}_tmax_c", f"{prefix}_future_temp_max_c"]
            nwp = nwp.dropna(subset=required)
            common_days &= set(nwp["target_date_local"])
            keep = keys + prefixed_nwp_feature_columns([prefix])
            out = out.merge(nwp[keep], on=keys, how="inner", validate="one_to_one")
        out = out[out["target_date_local"].isin(common_days)].copy()
        out.to_parquet(base_checkpoint, index=False)
        print(json.dumps({"dataset_stage": "base_metar_nwp", "status": "built", "rows": len(out)}))
    if spatial_checkpoint.exists() and not args.force:
        out = pd.read_parquet(spatial_checkpoint)
        print(json.dumps({"dataset_stage": "spatial", "status": "cached", "rows": len(out)}))
    else:
        neighbor_metars = {
            station: pd.read_parquet(metar_dir / f"metar_iem_{station}.parquet")
            for station in NEIGHBORS
        }
        out = add_spatial_metar_features_to_frame(
            out,
            neighbor_metars,
            timezone_name=TIMEZONE,
            stations=NEIGHBORS,
        )
        out.to_parquet(spatial_checkpoint, index=False)
        print(json.dumps({"dataset_stage": "spatial", "status": "built", "rows": len(out)}))
    station_metars = {
        station: pd.read_parquet(metar_dir / f"metar_iem_{station}.parquet")
        for station in ADVECTION_STATIONS
    }
    out = add_wind_advection_features_to_frame(
        out,
        station_metars,
        timezone_name=TIMEZONE,
        stations=ADVECTION_STATIONS,
        target_station=AIRPORT,
    )
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool)
    out = out[out["leakage_check_passed"]].sort_values(keys).reset_index(drop=True)
    Path(args.dataset_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.dataset_path, index=False)
    report = {
        "airport": AIRPORT,
        "target_days": int(target["quality_flags"].eq("ok").sum()),
        "dataset_rows": len(out),
        "dataset_days": int(out["target_date_local"].nunique()),
        "period": [out["target_date_local"].min(), out["target_date_local"].max()],
        "leakage_failures": int((~out["leakage_check_passed"]).sum()),
        "spatial_stations": NEIGHBORS,
        "advection_stations": ADVECTION_STATIONS,
    }
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    Path(args.report_dir, "rcss_multinwp_dataset.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str))
    return out


def _backtest(args: argparse.Namespace) -> dict:
    frame = prepare_metar_tmax_dataset(pd.read_parquet(args.dataset_path))
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"]).dt.date
    dates = sorted(frame["target_date_local"].unique())
    folds = _folds(
        dates,
        initial_train_days=args.train_days,
        calibration_days=args.calibration_days,
        test_days=args.test_days,
        step_days=args.step_days,
    )
    if len(folds) < 3:
        raise ValueError(f"At least 3 walk-forward folds are required, got {len(folds)}")
    all_scored = []
    fold_reports = []
    fold_dir = Path(args.report_dir) / "folds"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for fold_index, (train_dates, calibration_dates, test_dates) in enumerate(folds, start=1):
        fold_rows_path = fold_dir / f"fold_{fold_index:02d}_rows.parquet"
        fold_report_path = fold_dir / f"fold_{fold_index:02d}.json"
        if fold_rows_path.exists() and fold_report_path.exists() and not args.force:
            all_scored.append(pd.read_parquet(fold_rows_path))
            fold_reports.append(json.loads(fold_report_path.read_text(encoding="utf-8")))
            print(json.dumps({"fold": fold_index, "status": "cached"}))
            continue
        train = frame[frame["target_date_local"].isin(train_dates)].copy()
        calibration = frame[frame["target_date_local"].isin(calibration_dates)].copy()
        test = frame[frame["target_date_local"].isin(test_dates)].copy()
        weights = _fit_nwp_weights(train)
        train = _with_blend_features(train, weights)
        calibration = _with_blend_features(calibration, weights)
        test = _with_blend_features(test, weights)
        models = _fit_fold_models(train, calibration, args)
        scored = _score_fold(test, train, calibration, models, fold_index)
        all_scored.append(scored)
        fold_summary = _summary(scored, ["model_variant"])
        fold_report = {
            "fold": fold_index,
            "train": [str(train_dates[0]), str(train_dates[-1])],
            "calibration": [str(calibration_dates[0]), str(calibration_dates[-1])],
            "test": [str(test_dates[0]), str(test_dates[-1])],
            "weights": weights,
            "summary": json.loads(fold_summary.to_json(orient="records")),
        }
        fold_reports.append(fold_report)
        scored.to_parquet(fold_rows_path, index=False)
        fold_report_path.write_text(json.dumps(fold_report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"fold": fold_index, "test_days": len(test_dates), "weights": weights}))
    scored = pd.concat(all_scored, ignore_index=True)
    summary = _summary(scored, ["model_variant"])
    daytime = _summary(scored[scored["local_issue_hour"].between(10, 17)], ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _summary(scored, ["model_variant", "season"])
    selection = _select_variant(summary, daytime)
    report = {
        "airport": AIRPORT,
        "experiment": "RCSS leakage-safe D-1 JMA MSM/JMA GSM/ICON Global/ECMWF METAR Tmax",
        "data_period": [str(dates[0]), str(dates[-1])],
        "dataset_days": len(dates),
        "fold_count": len(folds),
        "independent_test_days": int(scored["target_date_local"].nunique()),
        "summary": json.loads(summary.to_json(orient="records")),
        "daytime_10_17_summary": json.loads(daytime.to_json(orient="records")),
        "selection": selection,
        "folds": fold_reports,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "rcss_multinwp_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "rcss_multinwp_backtest_summary.csv", index=False)
    daytime.to_csv(report_dir / "rcss_multinwp_backtest_10_17_summary.csv", index=False)
    by_hour.to_csv(report_dir / "rcss_multinwp_backtest_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "rcss_multinwp_backtest_by_season.csv", index=False)
    (report_dir / "rcss_multinwp_backtest.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"selection": selection, "summary": report["summary"]}, indent=2))
    return report


def _train(args: argparse.Namespace, selected_variant: str | None = None) -> dict:
    report_path = Path(args.report_dir) / "rcss_multinwp_backtest.json"
    if selected_variant is None:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        selected_variant = report["selection"]["selected_variant"]
    base_variant, shape = _parse_selected_variant(selected_variant)
    frame = prepare_metar_tmax_dataset(pd.read_parquet(args.dataset_path))
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"]).dt.date
    dates = sorted(frame["target_date_local"].unique())
    calibration_dates = dates[-args.final_calibration_days :]
    train_dates = dates[: -args.final_calibration_days]
    train = frame[frame["target_date_local"].isin(train_dates)].copy()
    calibration = frame[frame["target_date_local"].isin(calibration_dates)].copy()
    weights = _fit_nwp_weights(train)
    train = _with_blend_features(train, weights)
    calibration = _with_blend_features(calibration, weights)
    feature_columns = _candidate_feature_columns(base_variant)
    if base_variant.startswith("metar_") and base_variant.endswith("_single"):
        prefix = base_variant.removeprefix("metar_").removesuffix("_single")
        train = _alias_single_nwp(train, prefix)
        calibration = _alias_single_nwp(calibration, prefix)
    else:
        prefix = None
    if base_variant.startswith("regression_"):
        regression = RegressionResidualTmaxModel(feature_columns=feature_columns).fit(train, calibration)
        components = _regression_components(base_variant)
        model = regression.with_components(components, calibration)
    else:
        model = (
            _fit_ml(train, calibration, feature_columns, args.min_train_rows, args.max_iter)
            if base_variant == "metar_only"
            else _fit_ensemble(train, calibration, feature_columns, args.min_train_rows, args.max_iter)
        )
    artifact = MultiNwpMetarTmaxModel(
        ensemble=model,
        nwp_weights=weights,
        nwp_prefixes=PREFIXES,
        model_version=MODEL_VERSION,
        shape_strategy=shape,
        residual_nwp_prefix=prefix,
        minimum_nwp_models=2,
        enforce_minimum_with_residual_prefix=True,
    )
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    backtest = json.loads(report_path.read_text(encoding="utf-8"))
    selected_metrics = next(row for row in backtest["summary"] if row["model_variant"] == selected_variant)
    metadata = {
        **artifact.to_metadata(),
        "airport": AIRPORT,
        "timezone": TIMEZONE,
        "coordinates": {"latitude": LATITUDE, "longitude": LONGITUDE},
        "target": "maximum integer temperature reported by RCSS METAR during the local day",
        "training_source": "IEM METAR archive plus Open-Meteo Previous Runs day1",
        "nwp_models": RCSS_NWP_MODELS,
        "feature_columns": feature_columns,
        "spatial_stations": NEIGHBORS if "spatial_wind" in base_variant else [],
        "advection_stations": ADVECTION_STATIONS if "spatial_wind" in base_variant else [],
        "training_period": [str(train_dates[0]), str(train_dates[-1])],
        "calibration_period": [str(calibration_dates[0]), str(calibration_dates[-1])],
        "training_rows": len(train),
        "calibration_rows": len(calibration),
        "walk_forward_selected_variant": selected_variant,
        "walk_forward_metrics": selected_metrics,
        "walk_forward_test_days": backtest["independent_test_days"],
        "leakage_contract": "all NWP values are fixed 24-hour lead forecasts known by local day start; METAR is as-of issue time",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "Previous-day1 trajectories combine fixed 24-hour lead values rather than one single model initialization.",
            "RCTP is the only spatial station used to avoid mixing Taiwan's mountain regimes.",
            "Live prediction requires at least two of four NWP providers; otherwise no forecast is sent.",
        ],
    }
    Path(args.metadata_path).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))
    return metadata


def _fit_fold_models(train: pd.DataFrame, calibration: pd.DataFrame, args) -> dict[str, object]:
    tasks = [("metar_only", train, calibration, BASE_FEATURES, False)]
    for prefix in PREFIXES:
        single_train = _alias_single_nwp(train, prefix)
        single_cal = _alias_single_nwp(calibration, prefix)
        features = _candidate_feature_columns(f"metar_{prefix}_single")
        tasks.append((f"metar_{prefix}_single", single_train, single_cal, features, True))
    tasks.extend(
        [
            ("metar_multinwp", train, calibration, _candidate_feature_columns("metar_multinwp"), True),
            (
                "metar_multinwp_spatial_wind",
                train,
                calibration,
                _candidate_feature_columns("metar_multinwp_spatial_wind"),
                True,
            ),
        ]
    )
    fitted = joblib.Parallel(n_jobs=args.n_jobs, backend="loky")(
        joblib.delayed(_fit_candidate_task)(task, args.min_train_rows, args.max_iter)
        for task in tasks
    )
    models = dict(fitted)
    regression_features = _candidate_feature_columns("regression_mean")
    regression = RegressionResidualTmaxModel(feature_columns=regression_features).fit(
        train,
        calibration,
    )
    for name in ("ridge", "hist_gradient_boosting", "extra_trees"):
        models[f"regression_{name}"] = regression.with_components((name,), calibration)
    models["regression_mean"] = regression
    return models


def _fit_candidate_task(task, min_train_rows: int, max_iter: int):
    name, train, calibration, features, ensemble = task
    model = (
        _fit_ensemble(train, calibration, features, min_train_rows, max_iter)
        if ensemble
        else _fit_ml(train, calibration, features, min_train_rows, max_iter)
    )
    return name, model


def _candidate_feature_columns(name: str) -> list[str]:
    if name == "metar_only":
        return BASE_FEATURES
    if name.startswith("metar_") and name.endswith("_single"):
        prefix = name.removeprefix("metar_").removesuffix("_single")
        return list(dict.fromkeys(BASE_FEATURES + prefixed_nwp_feature_columns([prefix]) + AGGREGATE_FEATURES))
    features = BASE_FEATURES + prefixed_nwp_feature_columns(PREFIXES) + AGGREGATE_FEATURES
    if name.startswith("regression_"):
        features += spatial_feature_columns(NEIGHBORS)
        features += wind_advection_feature_columns(ADVECTION_STATIONS, target_station=AIRPORT)
    if name == "metar_multinwp_spatial_wind":
        features += spatial_feature_columns(NEIGHBORS)
        features += wind_advection_feature_columns(ADVECTION_STATIONS, target_station=AIRPORT)
    return list(dict.fromkeys(features))


def _fit_ml(train, calibration, features, min_rows, max_iter) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(
        feature_columns=features,
        min_rows=min_rows,
        max_iter=max_iter,
    ).fit(train)
    rows = _calibration_rows(model, calibration)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(rows)
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _fit_ensemble(train, calibration, features, min_rows, max_iter) -> IconD2MetarTmaxEnsemble:
    ml = _fit_ml(train, calibration, features, min_rows, max_iter)
    residuals = _residuals_by_hour(train)
    weight = _optimize_ml_weight(calibration, ml, residuals)
    final_residuals = _residuals_by_hour(pd.concat([train, calibration], ignore_index=True))
    return IconD2MetarTmaxEnsemble(ml, final_residuals, weight, MODEL_VERSION)


def _calibration_rows(model: MetarTmaxUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_upside_survival_frame(frame)
    rows = []
    for idx, row in frame.iterrows():
        item = {
            "local_issue_hour": int(row["local_issue_hour"]),
            "season": _season(row["target_date_local"]),
            "remaining_upside_c": float(row["remaining_upside_c"]),
        }
        for threshold in range(1, model.max_upside_c + 1):
            item[f"raw_probability_upside_ge_{threshold}c"] = float(raw.loc[idx, f"probability_upside_ge_{threshold}c"])
            item[f"actual_upside_ge_{threshold}c"] = float(row["remaining_upside_c"] >= threshold)
        rows.append(item)
    return pd.DataFrame(rows)


def _residuals_by_hour(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    work = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    work["residual_c"] = work["final_metar_tmax_c"].astype(float) - work["model_tmax_c"].astype(float)
    out = {-1: work.drop_duplicates("target_date_local")["residual_c"].to_numpy(dtype=float)}
    for hour, group in work.groupby("local_issue_hour"):
        samples = group["residual_c"].to_numpy(dtype=float)
        if len(samples) >= 20:
            out[int(hour)] = samples
    return out


def _optimize_ml_weight(calibration, ml, residuals) -> float:
    shell = IconD2MetarTmaxEnsemble(ml, residuals, 0.0, "calibration")
    cache = [(shell.residual_distribution(row), ml.predict_distribution(row), float(row["final_metar_tmax_c"])) for _, row in calibration.iterrows()]
    best = (np.inf, 0.0)
    for weight in np.linspace(0.0, 1.0, 21):
        losses = []
        for residual_dist, ml_dist, actual in cache:
            left = dict(zip(residual_dist.bins_c, residual_dist.probabilities))
            right = dict(zip(ml_dist.bins_c, ml_dist.probabilities))
            actual_bin = int(round(actual))
            probability = (1 - weight) * left.get(actual_bin, 0.0) + weight * right.get(actual_bin, 0.0)
            losses.append(-np.log(max(probability, 1e-12)))
        score = float(np.mean(losses))
        if score < best[0]:
            best = (score, float(weight))
    return best[1]


def _score_fold(test, train, calibration, models, fold_index) -> pd.DataFrame:
    history = pd.concat([train, calibration], ignore_index=True)
    raw_models = {}
    for prefix in PREFIXES:
        aliased = _alias_single_nwp(history, prefix)
        raw_models[f"raw_{prefix}_residual"] = _ResidualModel(_residuals_by_hour(aliased))
    raw_models["raw_multinwp_residual"] = _ResidualModel(_residuals_by_hour(history))
    rows = []
    for _, row in test.iterrows():
        actual = float(row["final_metar_tmax_c"])
        for name, model in raw_models.items():
            source_row = row if name == "raw_multinwp_residual" else _single_row(row, name.removeprefix("raw_").removesuffix("_residual"))
            rows.append(_score_row(source_row, name, model.predict_distribution(source_row), actual, fold_index))
        for name, model in models.items():
            source_row = row
            if name.startswith("metar_") and name.endswith("_single"):
                source_row = _single_row(row, name.removeprefix("metar_").removesuffix("_single"))
            dist = model.predict_distribution(source_row)
            rows.append(_score_row(source_row, name, dist, actual, fold_index))
            rows.append(_score_row(source_row, f"{name}__unimodal", project_unimodal_distribution(dist), actual, fold_index))
            rows.append(
                _score_row(
                    source_row,
                    f"{name}__temperature_unimodal_067",
                    project_unimodal_distribution(temperature_scale_distribution(dist, 0.67)),
                    actual,
                    fold_index,
                )
            )
    return pd.DataFrame(rows)


class _ResidualModel:
    def __init__(self, residuals_by_hour: dict[int, np.ndarray]):
        self.residuals_by_hour = residuals_by_hour

    def predict_distribution(self, row) -> TmaxDistribution:
        samples = self.residuals_by_hour.get(int(row["local_issue_hour"]), self.residuals_by_hour[-1])
        values = np.rint(float(row["model_tmax_c"]) + samples).astype(int)
        bins = np.arange(values.min(), values.max() + 1)
        probs = np.asarray([(values == value).sum() for value in bins], dtype=float)
        return TmaxDistribution(bins, probs).truncate_below(float(row["current_metar_max_c"]))


def _score_row(row, variant, dist, actual, fold_index) -> dict:
    current = float(row["current_metar_max_c"])
    low, high = dist.interval(0.80)
    return {
        "fold": fold_index,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": _season(row["target_date_local"]),
        "model_variant": variant,
        "actual_metar_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current + 1))), actual - current >= 1),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current + 2))), actual - current >= 2),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current + 3))), actual - current >= 3),
        "coverage_80": bool(low <= actual <= high),
        "interval_80_width_c": float(high - low),
        "mode_hit": int(round(actual)) == dist.most_likely_integer_c,
        "mode_error_ge_2c": abs(dist.most_likely_integer_c - int(round(actual))) >= 2,
        "shape_violations": unimodal_violation_count(dist),
    }


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in frame.groupby(groups, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        item = dict(zip(groups, key, strict=True))
        item.update(
            {
                "rows": len(group),
                "distinct_days": group["target_date_local"].nunique(),
                "mae_expected": mae(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "rmse_expected": rmse(group["actual_metar_tmax_c"], group["expected_tmax_c"]),
                "bias_expected": float(group["bias_expected"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
                "mean_interval_80_width_c": float(group["interval_80_width_c"].mean()),
                "mode_hit_rate": float(group["mode_hit"].mean()),
                "mode_error_ge_2c_rate": float(group["mode_error_ge_2c"].mean()),
                "mean_shape_violations": float(group["shape_violations"].mean()),
            }
        )
        rows.append(item)
    return pd.DataFrame(rows).sort_values(groups).reset_index(drop=True)


def _select_variant(summary: pd.DataFrame, daytime: pd.DataFrame) -> dict:
    candidates = summary[
        summary["model_variant"].str.startswith(("metar_", "regression_"))
    ].copy()
    day = daytime.set_index("model_variant")
    candidates["daytime_mae"] = candidates["model_variant"].map(day["mae_expected"])
    candidates["daytime_nll"] = candidates["model_variant"].map(day["mean_nll"])
    best_mae = candidates["mae_expected"].min()
    best_crps = candidates["mean_crps"].min()
    best_day_mae = candidates["daytime_mae"].min()
    eligible = candidates[
        (candidates["mae_expected"] <= best_mae + 0.08)
        & (candidates["mean_crps"] <= best_crps + 0.003)
        & (candidates["daytime_mae"] <= best_day_mae + 0.08)
        & (candidates["coverage_80"].between(0.75, 0.95))
        & (candidates["bias_expected"].abs() <= 0.20)
    ].copy()
    if eligible.empty:
        eligible = candidates.copy()
    eligible["selection_score"] = (
        eligible["mean_nll"]
        + 2.0 * eligible["mean_crps"]
        + 0.15 * eligible["mae_expected"]
        + 0.10 * eligible["daytime_mae"]
        + 0.25 * eligible["bias_expected"].abs()
    )
    selected = eligible.sort_values(["selection_score", "mean_nll"]).iloc[0]
    return {
        "selected_variant": selected["model_variant"],
        "selection_score": float(selected["selection_score"]),
        "eligible_variants": eligible.sort_values("selection_score")["model_variant"].tolist(),
        "criteria": "near-best MAE/CRPS/daytime MAE, sane coverage, |bias| <= 0.20; then composite probabilistic score",
    }


def _fit_nwp_weights(frame: pd.DataFrame) -> dict[str, float]:
    daily = frame.sort_values("issue_time_utc").drop_duplicates("target_date_local")
    columns = [f"{prefix}_tmax_c" for prefix in PREFIXES]
    clean = daily.dropna(subset=columns + ["final_metar_tmax_c"])
    X = clean[columns].to_numpy(dtype=float)
    y = clean["final_metar_tmax_c"].to_numpy(dtype=float)
    initial = np.full(len(PREFIXES), 1.0 / len(PREFIXES))

    def objective(weights):
        residual = X @ weights - y
        return float(np.mean(np.sqrt(residual**2 + 0.25)) + 0.08 * np.sum((weights - initial) ** 2))

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(PREFIXES),
        constraints={"type": "eq", "fun": lambda values: float(np.sum(values) - 1.0)},
        options={"maxiter": 500, "ftol": 1e-10},
    )
    weights = result.x if result.success else initial
    weights = np.clip(weights, 0.0, 1.0)
    weights /= weights.sum()
    return {prefix: float(value) for prefix, value in zip(PREFIXES, weights, strict=True)}


def _with_blend_features(frame: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = frame.copy()
    additions = pd.DataFrame(
        [blend_nwp_features(row, weights, prefixes=PREFIXES) for _, row in out.iterrows()],
        index=out.index,
    )
    for column in additions:
        out[column] = additions[column]
    out["nwp_model_minus_current_max_c"] = out["model_tmax_c"] - out["current_metar_max_c"]
    out["nwp_future_minus_current_max_c"] = out["model_future_temp_max_c"] - out["current_metar_max_c"]
    return out


def _alias_single_nwp(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out["model_tmax_c"] = out[f"{prefix}_tmax_c"]
    out["model_future_temp_max_c"] = out[f"{prefix}_future_temp_max_c"]
    out["nwp_model_minus_current_max_c"] = out["model_tmax_c"] - out["current_metar_max_c"]
    out["nwp_future_minus_current_max_c"] = out["model_future_temp_max_c"] - out["current_metar_max_c"]
    return out


def _single_row(row: pd.Series, prefix: str) -> pd.Series:
    out = row.copy()
    out["model_tmax_c"] = out[f"{prefix}_tmax_c"]
    out["model_future_temp_max_c"] = out[f"{prefix}_future_temp_max_c"]
    out["nwp_model_minus_current_max_c"] = out["model_tmax_c"] - out["current_metar_max_c"]
    out["nwp_future_minus_current_max_c"] = out["model_future_temp_max_c"] - out["current_metar_max_c"]
    return out


def _folds(dates, *, initial_train_days, calibration_days, test_days, step_days):
    out = []
    test_start = initial_train_days + calibration_days
    while test_start + test_days <= len(dates):
        calibration_start = test_start - calibration_days
        out.append((dates[:calibration_start], dates[calibration_start:test_start], dates[test_start:test_start + test_days]))
        test_start += step_days
    return out


def _parse_selected_variant(name: str) -> tuple[str, str]:
    if name.endswith("__temperature_unimodal_067"):
        return name.removesuffix("__temperature_unimodal_067"), "temperature_unimodal"
    if name.endswith("__unimodal"):
        return name.removesuffix("__unimodal"), "unimodal"
    return name, "raw"


def _regression_components(name: str) -> tuple[str, ...]:
    suffix = name.removeprefix("regression_")
    if suffix == "mean":
        return ("ridge", "hist_gradient_boosting", "extra_trees")
    if suffix not in {"ridge", "hist_gradient_boosting", "extra_trees"}:
        raise ValueError(f"Unknown regression variant: {name}")
    return (suffix,)


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _fetch_metar_with_retries(station, start, end):
    last_error = None
    for attempt in range(4):
        try:
            return IEMAdapter().fetch_metar(station, start, end)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"METAR download failed for {station}: {last_error}")


def _common_args(parser):
    parser.add_argument("--start-date", default="2024-02-05")
    parser.add_argument("--end-date", default="2026-07-21")
    parser.add_argument("--metar-dir", default="data/interim/rcss_multinwp")
    parser.add_argument("--nwp-cache-dir", default="data/forecasts/rcss_multinwp")
    parser.add_argument("--target-path", default="data/processed/metar_tmax_target_RCSS.parquet")
    parser.add_argument("--dataset-path", default="data/processed/metar_upside_dataset_RCSS_multinwp.parquet")
    parser.add_argument("--report-dir", default="data/reports/rcss_multinwp")
    parser.add_argument("--model-path", default=f"data/models/{MODEL_VERSION}.joblib")
    parser.add_argument("--metadata-path", default=f"data/models/{MODEL_VERSION}.metadata.json")
    parser.add_argument("--chunk-days", type=int, default=90)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--calibration-days", type=int, default=90)
    parser.add_argument("--test-days", type=int, default=60)
    parser.add_argument("--step-days", type=int, default=60)
    parser.add_argument("--final-calibration-days", type=int, default=120)
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--max-iter", type=int, default=70)
    parser.add_argument("--n-jobs", type=int, default=4)


def _parse_args():
    parser = argparse.ArgumentParser(description="Build, backtest and train the RCSS multi-NWP METAR Tmax model.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare-metar", "prepare-nwp", "build-dataset", "backtest", "train", "all"]:
        _common_args(sub.add_parser(name))
    return parser.parse_args()


if __name__ == "__main__":
    main()
