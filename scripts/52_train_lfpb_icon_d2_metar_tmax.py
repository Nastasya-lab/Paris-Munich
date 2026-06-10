from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
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
    survival_to_probabilities,
)
from weather_tmax_bot.models.model_registry import register_artifact
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_metar_tmax_icon_d2_v1"

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
    dataset = pd.read_parquet(args.metar_dataset)
    nwp = pd.read_parquet(args.nwp_archive)
    joined = _join_asof_nwp(dataset, nwp)
    if joined.empty:
        raise ValueError("No leakage-safe LFPB ICON-D2 rows available for training")

    output_dataset = Path(args.output_dataset)
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_dataset, index=False)

    frame = prepare_metar_tmax_dataset(joined)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["model_tmax_c"].notna()].copy()
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    if len(frame) < args.min_train_rows + args.min_calibration_rows + args.min_test_rows:
        raise ValueError(
            "Not enough rows for train/calibration/test split: "
            f"{len(frame)} rows available"
        )

    train_core, calibration, test, split = _time_split(
        frame,
        min_train_rows=args.min_train_rows,
        min_calibration_rows=args.min_calibration_rows,
        min_test_rows=args.min_test_rows,
    )
    feature_columns = list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS)

    icon_model = MetarTmaxUpsideModel(
        min_rows=args.min_train_rows,
        max_iter=args.max_iter,
        feature_columns=feature_columns,
    ).fit(train_core)
    icon_calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=icon_model.max_upside_c).fit(
        _survival_calibration_rows(icon_model, calibration)
    )
    icon_model.calibrator = icon_calibrator if icon_calibrator.fitted else None

    metar_model = MetarTmaxUpsideModel(
        min_rows=args.min_train_rows,
        max_iter=args.max_iter,
        feature_columns=list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES),
    ).fit(train_core)
    metar_calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=metar_model.max_upside_c).fit(
        _survival_calibration_rows(metar_model, calibration)
    )
    metar_model.calibrator = metar_calibrator if metar_calibrator.fitted else None

    calibration_residuals = _residual_samples(train_core)
    ml_weight = _optimize_ml_weight(calibration, icon_model, calibration_residuals)
    residuals = _residual_samples(pd.concat([train_core, calibration], ignore_index=True))
    ensemble = IconD2MetarTmaxEnsemble(
        ml_model=icon_model,
        residuals_by_hour=residuals,
        ml_weight=ml_weight,
        model_version=MODEL_VERSION,
    )
    scored = _score_holdout(
        test=test,
        icon_model=icon_model,
        icon_ensemble=ensemble,
        metar_model=metar_model,
        residuals=residuals,
    )
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _group_summary(scored, ["model_variant", "season"])
    candidate_metrics = _metrics_for(summary, "lfpb_icon_d2_ensemble_candidate")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_VERSION}.joblib"
    metadata_path = model_dir / f"{MODEL_VERSION}.metadata.json"
    joblib.dump(ensemble, model_path)

    metadata = {
        "model_name": "lfpb_icon_d2_metar_tmax_remaining_upside",
        "model_version": MODEL_VERSION,
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "training_source": "IEM METAR historical observations + Open-Meteo forecast-as-issued ICON-D2 single runs",
        "feature_set_version": "lfpb.metar_tmax.icon_d2.intraday_enhanced.v2",
        "source_registry_version": "2026-06-10.lfpb.icon_d2.intraday_enhanced",
        "feature_columns": feature_columns,
        "enhanced_intraday_feature_columns": ENHANCED_INTRADAY_FEATURES,
        "rows_joined": len(joined),
        "usable_rows": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "calibration_metadata": icon_calibrator.to_metadata(),
        "ensemble_metadata": ensemble.to_metadata(),
        "selected_ml_weight": ml_weight,
        "holdout_metrics": candidate_metrics,
        "comparison_summary": json.loads(summary.to_json(orient="records")),
        "data_snapshot_hash": stable_hash(
            {
                "rows": len(frame),
                "target_sum": float(frame["final_metar_tmax_c"].sum()),
                "model_tmax_sum": float(frame["model_tmax_c"].sum()),
                "target_start": str(frame["target_date_local"].min()),
                "target_end": str(frame["target_date_local"].max()),
            }
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "promotion_status": "production_artifact_updated",
        "promotion_note": "Production LFPB ICON-D2 model path updated in-place with enhanced intraday METAR features.",
        "limitations": [
            "Target is METAR Tmax, not official Meteo-France TX.",
            "TAF is not used because the IEM historical TAF archive returned zero LFPB rows.",
            "The model is trained on the currently available forecast-as-issued ICON-D2 overlap window.",
            "Enhanced intraday features are computed from as-of METAR only; live quality depends on AWC METAR parser coverage.",
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
    scored.to_parquet(report_dir / "lfpb_icon_d2_metar_tmax_holdout_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_icon_d2_metar_tmax_holdout_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_icon_d2_metar_tmax_holdout_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_icon_d2_metar_tmax_holdout_by_season.csv", index=False)
    (report_dir / "lfpb_icon_d2_metar_tmax_training.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/lfpb_icon_d2_metar_tmax_model.md").write_text(
        _markdown(metadata, summary, by_hour, by_season),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, default=str))


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
            if column in latest:
                merged[column] = latest[column]
        merged["nwp_model_minus_current_max_c"] = float(latest["model_tmax_c"]) - float(row["current_metar_max_c"])
        future = latest.get("model_future_temp_max_c")
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
    train_core = frame[frame["target_date_local"].isin(dates[:train_end_idx])].copy()
    calibration = frame[frame["target_date_local"].isin(dates[train_end_idx:calibration_end_idx])].copy()
    test = frame[frame["target_date_local"].isin(dates[calibration_end_idx:])].copy()

    while len(train_core) < min_train_rows and calibration_end_idx < len(dates) - 1:
        train_end_idx += 1
        calibration_end_idx += 1
        train_core = frame[frame["target_date_local"].isin(dates[:train_end_idx])].copy()
        calibration = frame[frame["target_date_local"].isin(dates[train_end_idx:calibration_end_idx])].copy()
        test = frame[frame["target_date_local"].isin(dates[calibration_end_idx:])].copy()

    if len(calibration) < min_calibration_rows or len(test) < min_test_rows:
        raise ValueError(
            "Time split produced insufficient rows: "
            f"train={len(train_core)}, calibration={len(calibration)}, test={len(test)}"
        )

    return train_core, calibration, test, {
        "method": "chronological_60_20_20_by_target_day",
        "train_start": str(train_core["target_date_local"].min()),
        "train_end": str(train_core["target_date_local"].max()),
        "calibration_start": str(calibration["target_date_local"].min()),
        "calibration_end": str(calibration["target_date_local"].max()),
        "test_start": str(test["target_date_local"].min()),
        "test_end": str(test["target_date_local"].max()),
        "train_rows": len(train_core),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_days": int(train_core["target_date_local"].nunique()),
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
            out[f"raw_probability_upside_ge_{threshold}c"] = float(
                raw.loc[index, f"probability_upside_ge_{threshold}c"]
            )
            out[f"actual_upside_ge_{threshold}c"] = float(row["remaining_upside_c"] >= threshold)
        rows.append(out)
    return pd.DataFrame(rows)


def _distribution_from_survival(row: pd.Series, survival: dict[int, float], max_upside_c: int) -> TmaxDistribution:
    probs = survival_to_probabilities(survival, max_upside_c)
    bins = np.rint(float(row["current_metar_max_c"]) + np.arange(max_upside_c + 1)).astype(int)
    return TmaxDistribution(bins, probs)


def _score_holdout(
    *,
    test: pd.DataFrame,
    icon_model: MetarTmaxUpsideModel,
    icon_ensemble: IconD2MetarTmaxEnsemble,
    metar_model: MetarTmaxUpsideModel,
    residuals: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("lfpb_icon_d2_ensemble_candidate", row, icon_ensemble.predict_distribution(row)))
        rows.append(_score("lfpb_icon_d2_ml_calibrated", row, icon_model.predict_distribution(row)))
        rows.append(_score("lfpb_metar_only_calibrated", row, metar_model.predict_distribution(row)))
        rows.append(_score("raw_icon_d2_residual_distribution", row, _raw_icon_residual_distribution(row, residuals)))
        rows.append(
            _score(
                "persistence_current_metar_max",
                row,
                TmaxDistribution(np.array([int(round(row["current_metar_max_c"]))]), np.array([1.0])),
            )
        )
    return pd.DataFrame(rows)


def _optimize_ml_weight(
    calibration: pd.DataFrame,
    icon_model: MetarTmaxUpsideModel,
    residuals: dict[int, np.ndarray],
) -> float:
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        ensemble = IconD2MetarTmaxEnsemble(
            ml_model=icon_model,
            residuals_by_hour=residuals,
            ml_weight=float(weight),
            model_version=MODEL_VERSION,
        )
        scores = [
            nll_integer_bin(ensemble.predict_distribution(row), float(row["final_metar_tmax_c"]))
            for _, row in calibration.iterrows()
        ]
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": _season(row["target_date_local"]),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current_max + 1))), bool(row["remaining_upside_c"] >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current_max + 2))), bool(row["remaining_upside_c"] >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current_max + 3))), bool(row["remaining_upside_c"] >= 3.0)),
        "coverage_80": _covered(dist, actual, 0.80),
    }


def _residual_samples(train: pd.DataFrame) -> dict[int, np.ndarray]:
    train = train.copy()
    train["residual"] = pd.to_numeric(train["final_metar_tmax_c"], errors="coerce") - pd.to_numeric(
        train["model_tmax_c"],
        errors="coerce",
    )
    out = {}
    for hour, group in train.groupby("local_issue_hour"):
        values = group["residual"].dropna().to_numpy(dtype=float)
        if len(values):
            out[int(hour)] = values
    out[-1] = train["residual"].dropna().to_numpy(dtype=float)
    return out


def _raw_icon_residual_distribution(row: pd.Series, residuals: dict[int, np.ndarray]) -> TmaxDistribution:
    samples = residuals.get(int(row["local_issue_hour"]), residuals.get(-1))
    if samples is None or len(samples) < 20:
        samples = residuals.get(-1, np.array([0.0]))
    rounded = np.rint(float(row["model_tmax_c"]) + samples).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    return TmaxDistribution(bins, probabilities).truncate_below(float(row["current_metar_max_c"]))


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
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _metrics_for(summary: pd.DataFrame, model_variant: str) -> dict:
    row = summary[summary["model_variant"] == model_variant]
    if row.empty:
        return {}
    return json.loads(row.iloc[0].to_json())


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _markdown(metadata: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_season: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB ICON-D2 METAR Tmax model",
            "",
            "Forecast-as-issued ICON-D2 candidate for daily maximum temperature reported by METAR.",
            "",
            f"- model version: `{metadata['model_version']}`",
            f"- target period: `{metadata['target_period'][0]}` to `{metadata['target_period'][1]}`",
            f"- usable rows: `{metadata['usable_rows']}`",
            f"- days joined: `{metadata['days_joined']}`",
            f"- promotion: `{metadata['promotion_status']}`",
            "",
            "## Holdout Overall",
            "",
            _table(summary),
            "",
            "## By Local Issue Hour",
            "",
            _table(by_hour),
            "",
            "## By Season",
            "",
            _table(by_season),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in metadata["limitations"]],
            "",
        ]
    )


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFPB ICON-D2-aware METAR Tmax model.")
    parser.add_argument("--metar-dataset", default="data/processed/metar_upside_dataset_LFPB_intraday_enhanced.parquet")
    parser.add_argument("--nwp-archive", default="data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    parser.add_argument("--output-dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    parser.add_argument("--min-calibration-rows", type=int, default=300)
    parser.add_argument("--min-test-rows", type=int, default=300)
    parser.add_argument("--max-iter", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    main()
