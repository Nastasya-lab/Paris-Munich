from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.discrete_hazard_tmax import (
    DiscreteHazardCalibrator,
    DiscreteHazardUpsideModel,
    hazard_calibration_rows,
)
from weather_tmax_bot.models.distribution import TmaxDistribution, unimodal_violation_count
from weather_tmax_bot.models.metar_tmax_model import IconD2MetarTmaxEnsemble, MetarTmaxUpsideModel, prepare_metar_tmax_dataset
from weather_tmax_bot.models.model_registry import register_artifact
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_discrete_hazard_spatial_wind_advection_shadow_v1"


def main() -> None:
    args = _parse_args()
    dataset = pd.read_parquet(args.dataset)
    frame = prepare_metar_tmax_dataset(dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame = frame[frame["model_tmax_c"].notna()].sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)
    feature_columns = _feature_columns(args.feature_metadata)

    model = DiscreteHazardUpsideModel(
        min_rows=args.min_train_rows,
        min_at_risk_rows=args.min_at_risk_rows,
        max_iter=args.max_iter,
        feature_columns=feature_columns,
    ).fit(train)
    calibrator = DiscreteHazardCalibrator(max_upside_c=model.max_upside_c).fit(hazard_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    residuals = _residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True))
    ensemble = IconD2MetarTmaxEnsemble(
        ml_model=model,
        residuals_by_hour=residuals,
        ml_weight=args.ml_weight,
        model_version=MODEL_VERSION,
    )
    scored = _score_holdout(test, ensemble)
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    metrics = summary.iloc[0].to_dict()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_VERSION}.joblib"
    metadata_path = model_dir / f"{MODEL_VERSION}.metadata.json"
    joblib.dump(ensemble, model_path)

    metadata = {
        "model_name": "lfpb_discrete_hazard_spatial_wind_advection_shadow",
        "model_version": MODEL_VERSION,
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "role": "shadow_diagnostic",
        "feature_set_version": "lfpb.metar_tmax.icon_d2.spatial_wind_advection.discrete_hazard.v1",
        "feature_columns": feature_columns,
        "usable_rows": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "selected_ml_weight": args.ml_weight,
        "hazard_model_metadata": model.to_metadata(),
        "calibration_metadata": calibrator.to_metadata(),
        "holdout_metrics": metrics,
        "comparison_note": "Backtest showed lower NLL than working survival wind/advection, but lower 80% coverage and mixed daytime behavior; use as shadow only.",
        "data_snapshot_hash": stable_hash(
            {
                "rows": len(frame),
                "target_sum": float(frame["final_metar_tmax_c"].sum()),
                "model_tmax_sum": float(frame["model_tmax_c"].sum()),
                "feature_count": len(feature_columns),
            }
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    register_artifact(
        version=MODEL_VERSION,
        artifact_type="model",
        path=model_path,
        metadata_path=metadata_path,
        metrics=metrics,
        model_dir=model_dir,
    )

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_discrete_hazard_shadow_holdout_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_discrete_hazard_shadow_holdout_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_discrete_hazard_shadow_holdout_by_hour.csv", index=False)
    (report_dir / "lfpb_discrete_hazard_shadow_training.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, default=str))


def _feature_columns(path: str | Path) -> list[str]:
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    columns = metadata.get("feature_columns") or []
    if not columns:
        raise ValueError(f"No feature_columns in {path}")
    return list(columns)


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


def _residual_samples_by_hour(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    data = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    data["residual"] = data["final_metar_tmax_c"].astype(float) - data["model_tmax_c"].astype(float)
    residuals = {int(hour): group["residual"].to_numpy(dtype=float) for hour, group in data.groupby("local_issue_hour")}
    residuals[-1] = data["residual"].to_numpy(dtype=float)
    return residuals


def _score_holdout(test: pd.DataFrame, ensemble: IconD2MetarTmaxEnsemble) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score(MODEL_VERSION, row, ensemble.predict_distribution(row)))
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    mode_error = abs(float(dist.most_likely_integer_c) - actual)
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "mode_hit": bool(mode_error < 0.5),
        "mode_error_ge_2c": bool(mode_error >= 2.0),
        "shape_violations": unimodal_violation_count(dist),
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
                "mode_hit_rate": float(group["mode_hit"].mean()),
                "mode_error_ge_2c_rate": float(group["mode_error_ge_2c"].mean()),
                "mean_shape_violations": float(group["shape_violations"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFPB discrete hazard shadow model.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2_spatial_advection.parquet")
    parser.add_argument("--feature-metadata", default="data/models/lfpb_metar_tmax_icon_d2_spatial_wind_advection_v1.metadata.json")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--min-train-rows", type=int, default=500)
    parser.add_argument("--min-at-risk-rows", type=int, default=80)
    parser.add_argument("--max-iter", type=int, default=70)
    parser.add_argument("--ml-weight", type=float, default=0.85)
    return parser.parse_args()


if __name__ == "__main__":
    main()
