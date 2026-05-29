from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.models.calibration import DiscreteSpreadCalibrator
from weather_tmax_bot.models.model_registry import register_artifact, save_model
from weather_tmax_bot.models.quantile_model import QuantileTmaxModel
from weather_tmax_bot.utils.hashing import stable_hash


def train_quantile_model(
    dataset: pd.DataFrame,
    model_version: str = "quantile_mvp",
    model_dir: str | Path = "data/models",
) -> QuantileTmaxModel:
    if "tmax_c" not in dataset.columns:
        raise ValueError("dataset must contain tmax_c")
    X = dataset.drop(columns=["tmax_c"])
    y = pd.to_numeric(dataset["tmax_c"], errors="coerce")
    mask = y.notna()
    model = QuantileTmaxModel().fit(X.loc[mask], y.loc[mask])
    calibrator = _fit_spread_calibrator(dataset)
    snapshot_hash = _dataset_snapshot_hash(dataset)
    save_model(
        model,
        {
            "model_name": "quantile_gradient_boosting",
            "model_version": model_version,
            "training_period": [str(dataset["target_date_local"].min()), str(dataset["target_date_local"].max())],
            "validation_period": None,
            "feature_set_version": "mvp.v1",
            "source_registry_version": "2026-05-28.v1",
            "data_snapshot_hash": snapshot_hash,
            "calibration_version": f"{model_version}.calibrator" if calibrator is not None else None,
            "git_commit": None,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        model_dir=model_dir,
    )
    register_artifact(
        version=model_version,
        artifact_type="model",
        path=Path(model_dir) / f"{model_version}.joblib",
        metadata_path=Path(model_dir) / f"{model_version}.metadata.json",
        model_dir=model_dir,
    )
    if calibrator is not None:
        save_model(
            calibrator,
            {
                "model_name": "discrete_spread_calibrator",
                "model_version": f"{model_version}.calibrator",
                "training_period": "validation_holdout_2025",
                "validation_period": "2025",
                "feature_set_version": "mvp.v1",
                "source_registry_version": "2026-05-28.v1",
                "data_snapshot_hash": snapshot_hash,
                "calibration_version": f"spread_sigma_{calibrator.sigma_bins}",
                "git_commit": None,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            model_dir=model_dir,
        )
        register_artifact(
            version=f"{model_version}.calibrator",
            artifact_type="calibrator",
            path=Path(model_dir) / f"{model_version}.calibrator.joblib",
            metadata_path=Path(model_dir) / f"{model_version}.calibrator.metadata.json",
            model_dir=model_dir,
        )
    return model


def _fit_spread_calibrator(dataset: pd.DataFrame) -> DiscreteSpreadCalibrator | None:
    df = dataset.copy()
    if "target_date_local" not in df.columns:
        return None
    df["target_date_local"] = pd.to_datetime(df["target_date_local"]).dt.date
    validation_start = pd.to_datetime("2025-01-01").date()
    df = df[df["issue_hour_utc"] == 6].copy()
    train = df[df["target_date_local"] < validation_start]
    valid = df[df["target_date_local"] >= validation_start]
    if len(train) < 365 or len(valid) < 30:
        return None
    temp_model = QuantileTmaxModel().fit(train.drop(columns=["tmax_c"]), train["tmax_c"])
    distributions = [
        temp_model.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]))
        for _, row in valid.iterrows()
    ]
    return DiscreteSpreadCalibrator().fit(distributions, valid["tmax_c"].to_numpy(dtype=float))


def _dataset_snapshot_hash(dataset: pd.DataFrame) -> str:
    parts = {
        "rows": len(dataset),
        "columns": sorted(dataset.columns),
        "target_start": str(dataset["target_date_local"].min()) if "target_date_local" in dataset.columns else None,
        "target_end": str(dataset["target_date_local"].max()) if "target_date_local" in dataset.columns else None,
        "target_sum": float(pd.to_numeric(dataset["tmax_c"], errors="coerce").sum()) if "tmax_c" in dataset else None,
    }
    return stable_hash(parts)
