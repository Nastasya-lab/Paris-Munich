from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.evaluation.metrics import bias, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.model_registry import promote_model, register_artifact, save_model
from weather_tmax_bot.models.nwp_residual_model import NWPResidualDistributionModel
from weather_tmax_bot.utils.hashing import stable_hash


def main() -> None:
    dataset = pd.read_parquet("data/processed/training_dataset.parquet")
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"]).dt.date
    nwp = dataset[(dataset["nwp_missing"] == False) & dataset["model_tmax_c"].notna()].copy()  # noqa: E712
    train_start = pd.to_datetime("2025-05-31").date()
    test_start = pd.to_datetime("2026-01-01").date()
    test_end = pd.to_datetime("2026-05-27").date()
    train = nwp[(nwp["target_date_local"] >= train_start) & (nwp["target_date_local"] < test_start)].copy()
    test = nwp[(nwp["target_date_local"] >= test_start) & (nwp["target_date_local"] <= test_end)].copy()
    model = NWPResidualDistributionModel().fit(train)
    rows = []
    for _, row in test.iterrows():
        actual = float(row["tmax_c"])
        dist = model.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]), row.get("observed_max_so_far_from_metar"))
        rows.append(
            {
                "target_date_local": row["target_date_local"].isoformat(),
                "issue_hour_utc": int(row["issue_hour_utc"]),
                "actual_tmax_c": actual,
                "expected_tmax_c": dist.expected_tmax_c,
                "median_tmax_c": dist.median_tmax_c,
                "nll": nll_integer_bin(dist, actual),
                "crps": crps_discrete(dist, actual),
            }
        )
    scored = pd.DataFrame(rows)
    metrics = {
        "rows": len(scored),
        "mae_expected": mae(scored["actual_tmax_c"], scored["expected_tmax_c"]),
        "rmse_expected": rmse(scored["actual_tmax_c"], scored["expected_tmax_c"]),
        "bias_expected": bias(scored["actual_tmax_c"], scored["expected_tmax_c"]),
        "mean_nll": float(scored["nll"].mean()),
        "mean_crps": float(scored["crps"].mean()),
    }
    version = "nwp_residual_icon_d2_20260531"
    full_model = NWPResidualDistributionModel().fit(nwp)
    metadata = {
        "model_name": "nwp_residual_distribution",
        "model_version": version,
        "training_period": [str(nwp["target_date_local"].min()), str(nwp["target_date_local"].max())],
        "validation_period": ["2026-01-01", "2026-05-27"],
        "feature_set_version": "nwp_residual.v1",
        "source_registry_version": "2026-05-31.nwp_single_runs",
        "data_snapshot_hash": stable_hash({"rows": len(nwp), "target_sum": float(nwp["tmax_c"].sum())}),
        "calibration_version": "empirical_residual_distribution",
        "git_commit": None,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    save_model(full_model, metadata, model_dir="data/models")
    register_artifact(
        version=version,
        artifact_type="model",
        path=Path("data/models") / f"{version}.joblib",
        metadata_path=Path("data/models") / f"{version}.metadata.json",
        metrics=metrics,
    )
    promoted = _passes_promotion(metrics)
    if promoted:
        promote_model(
            model_version=version,
            calibrator_version=None,
            reason="nwp_residual_holdout_gates_passed",
            metrics=metrics,
        )
    report = {"model_version": version, "promoted": promoted, "metrics": metrics}
    Path("data/reports/nwp_residual_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("docs/nwp_residual_model.md").write_text(_doc(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _passes_promotion(metrics: dict) -> bool:
    return (
        metrics["rows"] >= 300
        and metrics["mae_expected"] <= 1.25
        and metrics["rmse_expected"] <= 1.75
        and abs(metrics["bias_expected"]) <= 0.5
        and metrics["mean_nll"] <= 2.0
    )


def _doc(report: dict) -> str:
    metrics = report["metrics"]
    return (
        "# NWP residual model\n\n"
        "Production-compatible probabilistic MOS baseline using ICON-D2 `model_tmax_c` plus empirical residual distributions.\n\n"
        f"- model version: `{report['model_version']}`\n"
        f"- promoted: `{report['promoted']}`\n"
        f"- holdout rows: `{metrics['rows']}`\n"
        f"- MAE expected: `{metrics['mae_expected']}`\n"
        f"- RMSE expected: `{metrics['rmse_expected']}`\n"
        f"- bias expected: `{metrics['bias_expected']}`\n"
        f"- NLL: `{metrics['mean_nll']}`\n"
        f"- CRPS: `{metrics['mean_crps']}`\n"
    )


if __name__ == "__main__":
    main()
