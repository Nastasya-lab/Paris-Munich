from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import typer

from weather_tmax_bot.evaluation.leakage_audit import audit_training_dataset
from weather_tmax_bot.evaluation.quantile_backtest import holdout_quantile_backtest
from weather_tmax_bot.models.model_registry import promote_model
from weather_tmax_bot.models.train import train_quantile_model


def main(
    airport: str = typer.Option("EDDM"),
    dataset_path: str = typer.Option("data/processed/training_dataset.parquet"),
    model_version: str | None = typer.Option(None),
    auto_promote: bool = typer.Option(True),
    min_rows: int = typer.Option(1000),
):
    dataset = pd.read_parquet(dataset_path)
    if "leakage_check_passed" in dataset.columns:
        dataset = dataset[dataset["leakage_check_passed"] == True].copy()
    if "airport_icao" in dataset.columns:
        dataset = dataset[dataset["airport_icao"] == airport].copy()

    version = model_version or f"quantile_mvp_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    leakage_report, leakage_passed = audit_training_dataset(dataset)
    _, metrics = holdout_quantile_backtest(dataset)
    gates = _quality_gates(dataset, leakage_passed, metrics, min_rows=min_rows)

    train_quantile_model(dataset, model_version=version)

    promoted = False
    if auto_promote and all(gates.values()):
        promote_model(
            model_version=version,
            calibrator_version=f"{version}.calibrator",
            reason="automatic_retraining_gates_passed",
            metrics=metrics["calibrated_spread"],
        )
        promoted = True

    report = {
        "airport": airport,
        "model_version": version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "rows": len(dataset),
        "leakage_passed": leakage_passed,
        "quality_gates": gates,
        "promoted": promoted,
        "metrics": metrics,
        "leakage_report": leakage_report.to_dict(orient="records"),
    }
    _write_reports(report)
    print(json.dumps({"model_version": version, "promoted": promoted, "quality_gates": gates}, indent=2))


def _quality_gates(dataset: pd.DataFrame, leakage_passed: bool, metrics: dict, min_rows: int) -> dict[str, bool]:
    calibrated = metrics["calibrated_spread"]
    return {
        "min_rows": len(dataset) >= min_rows,
        "leakage_passed": leakage_passed,
        "finite_nll": pd.notna(calibrated["mean_nll"]) and calibrated["mean_nll"] < 20,
        "coverage_80_reasonable": 0.65 <= calibrated["coverage_80"] <= 0.95,
        "coverage_90_reasonable": 0.75 <= calibrated["coverage_90"] <= 0.99,
        "crps_reasonable": pd.notna(calibrated["mean_crps"]) and calibrated["mean_crps"] < 5,
    }


def _write_reports(report: dict) -> None:
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "retraining_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    metrics = report["metrics"]["calibrated_spread"]
    Path("docs/retraining_report.md").write_text(
        "# Retraining report\n\n"
        f"Model version: `{report['model_version']}`\n\n"
        f"Rows: {report['rows']}\n\n"
        f"Promoted: {report['promoted']}\n\n"
        "Quality gates:\n\n"
        + "\n".join(f"- `{key}`: {value}" for key, value in report["quality_gates"].items())
        + "\n\n"
        "Calibrated spread holdout metrics:\n\n"
        f"- MAE median: {metrics['mae_median']}\n"
        f"- RMSE mean: {metrics['rmse_mean']}\n"
        f"- NLL: {metrics['mean_nll']}\n"
        f"- CRPS: {metrics['mean_crps']}\n"
        f"- Coverage 50/80/90: {metrics['coverage_50']} / {metrics['coverage_80']} / {metrics['coverage_90']}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    typer.run(main)
