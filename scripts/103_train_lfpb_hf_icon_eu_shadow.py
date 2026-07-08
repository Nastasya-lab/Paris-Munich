from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

from weather_tmax_bot.evaluation.metrics import crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import unimodal_violation_count
from weather_tmax_bot.models.hf_icon_eu_shadow import (
    HfIconEuResidualPmfShadowModel,
    prepare_hf_icon_eu_training_frame,
)
from weather_tmax_bot.models.model_registry import register_artifact
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_hf_icon_eu_residual_pmf_shadow_v1"


def main() -> None:
    args = _parse_args()
    frame = _load_frame(args)
    if len(frame) < args.min_rows:
        raise SystemExit(f"Need at least {args.min_rows} HF ICON-EU rows, got {len(frame)}")
    train, calibration, test, split = _time_split(frame)
    train_calibration = pd.concat([train, calibration], ignore_index=True)
    holdout_model = HfIconEuResidualPmfShadowModel(
        model_version=MODEL_VERSION,
        minimum_run_hour_samples=args.minimum_run_hour_samples,
    ).fit(train_calibration)
    scored = _score_holdout(test, holdout_model)
    summary = _summary(scored, ["model_variant"])
    by_run_hour = _summary(scored, ["model_variant", "model_run_hour"])
    metrics = summary.iloc[0].to_dict()

    final_model = HfIconEuResidualPmfShadowModel(
        model_version=MODEL_VERSION,
        minimum_run_hour_samples=args.minimum_run_hour_samples,
    ).fit(frame)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_VERSION}.joblib"
    metadata_path = model_dir / f"{MODEL_VERSION}.metadata.json"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_name": "lfpb_hf_icon_eu_residual_pmf_shadow",
        "model_version": MODEL_VERSION,
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "role": "shadow_diagnostic",
        "training_source": "openclimatefix/dwd-icon-eu nearest LFPB grid point",
        "runtime_source_note": "Runtime may use Open-Meteo ICON-EU live aggregates when HF archive is not current.",
        "feature_set_version": "lfpb.hf_icon_eu.minimal_residual_pmf.v1",
        "included_run_hours": sorted(int(hour) for hour in frame["model_run_hour"].unique()),
        "usable_rows": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "holdout_metrics": metrics,
        "by_run_hour": json.loads(by_run_hour.to_json(orient="records")),
        "model_metadata": final_model.to_metadata(),
        "limitations": [
            "Shadow only; does not affect production probabilities or trading unless explicitly promoted later.",
            "HF archive is historical and not current for 2026; live usage relies on compatible ICON-EU runtime features.",
            "Minimal variable set uses temperature-derived features only in v1.",
            "18Z is excluded by default because previous backtests showed weaker same-day behavior.",
        ],
        "data_snapshot_hash": stable_hash(
            {
                "rows": len(frame),
                "target_sum": float(frame["metar_tmax_c"].sum()),
                "model_tmax_sum": float(frame["model_tmax_c"].sum()),
                "run_hours": sorted(int(hour) for hour in frame["model_run_hour"].unique()),
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
    scored.to_parquet(report_dir / "lfpb_hf_icon_eu_shadow_holdout_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_hf_icon_eu_shadow_holdout_summary.csv", index=False)
    by_run_hour.to_csv(report_dir / "lfpb_hf_icon_eu_shadow_holdout_by_run_hour.csv", index=False)
    (report_dir / "lfpb_hf_icon_eu_shadow_training.json").write_text(
        json.dumps(metadata, indent=2, default=str),
        encoding="utf-8",
    )
    Path(args.doc_path).write_text(_markdown(metadata, summary, by_run_hour, scored), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


def _load_frame(args: argparse.Namespace) -> pd.DataFrame:
    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(args.input_glob)
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    target = pd.read_parquet(args.target)
    frame = prepare_hf_icon_eu_training_frame(raw, target)
    allowed = {int(hour) for hour in args.run_hour}
    frame = frame[frame["model_run_hour"].astype(int).isin(allowed)].copy()
    return frame.sort_values(["target_date_local", "model_run_hour"]).reset_index(drop=True)


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
        "train_days": int(train["target_date_local"].nunique()),
        "calibration_days": int(calibration["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
    }


def _score_holdout(test: pd.DataFrame, model: HfIconEuResidualPmfShadowModel) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        actual = float(row["metar_tmax_c"])
        distribution = model.predict_distribution(row)
        rows.append(
            {
                "model_variant": MODEL_VERSION,
                "target_date_local": str(row["target_date_local"]),
                "model_run_hour": int(row["model_run_hour"]),
                "model_tmax_c": float(row["model_tmax_c"]),
                "actual_metar_tmax_c": actual,
                "expected_tmax_c": distribution.expected_tmax_c,
                "most_likely_integer_c": distribution.most_likely_integer_c,
                "mae_expected": abs(distribution.expected_tmax_c - actual),
                "bias_expected": distribution.expected_tmax_c - actual,
                "mode_hit": bool(abs(distribution.most_likely_integer_c - actual) < 0.5),
                "mode_error_ge_2c": bool(abs(distribution.most_likely_integer_c - actual) >= 2.0),
                "nll": nll_integer_bin(distribution, actual),
                "crps": crps_discrete(distribution, actual),
                "shape_violations": unimodal_violation_count(distribution),
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
                "mode_hit_rate": float(group["mode_hit"].mean()),
                "mode_error_ge_2c_rate": float(group["mode_error_ge_2c"].mean()),
                "mean_shape_violations": float(group["shape_violations"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _markdown(metadata: dict, summary: pd.DataFrame, by_run_hour: pd.DataFrame, scored: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB HF ICON-EU Shadow Training",
            "",
            f"- created: `{metadata['created_at_utc']}`",
            f"- model version: `{metadata['model_version']}`",
            f"- role: `{metadata['role']}`",
            f"- period: `{metadata['target_period'][0]}`..`{metadata['target_period'][1]}`",
            f"- rows: `{metadata['usable_rows']}`",
            f"- run hours: `{metadata['included_run_hours']}`",
            "",
            "## Holdout Summary",
            "",
            "```csv",
            summary.to_csv(index=False),
            "```",
            "",
            "## By Run Hour",
            "",
            "```csv",
            by_run_hour.to_csv(index=False),
            "```",
            "",
            "## Worst Rows",
            "",
            "```csv",
            scored.sort_values("mae_expected", ascending=False).head(25).to_csv(index=False),
            "```",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LFPB HF ICON-EU residual PMF shadow model.")
    parser.add_argument("--input-glob", default="data/reports/hf_icon_eu_lfpb_*z_minimal_cache.parquet")
    parser.add_argument("--target", default="data/processed/metar_tmax_target_LFPB.parquet")
    parser.add_argument("--run-hour", type=int, action="append", default=[0, 6, 12])
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--minimum-run-hour-samples", type=int, default=30)
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--doc-path", default="docs/lfpb_hf_icon_eu_shadow_training.md")
    return parser.parse_args()


if __name__ == "__main__":
    main()
