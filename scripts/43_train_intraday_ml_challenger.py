from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.intraday_ml import (
    EDDM_SPATIAL_METAR_FEATURES,
    ENHANCED_METAR_INTRADAY_FEATURES,
    IntradayMLSurvivalCalibrator,
    IntradayMLUpsideModel,
    infer_intraday_ml_context,
    prepare_intraday_ml_dataset,
)
from weather_tmax_bot.utils.hashing import stable_hash

VERSION = "intraday_ml_core_challenger_v1"
MODEL_PATH = Path("data/models") / f"{VERSION}.joblib"
METADATA_PATH = Path("data/models") / f"{VERSION}.metadata.json"
MODEL_MAX_ITER = 30
MODEL_MAX_UPSIDE_C = 12


def main() -> None:
    source = Path("data/processed/intraday_ml_dataset_enhanced_spatial.parquet")
    if not source.exists():
        source = Path("data/processed/intraday_ml_dataset_enhanced.parquet")
    if not source.exists():
        source = Path("data/processed/intraday_ml_dataset.parquet")
    if source.exists():
        dataset = pd.read_parquet(source)
    else:
        dataset = prepare_intraday_ml_dataset(pd.read_parquet("data/processed/training_dataset.parquet"))
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    usable = dataset[dataset["target_date_local"] <= pd.to_datetime("2025-12-30").date()].copy()
    scored, fold_inventory = _rolling_backtest(usable)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "issue_hour_utc"])
    by_fold = _group_summary(scored, ["model_variant", "fold_start"])
    calibration_rows, calibration_folds = _build_oof_calibration_rows(usable)
    final_calibrator = IntradayMLSurvivalCalibrator(max_upside_c=MODEL_MAX_UPSIDE_C).fit(calibration_rows)
    calibration_deployment = _calibration_deployment_decision(summary)
    model = IntradayMLUpsideModel(max_iter=MODEL_MAX_ITER, max_upside_c=MODEL_MAX_UPSIDE_C).fit(usable)
    model.calibrator = final_calibrator if calibration_deployment["accepted"] else None
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model_name": "ordinal_intraday_remaining_upside",
        "model_version": VERSION,
        "mode": "late_day_production_component",
        "training_source": str(source),
        "training_period": [str(usable["target_date_local"].min()), str(usable["target_date_local"].max())],
        "training_rows": len(usable),
        "calibration_version": "intraday_ml_contextual_survival_oof_v2",
        "calibration_rows": len(calibration_rows),
        "calibration_metadata": final_calibrator.to_metadata(),
        "calibration_deployment": calibration_deployment,
        "calibration_folds": calibration_folds,
        "feature_set_version": "intraday_ml_core.enhanced_metar.spatial_edmo_edma_etsi_etsl.v3",
        "feature_columns": model.feature_columns,
        "enhanced_intraday_feature_columns": ENHANCED_METAR_INTRADAY_FEATURES,
        "spatial_feature_columns": EDDM_SPATIAL_METAR_FEATURES,
        "spatial_neighbor_stations": ["EDMO", "EDMA", "ETSI", "ETSL"],
        "max_iter": MODEL_MAX_ITER,
        "max_upside_c": MODEL_MAX_UPSIDE_C,
        "data_snapshot_hash": stable_hash({"rows": len(usable), "target_sum": float(usable["tmax_c"].sum())}),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rolling_backtest": json.loads(summary.to_json(orient="records")),
        "folds": fold_inventory,
        "limitations": [
            "Historical TAF archive is empty, so TAF values degrade to missing flags during training.",
            "ICON-D2 features are optional because honest historical NWP overlap starts in late May 2025.",
            "Calibration is learned from historical out-of-fold predictions and deployed only if it passes the gate.",
            "EDDM spatial METAR neighbors are used as as-of features when available; missing live neighbors degrade through the model imputer.",
            "This artifact is promoted only through late-day phase arbitration, not as the full-day base model.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    write_parquet(scored, "data/reports/intraday_ml_rolling_rows.parquet")
    write_parquet(summary, "data/reports/intraday_ml_rolling_summary.parquet")
    write_parquet(by_hour, "data/reports/intraday_ml_rolling_by_hour.parquet")
    write_parquet(by_fold, "data/reports/intraday_ml_rolling_by_fold.parquet")
    Path("data/reports/intraday_ml_training_report.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    Path("docs/intraday_ml_challenger.md").write_text(_doc(metadata, by_hour, by_fold), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


def _rolling_backtest(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    for fold_start in pd.date_range("2025-08-01", "2025-12-01", freq="MS").date:
        fold_end = (pd.Timestamp(fold_start) + pd.offsets.MonthEnd(1)).date()
        calibration_start = (pd.Timestamp(fold_start) - pd.Timedelta(days=90)).date()
        train_core = dataset[dataset["target_date_local"] < calibration_start].copy()
        calibration = dataset[
            (dataset["target_date_local"] >= calibration_start) & (dataset["target_date_local"] < fold_start)
        ].copy()
        train = dataset[dataset["target_date_local"] < fold_start].copy()
        test = dataset[(dataset["target_date_local"] >= fold_start) & (dataset["target_date_local"] <= fold_end)].copy()
        if len(train_core) < 300 or len(calibration) < 100 or test.empty:
            folds.append({
                "fold_start": fold_start.isoformat(),
                "fold_end": fold_end.isoformat(),
                "status": "skipped",
                "train_core_rows": len(train_core),
                "calibration_rows": len(calibration),
                "train_rows": len(train),
                "test_rows": len(test),
            })
            continue
        model = IntradayMLUpsideModel(max_iter=MODEL_MAX_ITER, max_upside_c=MODEL_MAX_UPSIDE_C).fit(train_core)
        calibrator = IntradayMLSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(
            _survival_calibration_rows(model, calibration)
        )
        raw_predictions = _predict_distributions_frame(model, test, calibrator=None)
        calibrated_predictions = _predict_distributions_frame(model, test, calibrator=calibrator)
        for (_, row), raw_prediction, calibrated_prediction in zip(test.iterrows(), raw_predictions, calibrated_predictions):
            baseline_dist, baseline_details = _empirical_upside_distribution(train, row)
            dist, details = raw_prediction
            rows.append(_score("intraday_ml_core_challenger_v1_raw", row, dist, details, fold_start))
            calibrated_dist, calibrated_details = calibrated_prediction
            rows.append(_score("intraday_ml_core_challenger_v1", row, calibrated_dist, calibrated_details, fold_start))
            rows.append(_score("empirical_hourly_upside_baseline", row, baseline_dist, baseline_details, fold_start))
        folds.append({
            "fold_start": fold_start.isoformat(),
            "fold_end": fold_end.isoformat(),
            "status": "evaluated",
            "train_core_rows": len(train_core),
            "calibration_rows": len(calibration),
            "train_rows": len(train),
            "test_rows": len(test),
            "calibrated_thresholds": calibrator.to_metadata()["calibrated_thresholds"],
        })
    if not rows:
        raise ValueError("intraday ML rolling backtest found no evaluable folds")
    return pd.DataFrame(rows), folds


def _build_oof_calibration_rows(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    for fold_start in pd.date_range("2024-03-01", "2025-12-01", freq="3MS").date:
        fold_end = (pd.Timestamp(fold_start) + pd.offsets.MonthEnd(1)).date()
        train = dataset[dataset["target_date_local"] < fold_start].copy()
        holdout = dataset[(dataset["target_date_local"] >= fold_start) & (dataset["target_date_local"] <= fold_end)].copy()
        if len(train) < 300 or holdout.empty:
            folds.append({
                "fold_start": fold_start.isoformat(),
                "fold_end": fold_end.isoformat(),
                "status": "skipped",
                "train_rows": len(train),
                "holdout_rows": len(holdout),
            })
            continue
        model = IntradayMLUpsideModel(max_iter=MODEL_MAX_ITER, max_upside_c=MODEL_MAX_UPSIDE_C).fit(train)
        frame = _survival_calibration_rows(model, holdout)
        rows.append(frame)
        folds.append({
            "fold_start": fold_start.isoformat(),
            "fold_end": fold_end.isoformat(),
            "status": "evaluated",
            "train_rows": len(train),
            "holdout_rows": len(holdout),
        })
    if not rows:
        raise ValueError("could not build out-of-fold intraday ML calibration rows")
    return pd.concat(rows, ignore_index=True), folds


def _survival_calibration_rows(model: IntradayMLUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    survival_frame = model.predict_upside_survival_frame(frame)
    for idx, row in frame.iterrows():
        survival = survival_frame.loc[idx]
        remaining_upside = float(row["remaining_upside_c"])
        out = {
            "target_date_local": row["target_date_local"].isoformat(),
            "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
            "issue_hour_utc": int(row["issue_hour_utc"]),
            "remaining_upside_c": remaining_upside,
        }
        out.update(infer_intraday_ml_context(row))
        for threshold in range(1, model.max_upside_c + 1):
            out[f"raw_probability_upside_ge_{threshold}c"] = float(survival.loc[threshold])
            out[f"actual_upside_ge_{threshold}c"] = float(remaining_upside >= threshold)
        rows.append(out)
    return pd.DataFrame(rows)


def _predict_distributions_frame(
    model: IntradayMLUpsideModel,
    frame: pd.DataFrame,
    *,
    calibrator: IntradayMLSurvivalCalibrator | None,
) -> list[tuple[TmaxDistribution, dict]]:
    survival_frame = model.predict_upside_survival_frame(frame)
    predictions = []
    for idx, row in frame.iterrows():
        raw_survival = {
            threshold: float(survival_frame.loc[idx, threshold])
            for threshold in range(1, model.max_upside_c + 1)
        }
        survival = (
            calibrator.transform(
                raw_survival,
                issue_hour_utc=row.get("issue_hour_utc"),
                context=infer_intraday_ml_context(row),
            )
            if calibrator is not None
            else raw_survival
        )
        probs = _survival_to_probabilities(survival, model.max_upside_c)
        observed_max = float(row["observed_max_so_far_from_metar"])
        bins = np.rint(observed_max + np.arange(model.max_upside_c + 1)).astype(int)
        predictions.append(
            (
                TmaxDistribution(bins, probs),
                {
                    "probability_peak_already_passed": float(probs[0]),
                    "probability_upside_ge_1c": survival[1],
                    "probability_upside_ge_2c": survival[2],
                    "probability_upside_ge_3c": survival[3],
                },
            )
        )
    return predictions


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution, details: dict, fold_start) -> dict:
    actual = float(row["tmax_c"])
    return {
        "model_variant": model_variant,
        "fold_start": fold_start.isoformat(),
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_peak_already_passed": brier(details["probability_peak_already_passed"], bool(row["peak_already_passed"])),
        "brier_upside_ge_1c": brier(details["probability_upside_ge_1c"], bool(row["upside_ge_1c"])),
        "brier_upside_ge_2c": brier(details["probability_upside_ge_2c"], bool(row["upside_ge_2c"])),
        "brier_upside_ge_3c": brier(details["probability_upside_ge_3c"], bool(row["upside_ge_3c"])),
        "covered_80": _covered(dist, actual, 0.80),
        "probability_above_actual_integer_bin": float(dist.probabilities[dist.bins_c > round(actual)].sum()),
    }


def _empirical_upside_distribution(train: pd.DataFrame, row: pd.Series) -> tuple[TmaxDistribution, dict]:
    candidates = train[train["issue_hour_utc"] == row["issue_hour_utc"]]
    if len(candidates) < 100:
        candidates = train
    observed_max = float(row["observed_max_so_far_from_metar"])
    upside = pd.to_numeric(candidates["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    rounded = np.rint(observed_max + upside).astype(int)
    bins = np.arange(rounded.min(), rounded.max() + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    dist = TmaxDistribution(bins, probabilities)
    return dist, {
        "probability_peak_already_passed": float((upside < 0.5).mean()),
        "probability_upside_ge_1c": float((upside >= 1).mean()),
        "probability_upside_ge_2c": float((upside >= 2).mean()),
        "probability_upside_ge_3c": float((upside >= 3).mean()),
    }


def _survival_to_probabilities(survival: dict[int, float], max_upside_c: int) -> np.ndarray:
    survival_values = np.array([survival[threshold] for threshold in range(1, max_upside_c + 1)], dtype=float)
    probs = np.empty(max_upside_c + 1, dtype=float)
    probs[0] = 1.0 - survival_values[0]
    probs[1:-1] = survival_values[:-1] - survival_values[1:]
    probs[-1] = survival_values[-1]
    return np.clip(probs, 0.0, 1.0)


def _summary(scored: pd.DataFrame) -> dict:
    return {
        "rows": len(scored),
        "distinct_target_days": int(scored["target_date_local"].nunique()),
        "mae_expected": mae(scored["actual_tmax_c"], scored["expected_tmax_c"]),
        "rmse_expected": rmse(scored["actual_tmax_c"], scored["expected_tmax_c"]),
        "bias_expected": float((scored["expected_tmax_c"] - scored["actual_tmax_c"]).mean()),
        "mean_nll": float(scored["nll"].mean()),
        "mean_crps": float(scored["crps"].mean()),
        "brier_peak_already_passed": float(scored["brier_peak_already_passed"].mean()),
        "brier_upside_ge_1c": float(scored["brier_upside_ge_1c"].mean()),
        "brier_upside_ge_2c": float(scored["brier_upside_ge_2c"].mean()),
        "brier_upside_ge_3c": float(scored["brier_upside_ge_3c"].mean()),
        "coverage_80": float(scored["covered_80"].mean()),
        "mean_false_upside_probability": float(scored["probability_above_actual_integer_bin"].mean()),
    }


def _calibration_deployment_decision(summary: pd.DataFrame) -> dict:
    indexed = summary.set_index("model_variant")
    raw = indexed.loc["intraday_ml_core_challenger_v1_raw"]
    calibrated = indexed.loc["intraday_ml_core_challenger_v1"]
    checks = {
        "nll_not_worse": float(calibrated["mean_nll"]) <= float(raw["mean_nll"]),
        "coverage_80_not_worse": float(calibrated["coverage_80"]) >= float(raw["coverage_80"]),
        "mae_degradation_within_25pct": float(calibrated["mae_expected"]) <= 1.25 * float(raw["mae_expected"]),
        "crps_degradation_within_20pct": float(calibrated["mean_crps"]) <= 1.20 * float(raw["mean_crps"]),
        "false_upside_increase_within_8pp": float(calibrated["mean_false_upside_probability"])
        <= float(raw["mean_false_upside_probability"]) + 0.08,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "raw": {key: float(raw[key]) for key in ("mae_expected", "mean_nll", "mean_crps", "coverage_80", "mean_false_upside_probability")},
        "calibrated": {
            key: float(calibrated[key])
            for key in ("mae_expected", "mean_nll", "mean_crps", "coverage_80", "mean_false_upside_probability")
        },
    }


def _group_summary(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(columns, keys)), **_summary(group)})
    return pd.DataFrame(rows)


def _covered(dist, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _doc(metadata: dict, by_hour: pd.DataFrame, by_fold: pd.DataFrame) -> str:
    summary = metadata["rolling_backtest"]
    lines = [
        "# Intraday ML challenger",
        "",
        "Late-day ordinal remaining-upside component. It predicts a monotonic survival curve for future Tmax increases and converts that curve into integer-bin probabilities.",
        "",
        f"- model version: `{metadata['model_version']}`",
        f"- training rows: `{metadata['training_rows']}`",
        f"- calibration rows: `{metadata['calibration_rows']}`",
        f"- calibration version: `{metadata['calibration_version']}`",
        f"- calibration deployment accepted: `{metadata['calibration_deployment']['accepted']}`",
        f"- calibration context count: `{metadata['calibration_metadata'].get('context_count', 0)}`",
        f"- training period: `{metadata['training_period'][0]}` to `{metadata['training_period'][1]}`",
        f"- calibrated thresholds: `{metadata['calibration_metadata']['calibrated_thresholds']}`",
        "## Rolling comparison",
        "",
        _table(pd.DataFrame(summary)),
        "",
        "## By issue hour",
        "",
        _table(by_hour),
        "",
        "## By fold",
        "",
        _table(by_fold),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metadata["limitations"])
    lines.extend(["", "## Decision", "", "Keep this model in shadow mode until historical comparison and forward outcomes are reviewed.", ""])
    return "\n".join(lines)


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
