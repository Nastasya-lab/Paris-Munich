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
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
    survival_to_probabilities,
)
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_metar_tmax_upside_v1"


def main() -> None:
    args = _parse_args()
    dataset = pd.read_parquet(args.dataset)
    frame = prepare_metar_tmax_dataset(dataset)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["test_year"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.year
    frame = frame[frame["airport_icao"].fillna(args.airport) == args.airport].copy() if "airport_icao" in frame.columns else frame
    if frame.empty:
        raise ValueError(f"No rows found for {args.airport}")

    scored, fold_inventory = _rolling_year_backtest(frame, args.min_train_rows)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_year = _group_summary(scored, ["model_variant", "test_year"])
    by_season_hour = _group_summary(scored, ["model_variant", "season", "local_issue_hour"])

    final_model = MetarTmaxUpsideModel(min_rows=args.min_train_rows).fit(frame)
    final_calibration_rows, final_calibration_folds = _build_oof_calibration_rows(frame, args.min_train_rows)
    final_calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=final_model.max_upside_c).fit(final_calibration_rows)
    final_model.calibrator = final_calibrator if final_calibrator.fitted else None
    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{MODEL_VERSION}.joblib"
    metadata_path = model_dir / f"{MODEL_VERSION}.metadata.json"
    joblib.dump(final_model, model_path)

    metadata = {
        "model_name": "lfpb_metar_tmax_remaining_upside",
        "model_version": MODEL_VERSION,
        "airport": args.airport,
        "target": "daily maximum temperature reported by METAR, integer-bin probabilistic distribution",
        "training_rows": len(frame),
        "training_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "feature_columns": final_model.feature_columns,
        "max_upside_c": final_model.max_upside_c,
        "fold_inventory": fold_inventory,
        "calibration_rows": len(final_calibration_rows),
        "calibration_folds": final_calibration_folds,
        "calibration_metadata": final_calibrator.to_metadata(),
        "calibration_attached_to_final_model": final_model.calibrator is not None,
        "data_snapshot_hash": stable_hash({"rows": len(frame), "target_sum": float(frame["final_metar_tmax_c"].sum())}),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "rolling_summary": json.loads(summary.to_json(orient="records")),
        "limitations": [
            "Target is METAR Tmax, not official Météo-France TX.",
            "Model is trained on historical IEM METAR and 6-minute precipitation features where available.",
            "TAF and forecast-as-issued NWP are not included in this first LFPB METAR target model.",
            "Backtest is time-based by year; no random split is used.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_metar_tmax_model_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_metar_tmax_model_backtest_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_metar_tmax_model_backtest_by_hour.csv", index=False)
    by_year.to_csv(report_dir / "lfpb_metar_tmax_model_backtest_by_year.csv", index=False)
    by_season_hour.to_csv(report_dir / "lfpb_metar_tmax_model_backtest_by_season_hour.csv", index=False)
    (report_dir / "lfpb_metar_tmax_model_backtest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (report_dir / "lfpb_metar_tmax_model_backtest.md").write_text(
        _markdown_report(metadata, summary, by_hour, by_year),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, default=str))


def _rolling_year_backtest(frame: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    years = sorted(int(year) for year in frame["test_year"].dropna().unique())
    for year in years:
        train = frame[frame["test_year"] < year].copy()
        test = frame[frame["test_year"] == year].copy()
        if len(train) < min_train_rows or test.empty:
            folds.append(
                {
                    "test_year": year,
                    "status": "skipped",
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
            )
            continue
        model = MetarTmaxUpsideModel(min_rows=min_train_rows).fit(train)
        calibrated_model, calibration_fold = _fit_fold_calibrated_model(frame, year, min_train_rows)
        priors = _build_hourly_upside_priors(train)
        ml_survival = model.predict_upside_survival_frame(test)
        calibrated_survival = calibrated_model.predict_upside_survival_frame(test) if calibrated_model else None
        for index, row in test.iterrows():
            ml_dist = _distribution_from_survival(row, ml_survival.loc[index], model.max_upside_c)
            persistence_dist = TmaxDistribution(np.array([int(round(row["current_metar_max_c"]))]), np.array([1.0]))
            phase_prior_dist = _phase_prior_distribution(row, priors, train)
            rows.append(_score("metar_tmax_ml_upside_v1", row, ml_dist, year))
            if calibrated_model and calibrated_survival is not None:
                raw_survival = _survival_dict_from_row(calibrated_survival.loc[index], calibrated_model.max_upside_c)
                calibrated = calibrated_model.calibrator.transform(
                    raw_survival,
                    local_issue_hour=row.get("local_issue_hour"),
                    season=_season(row["target_date_local"]),
                )
                calibrated_dist = _distribution_from_survival_dict(row, calibrated, calibrated_model.max_upside_c)
                rows.append(_score("metar_tmax_ml_upside_v1_calibrated", row, calibrated_dist, year))
            rows.append(_score("metar_tmax_persistence", row, persistence_dist, year))
            rows.append(_score("metar_tmax_hour_phase_prior", row, phase_prior_dist, year))
        folds.append(
            {
                "test_year": year,
                "status": "evaluated",
                "train_rows": len(train),
                "test_rows": len(test),
                "train_start": str(train["target_date_local"].min()),
                "train_end": str(train["target_date_local"].max()),
                "calibration": calibration_fold,
            }
        )
    if not rows:
        raise ValueError("No evaluable LFPB METAR Tmax folds")
    return pd.DataFrame(rows), folds


def _build_oof_calibration_rows(frame: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    years = sorted(int(year) for year in frame["test_year"].dropna().unique())
    for calibration_year in years:
        train = frame[frame["test_year"] < calibration_year].copy()
        calibration = frame[frame["test_year"] == calibration_year].copy()
        if len(train) < min_train_rows or len(calibration) < 500:
            folds.append(
                {
                    "calibration_year": calibration_year,
                    "status": "skipped",
                    "train_rows": len(train),
                    "calibration_rows": len(calibration),
                }
            )
            continue
        model = MetarTmaxUpsideModel(min_rows=min_train_rows).fit(train)
        fold_rows = _survival_calibration_rows(model, calibration)
        rows.append(fold_rows)
        folds.append(
            {
                "calibration_year": calibration_year,
                "status": "evaluated",
                "train_rows": len(train),
                "calibration_rows": len(fold_rows),
            }
        )
    if not rows:
        raise ValueError("No out-of-fold calibration rows for LFPB METAR Tmax model")
    return pd.concat(rows, ignore_index=True), folds


def _distribution_from_survival(row: pd.Series, survival_row: pd.Series, max_upside_c: int) -> TmaxDistribution:
    survival = _survival_dict_from_row(survival_row, max_upside_c)
    return _distribution_from_survival_dict(row, survival, max_upside_c)


def _distribution_from_survival_dict(row: pd.Series, survival: dict[int, float], max_upside_c: int) -> TmaxDistribution:
    probs = survival_to_probabilities(survival, max_upside_c)
    bins = np.rint(float(row["current_metar_max_c"]) + np.arange(max_upside_c + 1)).astype(int)
    return TmaxDistribution(bins, probs)


def _survival_dict_from_row(survival_row: pd.Series, max_upside_c: int) -> dict[int, float]:
    return {
        threshold: float(survival_row[f"probability_upside_ge_{threshold}c"])
        for threshold in range(1, max_upside_c + 1)
    }


def _fit_fold_calibrated_model(
    frame: pd.DataFrame,
    test_year: int,
    min_train_rows: int,
) -> tuple[MetarTmaxUpsideModel | None, dict]:
    train_core = frame[frame["test_year"] < test_year - 1].copy()
    calibration = frame[frame["test_year"] == test_year - 1].copy()
    if len(train_core) < min_train_rows or len(calibration) < 500:
        return None, {
            "status": "skipped",
            "train_core_rows": len(train_core),
            "calibration_rows": len(calibration),
        }
    model = MetarTmaxUpsideModel(min_rows=min_train_rows).fit(train_core)
    calibration_rows = _survival_calibration_rows(model, calibration)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(calibration_rows)
    model.calibrator = calibrator if calibrator.fitted else None
    return model, {
        "status": "evaluated" if calibrator.fitted else "no_calibrator",
        "train_core_rows": len(train_core),
        "calibration_rows": len(calibration_rows),
        "calibration_metadata": calibrator.to_metadata(),
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


def _build_hourly_upside_priors(train: pd.DataFrame) -> dict[tuple[int, str], np.ndarray]:
    priors: dict[tuple[int, str], np.ndarray] = {}
    train = train.copy()
    train["season"] = train["target_date_local"].map(_season)
    for key, group in train.groupby(["local_issue_hour", "season"]):
        priors[(int(key[0]), str(key[1]))] = pd.to_numeric(group["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    return priors


def _phase_prior_distribution(row: pd.Series, priors: dict[tuple[int, str], np.ndarray], train: pd.DataFrame) -> TmaxDistribution:
    hour = int(row["local_issue_hour"])
    season = _season(row["target_date_local"])
    samples = priors.get((hour, season))
    if samples is None or len(samples) < 30:
        samples = pd.to_numeric(train.loc[train["local_issue_hour"] == hour, "remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    if samples is None or len(samples) == 0:
        samples = np.array([0.0])
    rounded = np.rint(float(row["current_metar_max_c"]) + np.clip(samples, 0.0, 12.0)).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probs = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    return TmaxDistribution(bins, probs)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution, test_year: int) -> dict:
    actual = float(row["final_metar_tmax_c"])
    expected = dist.expected_tmax_c
    return {
        "model_variant": model_variant,
        "test_year": test_year,
        "airport_icao": row.get("airport_icao", "LFPB"),
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": _season(row["target_date_local"]),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": float(row["current_metar_max_c"]),
        "remaining_upside_c": float(row["remaining_upside_c"]),
        "expected_tmax_c": expected,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "mae_expected": abs(expected - actual),
        "bias_expected": expected - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(row["current_metar_max_c"] + 1))), bool(row["remaining_upside_c"] >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(row["current_metar_max_c"] + 2))), bool(row["remaining_upside_c"] >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(row["current_metar_max_c"] + 3))), bool(row["remaining_upside_c"] >= 3.0)),
        "covered_80": _covered(dist, actual, 0.80),
        "probability_above_actual_integer_bin": float(dist.probabilities[dist.bins_c > round(actual)].sum()),
    }


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
                "coverage_80": float(group["covered_80"].mean()),
                "mean_false_upside_probability": float(group["probability_above_actual_integer_bin"].mean()),
            }
        )
    return pd.DataFrame(rows)


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


def _markdown_report(metadata: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_year: pd.DataFrame) -> str:
    lines = [
        "# LFPB METAR Tmax model backtest",
        "",
        "This report evaluates the first Paris Le Bourget METAR-target model. The target is daily maximum temperature reported by METAR, not official station TX.",
        "",
        f"- model version: `{metadata['model_version']}`",
        f"- training rows for final artifact: `{metadata['training_rows']}`",
        f"- training period: `{metadata['training_period'][0]}` to `{metadata['training_period'][1]}`",
        f"- folds: `{len(metadata['fold_inventory'])}`",
        "",
        "## Overall",
        "",
        _table(summary),
        "",
        "## By Local Issue Hour",
        "",
        _table(by_hour),
        "",
        "## By Test Year",
        "",
        _table(by_year),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metadata["limitations"])
    lines.append("")
    return "\n".join(lines)


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
    parser = argparse.ArgumentParser(description="Backtest and train LFPB METAR Tmax remaining-upside model.")
    parser.add_argument("--airport", default="LFPB")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB.parquet")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    main()
