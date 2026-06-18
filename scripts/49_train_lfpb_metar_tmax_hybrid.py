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
    MetarTmaxHybridModel,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    mix_distributions,
    prepare_metar_tmax_dataset,
    survival_to_probabilities,
)
from weather_tmax_bot.utils.hashing import stable_hash


MODEL_VERSION = "lfpb_metar_tmax_hybrid_v1"


def main() -> None:
    args = _parse_args()
    frame = prepare_metar_tmax_dataset(pd.read_parquet(args.dataset))
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["test_year"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.year
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["airport_icao"].fillna(args.airport) == args.airport].copy() if "airport_icao" in frame.columns else frame

    scored, folds = _rolling_hybrid_backtest(frame, args.min_train_rows)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_year = _group_summary(scored, ["model_variant", "test_year"])

    final_base = MetarTmaxUpsideModel(min_rows=args.min_train_rows, max_iter=45).fit(frame)
    calibration_rows = _build_oof_calibration_rows(frame, args.min_train_rows)
    final_calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=final_base.max_upside_c).fit(calibration_rows)
    final_base.calibrator = final_calibrator if final_calibrator.fitted else None
    final_priors, final_global_prior = _build_phase_priors(frame)
    final_weight = _select_final_weight(scored)
    hybrid = MetarTmaxHybridModel(
        base_model=final_base,
        phase_priors=final_priors,
        global_prior=final_global_prior,
        blend_weight=final_weight,
        model_version=MODEL_VERSION,
    )

    model_dir = Path("data/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid, model_dir / f"{MODEL_VERSION}.joblib")

    metadata = {
        "model_name": "lfpb_metar_tmax_hybrid_remaining_upside",
        "model_version": MODEL_VERSION,
        "airport": args.airport,
        "target": "daily maximum temperature reported by METAR",
        "training_rows": len(frame),
        "training_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "blend_weight_phase_prior": final_weight,
        "base_model": final_base.feature_columns,
        "calibration_metadata": final_calibrator.to_metadata(),
        "fold_inventory": folds,
        "rolling_summary": json.loads(summary.to_json(orient="records")),
        "data_snapshot_hash": stable_hash({"rows": len(frame), "target_sum": float(frame["final_metar_tmax_c"].sum())}),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "limitations": [
            "IEM historical TAF archive returned zero LFPB rows, so TAF is not used.",
            "Hybrid weight is selected from rolling validation behavior, not from the current live day.",
            "Target is METAR Tmax, not official Meteo-France TX.",
        ],
    }
    (model_dir / f"{MODEL_VERSION}.metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_metar_tmax_hybrid_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_metar_tmax_hybrid_backtest_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_metar_tmax_hybrid_backtest_by_hour.csv", index=False)
    by_year.to_csv(report_dir / "lfpb_metar_tmax_hybrid_backtest_by_year.csv", index=False)
    (report_dir / "lfpb_metar_tmax_hybrid_backtest.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (report_dir / "lfpb_metar_tmax_hybrid_backtest.md").write_text(_markdown(metadata, summary, by_hour, by_year), encoding="utf-8")
    print(json.dumps(metadata, indent=2, default=str))


def _rolling_hybrid_backtest(frame: pd.DataFrame, min_train_rows: int) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    for test_year in sorted(int(year) for year in frame["test_year"].dropna().unique()):
        train_core = frame[frame["test_year"] < test_year - 1].copy()
        calibration = frame[frame["test_year"] == test_year - 1].copy()
        test = frame[frame["test_year"] == test_year].copy()
        if len(train_core) < min_train_rows or len(calibration) < 500 or test.empty:
            folds.append(
                {
                    "test_year": test_year,
                    "status": "skipped",
                    "train_core_rows": len(train_core),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                }
            )
            continue
        base = MetarTmaxUpsideModel(min_rows=min_train_rows, max_iter=45).fit(train_core)
        calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=base.max_upside_c).fit(_survival_calibration_rows(base, calibration))
        base.calibrator = calibrator if calibrator.fitted else None
        priors, global_prior = _build_phase_priors(train_core)
        calibration_predictions = _predict_components(base, priors, global_prior, calibration)
        blend_weight = _optimize_blend_weight(calibration_predictions)
        test_predictions = _predict_components(base, priors, global_prior, test)
        for item in test_predictions:
            row = item["row"]
            rows.append(_score("metar_tmax_ml_calibrated", row, item["base_dist"], test_year))
            rows.append(_score("metar_tmax_phase_prior", row, item["prior_dist"], test_year))
            rows.append(
                _score(
                    "metar_tmax_hybrid_v1",
                    row,
                    mix_distributions(item["base_dist"], item["prior_dist"], blend_weight),
                    test_year,
                    blend_weight=blend_weight,
                )
            )
        folds.append(
            {
                "test_year": test_year,
                "status": "evaluated",
                "train_core_rows": len(train_core),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
                "blend_weight_phase_prior": blend_weight,
            }
        )
    if not rows:
        raise ValueError("No evaluable hybrid folds")
    return pd.DataFrame(rows), folds


def _predict_components(
    base: MetarTmaxUpsideModel,
    priors: dict[str, np.ndarray],
    global_prior: np.ndarray,
    frame: pd.DataFrame,
) -> list[dict]:
    hybrid_for_prior = MetarTmaxHybridModel(base, priors, global_prior, blend_weight=1.0)
    raw_survival = base.predict_upside_survival_frame(frame)
    out = []
    for index, row in frame.iterrows():
        survival = {
            threshold: float(raw_survival.loc[index, f"probability_upside_ge_{threshold}c"])
            for threshold in range(1, base.max_upside_c + 1)
        }
        if base.calibrator is not None:
            survival = base.calibrator.transform(
                survival,
                local_issue_hour=row.get("local_issue_hour"),
                season=_season(row["target_date_local"]),
            )
        out.append(
            {
                "row": row,
                "base_dist": _distribution_from_survival(row, survival, base.max_upside_c),
                "prior_dist": hybrid_for_prior.phase_prior_distribution(row),
            }
        )
    return out


def _distribution_from_survival(row: pd.Series, survival: dict[int, float], max_upside_c: int) -> TmaxDistribution:
    probs = survival_to_probabilities(survival, max_upside_c)
    bins = np.rint(float(row["current_metar_max_c"]) + np.arange(max_upside_c + 1)).astype(int)
    return TmaxDistribution(bins, probs)


def _optimize_blend_weight(predictions: list[dict]) -> float:
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 0.80, 17):
        scores = []
        for item in predictions:
            dist = mix_distributions(item["base_dist"], item["prior_dist"], float(weight))
            scores.append(nll_integer_bin(dist, float(item["row"]["final_metar_tmax_c"])))
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _select_final_weight(scored: pd.DataFrame) -> float:
    hybrid = scored[scored["model_variant"] == "metar_tmax_hybrid_v1"]
    if hybrid.empty:
        return 0.35
    by_year = hybrid.groupby("test_year")["blend_weight_phase_prior"].first().dropna()
    return float(np.median(by_year.to_numpy(dtype=float))) if not by_year.empty else 0.35


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


def _build_oof_calibration_rows(frame: pd.DataFrame, min_train_rows: int) -> pd.DataFrame:
    rows = []
    for calibration_year in sorted(int(year) for year in frame["test_year"].dropna().unique()):
        train = frame[frame["test_year"] < calibration_year].copy()
        calibration = frame[frame["test_year"] == calibration_year].copy()
        if len(train) < min_train_rows or len(calibration) < 500:
            continue
        rows.append(_survival_calibration_rows(MetarTmaxUpsideModel(min_rows=min_train_rows, max_iter=45).fit(train), calibration))
    if not rows:
        raise ValueError("No OOF calibration rows for final hybrid")
    return pd.concat(rows, ignore_index=True)


def _build_phase_priors(frame: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    priors: dict[str, np.ndarray] = {}
    data = frame.copy()
    data["season"] = data["target_date_local"].map(_season)
    for (hour, season), group in data.groupby(["local_issue_hour", "season"], dropna=True):
        priors[_context_key(hour, season)] = pd.to_numeric(group["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    for hour, group in data.groupby("local_issue_hour", dropna=True):
        priors[_context_key(hour, "all")] = pd.to_numeric(group["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    global_prior = pd.to_numeric(data["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
    return priors, global_prior


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution, test_year: int, blend_weight: float | None = None) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "test_year": test_year,
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
        "blend_weight_phase_prior": np.nan if blend_weight is None else float(blend_weight),
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
                "coverage_80": float(group["coverage_80"].mean()),
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


def _context_key(local_issue_hour, season) -> str:
    return f"{int(float(local_issue_hour))}|{season}"


def _markdown(metadata: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_year: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB METAR Tmax hybrid backtest",
            "",
            "Hybrid candidate blending calibrated ML remaining-upside distribution with empirical hourly/seasonal phase prior.",
            "",
            f"- model version: `{metadata['model_version']}`",
            f"- final phase-prior blend weight: `{metadata['blend_weight_phase_prior']:.2f}`",
            f"- training rows: `{metadata['training_rows']}`",
            "",
            "## Overall",
            "",
            _table(summary),
            "",
            "## By local issue hour",
            "",
            _table(by_hour),
            "",
            "## By test year",
            "",
            _table(by_year),
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
    parser = argparse.ArgumentParser(description="Train/backtest LFPB METAR Tmax hybrid model.")
    parser.add_argument("--airport", default="LFPB")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB.parquet")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    return parser.parse_args()


if __name__ == "__main__":
    main()
