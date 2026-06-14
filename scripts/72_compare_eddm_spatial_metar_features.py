from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.spatial_metar import add_spatial_metar_features_to_frame, spatial_feature_columns
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.intraday_ml import (
    CORE_INTRADAY_ML_FEATURES,
    ENHANCED_METAR_INTRADAY_FEATURES,
    IntradayMLSurvivalCalibrator,
    IntradayMLUpsideModel,
    infer_intraday_ml_context,
)


AIRPORT = "EDDM"
TIMEZONE_NAME = "Europe/Berlin"
NEIGHBOR_STATIONS = ["EDMO", "EDMA", "ETSI", "ETSL"]

def main() -> None:
    dataset = _load_or_build_dataset()
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    usable = dataset[dataset["target_date_local"] <= pd.to_datetime("2025-12-30").date()].copy()
    base_features = list(CORE_INTRADAY_ML_FEATURES) + list(ENHANCED_METAR_INTRADAY_FEATURES)
    spatial_features = spatial_feature_columns(NEIGHBOR_STATIONS)
    candidate_features = base_features + spatial_features
    train, calibration, test, split = _time_split(usable)
    base_model = _fit_calibrated(train, calibration, base_features)
    candidate_model = _fit_calibrated(train, calibration, candidate_features)
    scored = _score_holdout(test, base_model, candidate_model)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "issue_hour_utc"])
    by_availability = _group_summary(scored, ["model_variant", "spatial_available_station_count"])
    base = _row(summary, "enhanced_intraday_ml")
    candidate = _row(summary, "enhanced_spatial_metar_ml")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": AIRPORT,
        "experiment": "enhanced Munich intraday ML vs neighbor METAR spatial context",
        "method": "fast_chronological_gate_calibrated_ordinal_max_upside_12_max_iter_18",
        "target": "official DWD daily Tmax; spatial features use only as-of neighbor METAR",
        "dataset_rows": len(dataset),
        "usable_rows": len(usable),
        "days": int(usable["target_date_local"].nunique()),
        "period": [str(usable["target_date_local"].min()), str(usable["target_date_local"].max())],
        "split": split,
        "base_feature_count": len(base_features),
        "candidate_feature_count": len(candidate_features),
        "neighbor_stations": NEIGHBOR_STATIONS,
        "spatial_feature_columns": spatial_features,
        "neighbor_coverage": _neighbor_coverage(usable),
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, candidate, by_hour),
    }
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored, report_dir / "eddm_spatial_metar_feature_comparison_rows.parquet")
    summary.to_csv(report_dir / "eddm_spatial_metar_feature_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "eddm_spatial_metar_feature_comparison_by_hour.csv", index=False)
    by_availability.to_csv(report_dir / "eddm_spatial_metar_feature_comparison_by_availability.csv", index=False)
    (report_dir / "eddm_spatial_metar_feature_comparison.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/eddm_spatial_metar_feature_comparison.md").write_text(
        _markdown(report, summary, by_hour, by_availability),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _load_or_build_dataset() -> pd.DataFrame:
    output = Path("data/processed/intraday_ml_dataset_enhanced_spatial.parquet")
    if output.exists():
        return pd.read_parquet(output)
    source = Path("data/processed/intraday_ml_dataset_enhanced.parquet")
    if not source.exists():
        raise FileNotFoundError("Run scripts/68_compare_eddm_intraday_enhanced_features.py first")
    frame = pd.read_parquet(source)
    neighbors = {
        station: pd.read_parquet(f"data/interim/metar_iem_{station}.parquet")
        for station in NEIGHBOR_STATIONS
    }
    out = add_spatial_metar_features_to_frame(
        frame,
        neighbors,
        timezone_name=TIMEZONE_NAME,
        stations=NEIGHBOR_STATIONS,
    )
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool) & out[
        "spatial_leakage_check_passed"
    ].fillna(False).astype(bool)
    write_parquet(out, output)
    return out


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
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


def _fit_calibrated(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str]) -> IntradayMLUpsideModel:
    model = IntradayMLUpsideModel(feature_columns=features, max_iter=18, max_upside_c=12).fit(train)
    calibrator = IntradayMLSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(
        _survival_calibration_rows(model, calibration)
    )
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _survival_calibration_rows(model: IntradayMLUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    survival_frame = model.predict_upside_survival_frame(frame)
    for idx, row in frame.iterrows():
        survival = survival_frame.loc[idx]
        remaining_upside = float(row["remaining_upside_c"])
        out = {
            "target_date_local": str(row["target_date_local"]),
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


def _score_holdout(
    test: pd.DataFrame,
    base_model: IntradayMLUpsideModel,
    candidate_model: IntradayMLUpsideModel,
) -> pd.DataFrame:
    rows = []
    base_predictions = _predict_distributions_frame(base_model, test)
    candidate_predictions = _predict_distributions_frame(candidate_model, test)
    for (idx, row), base_prediction, candidate_prediction in zip(test.iterrows(), base_predictions, candidate_predictions):
        rows.append(_score("enhanced_intraday_ml", row, base_prediction))
        rows.append(_score("enhanced_spatial_metar_ml", row, candidate_prediction))
    return pd.DataFrame(rows)


def _predict_distributions_frame(model: IntradayMLUpsideModel, frame: pd.DataFrame) -> list[tuple[TmaxDistribution, dict]]:
    survival_frame = model.predict_upside_survival_frame(frame)
    predictions = []
    for idx, row in frame.iterrows():
        raw_survival = {
            threshold: float(survival_frame.loc[idx, threshold])
            for threshold in range(1, model.max_upside_c + 1)
        }
        survival = (
            model.calibrator.transform(
                raw_survival,
                issue_hour_utc=row.get("issue_hour_utc"),
                context=infer_intraday_ml_context(row),
            )
            if model.calibrator is not None
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


def _survival_to_probabilities(survival: dict[int, float], max_upside_c: int) -> np.ndarray:
    survival_values = np.array([survival[threshold] for threshold in range(1, max_upside_c + 1)], dtype=float)
    probs = np.empty(max_upside_c + 1, dtype=float)
    probs[0] = 1.0 - survival_values[0]
    for idx in range(1, max_upside_c):
        probs[idx] = survival_values[idx - 1] - survival_values[idx]
    probs[-1] = survival_values[-1]
    return np.clip(probs, 0.0, 1.0)


def _score(model_variant: str, row: pd.Series, prediction: tuple[TmaxDistribution, dict]) -> dict:
    dist, details = prediction
    actual = float(row["tmax_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "spatial_available_station_count": int(row.get("spatial_available_station_count", 0)),
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_peak_already_passed": brier(details["probability_peak_already_passed"], bool(row["peak_already_passed"])),
        "brier_upside_ge_1c": brier(details["probability_upside_ge_1c"], bool(row["upside_ge_1c"])),
        "brier_upside_ge_2c": brier(details["probability_upside_ge_2c"], bool(row["upside_ge_2c"])),
        "brier_upside_ge_3c": brier(details["probability_upside_ge_3c"], bool(row["upside_ge_3c"])),
        "coverage_80": _covered(dist, actual, 0.80),
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
                "mae_expected": mae(group["actual_tmax_c"], group["expected_tmax_c"]),
                "rmse_expected": rmse(group["actual_tmax_c"], group["expected_tmax_c"]),
                "bias_expected": float(group["bias_expected"].mean()),
                "mean_nll": float(group["nll"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "brier_peak_already_passed": float(group["brier_peak_already_passed"].mean()),
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
                "mean_false_upside_probability": float(group["probability_above_actual_integer_bin"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _recommendation(base: dict, candidate: dict, by_hour: pd.DataFrame) -> dict:
    mae_delta = float(candidate["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(base["mean_crps"])
    false_upside_delta = float(candidate["mean_false_upside_probability"]) - float(
        base["mean_false_upside_probability"]
    )
    max_hour_nll_regression = _max_group_regression(
        by_hour,
        candidate="enhanced_spatial_metar_ml",
        base="enhanced_intraday_ml",
        metric="mean_nll",
        key="issue_hour_utc",
    )
    checks = {
        "mae_improves_at_least_0_02c": mae_delta <= -0.02,
        "nll_not_materially_worse": nll_delta <= 0.03,
        "crps_not_materially_worse": crps_delta <= 0.005,
        "false_upside_not_worse_by_3pp": false_upside_delta <= 0.03,
        "max_hour_nll_regression_within_0_15": max_hour_nll_regression <= 0.15,
    }
    decision = "promote_to_main_model" if all(checks.values()) else "do_not_promote_yet"
    return {
        "decision": decision,
        "checks": checks,
        "candidate_minus_base_mae": mae_delta,
        "candidate_minus_base_nll": nll_delta,
        "candidate_minus_base_crps": crps_delta,
        "candidate_minus_base_false_upside_probability": false_upside_delta,
        "max_hour_nll_regression": max_hour_nll_regression,
    }


def _neighbor_coverage(frame: pd.DataFrame) -> dict:
    out = {
        "any_neighbor_available_rate": float((frame["spatial_available_station_count"] > 0).mean()),
        "all_neighbors_available_rate": float((frame["spatial_available_station_count"] >= len(NEIGHBOR_STATIONS)).mean()),
        "mean_available_station_count": float(frame["spatial_available_station_count"].mean()),
    }
    for station in NEIGHBOR_STATIONS:
        column = f"spatial_{station.lower()}_available"
        out[f"{station}_available_rate"] = float(frame[column].mean()) if column in frame.columns else 0.0
    return out


def _max_group_regression(grouped: pd.DataFrame, *, candidate: str, base: str, metric: str, key: str) -> float:
    candidate_rows = grouped[grouped["model_variant"] == candidate][[key, metric]]
    base_rows = grouped[grouped["model_variant"] == base][[key, metric]]
    merged = candidate_rows.merge(base_rows, on=key, suffixes=("_candidate", "_base"))
    if merged.empty:
        return 0.0
    return float((merged[f"{metric}_candidate"] - merged[f"{metric}_base"]).max())


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_availability: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# EDDM spatial METAR feature comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- target: {report['target']}",
            f"- period: `{report['period'][0]}` to `{report['period'][1]}`",
            f"- rows: `{report['usable_rows']}`",
            f"- days: `{report['days']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            f"- neighbors: `{', '.join(report['neighbor_stations'])}`",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## By UTC Issue Hour",
            "",
            _table(by_hour),
            "",
            "## By Neighbor Availability",
            "",
            _table(by_availability),
            "",
        ]
    )


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
