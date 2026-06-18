from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    prepare_metar_tmax_dataset,
)


BASE_NWP_COLUMNS = [
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

ENHANCED_NWP_COLUMNS = [
    "model_cloud_cover_max",
    "model_low_cloud_cover_mean",
    "model_low_cloud_cover_max",
    "model_mid_cloud_cover_mean",
    "model_mid_cloud_cover_max",
    "model_high_cloud_cover_mean",
    "model_high_cloud_cover_max",
    "model_precip_probability_max",
    "model_precip_hours",
    "model_rain_sum",
    "model_rain_hours",
    "model_showers_sum",
    "model_showers_hours",
    "model_has_thunderstorm_code",
    "model_has_rain_code",
    "model_direct_radiation_sum",
    "model_diffuse_radiation_sum",
    "model_sunshine_duration_sum",
    "model_cape_max",
    "model_lifted_index_min",
    "model_future_cloud_cover_max",
    "model_future_low_cloud_cover_mean",
    "model_future_low_cloud_cover_max",
    "model_future_mid_cloud_cover_mean",
    "model_future_mid_cloud_cover_max",
    "model_future_high_cloud_cover_mean",
    "model_future_high_cloud_cover_max",
    "model_future_precip_probability_max",
    "model_future_precip_hours",
    "model_future_rain_sum",
    "model_future_rain_hours",
    "model_future_showers_sum",
    "model_future_showers_hours",
    "model_future_has_thunderstorm_code",
    "model_future_has_rain_code",
    "model_future_direct_radiation_sum",
    "model_future_diffuse_radiation_sum",
    "model_future_sunshine_duration_sum",
    "model_future_cape_max",
    "model_future_lifted_index_min",
]


def main() -> None:
    dataset = pd.read_parquet("data/processed/metar_upside_dataset_LFPB.parquet")
    nwp = pd.read_parquet("data/forecasts/open_meteo_single_runs_icon_d2_LFPB_enhanced.parquet")
    joined = _join_asof_nwp(dataset, nwp)
    joined = joined[joined["model_tmax_c"].notna()].copy()
    if joined.empty:
        raise ValueError("No enhanced ICON-D2 rows joined")

    frame = prepare_metar_tmax_dataset(joined)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train, calibration, test, split = _time_split(frame)
    min_rows = max(180, min(900, len(train)))
    residuals = _residual_samples(train)

    baseline_model = _fit_model(train, calibration, list(DEFAULT_METAR_TMAX_FEATURES) + BASE_NWP_COLUMNS, min_rows)
    enhanced_model = _fit_model(
        train,
        calibration,
        list(DEFAULT_METAR_TMAX_FEATURES) + BASE_NWP_COLUMNS + ENHANCED_NWP_COLUMNS,
        min_rows,
    )
    baseline_weight = _optimize_ml_weight(calibration, baseline_model, residuals)
    enhanced_weight = _optimize_ml_weight(calibration, enhanced_model, residuals)
    baseline_ensemble = IconD2MetarTmaxEnsemble(baseline_model, residuals, baseline_weight, "lfpb_icon_base_partial")
    enhanced_ensemble = IconD2MetarTmaxEnsemble(enhanced_model, residuals, enhanced_weight, "lfpb_icon_enhanced_partial")

    scored = _score_holdout(test, baseline_model, enhanced_model, baseline_ensemble, enhanced_ensemble, residuals)
    summary = _summary(scored, ["model_variant"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(report_dir / "lfpb_icon_enhanced_nwp_joined_partial.parquet", index=False)
    scored.to_parquet(report_dir / "lfpb_icon_enhanced_nwp_comparison_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_icon_enhanced_nwp_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_icon_enhanced_nwp_comparison_by_hour.csv", index=False)
    base = _row(summary, "base_icon_ensemble")
    enhanced = _row(summary, "enhanced_icon_ensemble")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "experiment": "partial enhanced ICON-D2 cloud/precip/radiation feature test",
        "rows_joined": len(frame),
        "days_joined": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "baseline_ml_weight": baseline_weight,
        "enhanced_ml_weight": enhanced_weight,
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, enhanced, frame),
        "limitations": [
            "This is a partial-window stress test because the full enhanced ICON-D2 archive is not downloaded yet.",
            "Do not promote based on this report alone; rerun after the enhanced archive covers the full 2025-07-27..2026-05-30 overlap.",
            "Enhanced future aggregates are relative to model availability time, matching the existing NWP archive convention.",
        ],
    }
    (report_dir / "lfpb_icon_enhanced_nwp_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path("docs/lfpb_icon_enhanced_nwp_comparison.md").write_text(_markdown(report, summary, by_hour), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


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
        for column in BASE_NWP_COLUMNS + ENHANCED_NWP_COLUMNS:
            if column in latest:
                merged[column] = latest[column]
        merged["nwp_model_minus_current_max_c"] = float(latest["model_tmax_c"]) - float(row["current_metar_max_c"])
        future = latest.get("model_future_temp_max_c")
        merged["nwp_future_minus_current_max_c"] = np.nan if pd.isna(future) else float(future) - float(row["current_metar_max_c"])
        merged["nwp_knowledge_time_utc"] = latest["knowledge_time_utc"].isoformat()
        merged["nwp_model_run_time_utc"] = latest["model_run_time_utc"].isoformat()
        merged["max_feature_knowledge_time_utc"] = max(
            pd.Timestamp(row["max_feature_knowledge_time_utc"]),
            latest["knowledge_time_utc"],
        ).isoformat()
        merged["leakage_check_passed"] = bool(pd.Timestamp(merged["max_feature_knowledge_time_utc"]) <= row["issue_time_utc"])
        rows.append(merged)
    out = pd.DataFrame(rows)
    return out[out["leakage_check_passed"].fillna(False).astype(bool)].reset_index(drop=True) if not out.empty else out


def _time_split(frame: pd.DataFrame):
    dates = sorted(frame["target_date_local"].unique())
    if len(dates) < 30:
        raise ValueError(f"Partial enhanced archive has too few days for stress test: {len(dates)}")
    train_end = max(1, int(len(dates) * 0.60))
    calibration_end = max(train_end + 1, int(len(dates) * 0.80))
    train = frame[frame["target_date_local"].isin(dates[:train_end])].copy()
    calibration = frame[frame["target_date_local"].isin(dates[train_end:calibration_end])].copy()
    test = frame[frame["target_date_local"].isin(dates[calibration_end:])].copy()
    return train, calibration, test, {
        "method": "chronological_60_20_20_on_available_enhanced_archive",
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


def _fit_model(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str], min_rows: int) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(min_rows=min_rows, max_iter=30, feature_columns=features).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    return model


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


def _optimize_ml_weight(calibration: pd.DataFrame, model: MetarTmaxUpsideModel, residuals: dict[int, np.ndarray]) -> float:
    best_weight = 0.0
    best_score = np.inf
    sample = calibration.iloc[:: max(len(calibration) // 120, 1)]
    for weight in np.linspace(0.0, 1.0, 5):
        ensemble = IconD2MetarTmaxEnsemble(model, residuals, float(weight))
        score = float(np.mean([nll_integer_bin(ensemble.predict_distribution(row), float(row["final_metar_tmax_c"])) for _, row in sample.iterrows()]))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _score_holdout(
    test: pd.DataFrame,
    baseline_model: MetarTmaxUpsideModel,
    enhanced_model: MetarTmaxUpsideModel,
    baseline_ensemble: IconD2MetarTmaxEnsemble,
    enhanced_ensemble: IconD2MetarTmaxEnsemble,
    residuals: dict[int, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("base_icon_ml", row, baseline_model.predict_distribution(row)))
        rows.append(_score("enhanced_icon_ml", row, enhanced_model.predict_distribution(row)))
        rows.append(_score("base_icon_ensemble", row, baseline_ensemble.predict_distribution(row)))
        rows.append(_score("enhanced_icon_ensemble", row, enhanced_ensemble.predict_distribution(row)))
        rows.append(_score("raw_icon_residual_distribution", row, _raw_residual_distribution(row, residuals)))
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "actual_metar_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "coverage_80": _covered(dist, actual, 0.80),
    }


def _residual_samples(train: pd.DataFrame) -> dict[int, np.ndarray]:
    frame = train.copy()
    frame["residual"] = pd.to_numeric(frame["final_metar_tmax_c"], errors="coerce") - pd.to_numeric(frame["model_tmax_c"], errors="coerce")
    out = {int(hour): group["residual"].dropna().to_numpy(dtype=float) for hour, group in frame.groupby("local_issue_hour")}
    out[-1] = frame["residual"].dropna().to_numpy(dtype=float)
    return out


def _raw_residual_distribution(row: pd.Series, residuals: dict[int, np.ndarray]) -> TmaxDistribution:
    samples = residuals.get(int(row["local_issue_hour"]), residuals.get(-1, np.array([0.0])))
    rounded = np.rint(float(row["model_tmax_c"]) + samples).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    return TmaxDistribution(bins, probabilities).truncate_below(float(row["current_metar_max_c"]))


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
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _recommendation(base: dict, enhanced: dict, frame: pd.DataFrame) -> dict:
    if not base or not enhanced:
        return {"decision": "insufficient_rows", "reason": "Missing baseline or enhanced summary row."}
    mae_delta = float(enhanced["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(enhanced["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(enhanced["mean_crps"]) - float(base["mean_crps"])
    if int(frame["target_date_local"].nunique()) < 180:
        decision = "partial_only_do_not_promote"
        reason = "Enhanced archive covers too few days for a production decision."
    elif mae_delta < -0.03 and nll_delta <= 0.02 and crps_delta <= 0.02:
        decision = "consider_shadow_promotion"
        reason = "Enhanced weather features improved point error without hurting probabilistic metrics."
    else:
        decision = "do_not_promote_yet"
        reason = "Enhanced weather features did not pass the promotion gate."
    return {
        "decision": decision,
        "reason": reason,
        "enhanced_minus_base_mae": mae_delta,
        "enhanced_minus_base_nll": nll_delta,
        "enhanced_minus_base_crps": crps_delta,
    }


def _row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB enhanced ICON-D2 NWP feature comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- period: `{report['target_period'][0]}` to `{report['target_period'][1]}`",
            f"- rows joined: `{report['rows_joined']}`",
            f"- days joined: `{report['days_joined']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            f"- reason: {report['recommendation']['reason']}",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## By Hour",
            "",
            _table(by_hour),
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
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
