from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    IconD2MetarTmaxEnsemble,
    MetarTmaxSurvivalCalibrator,
    MetarTmaxUpsideModel,
    mix_distributions,
    prepare_metar_tmax_dataset,
)


NWP_COLUMNS = [
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

COMBO_DELTA_COLUMNS = [
    "icon_minus_ecmwf_model_tmax_c",
    "icon_minus_ecmwf_future_temp_max_c",
    "icon_minus_ecmwf_cloud_cover_mean",
    "icon_minus_ecmwf_precip_sum",
    "icon_minus_ecmwf_shortwave_radiation_sum",
    "icon_minus_ecmwf_wind_speed_max",
    "abs_icon_ecmwf_model_tmax_spread_c",
    "mean_icon_ecmwf_model_tmax_c",
    "max_icon_ecmwf_model_tmax_c",
    "min_icon_ecmwf_model_tmax_c",
]


def main() -> None:
    dataset = pd.read_parquet("data/processed/metar_upside_dataset_LFPB.parquet")
    icon = pd.read_parquet("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    ecmwf = pd.read_parquet("data/forecasts/open_meteo_single_runs_ecmwf_ifs_LFPB.parquet")

    common = _join_common_nwp(dataset, icon, ecmwf)
    if common.empty:
        raise ValueError("No common leakage-safe ICON/ECMWF rows available")

    frame = prepare_metar_tmax_dataset(common)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)
    train_core, calibration, test, split = _time_split(frame)

    variants = _fit_variants(train_core, calibration)
    scored = _score_holdout(test, variants)
    summary = _group_summary(scored, ["model_variant"])
    by_phase = _group_summary(scored, ["model_variant", "phase"])
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _group_summary(scored, ["model_variant", "season"])

    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    common.to_parquet(report_dir / "lfpb_icon_ecmwf_training_common_rows.parquet", index=False)
    scored.to_parquet(report_dir / "lfpb_icon_ecmwf_model_comparison_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_icon_ecmwf_model_comparison_summary.csv", index=False)
    by_phase.to_csv(report_dir / "lfpb_icon_ecmwf_model_comparison_by_phase.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_icon_ecmwf_model_comparison_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_icon_ecmwf_model_comparison_by_season.csv", index=False)

    best = summary.sort_values(["mae_expected", "mean_nll", "mean_crps"]).iloc[0].to_dict()
    icon_row = _summary_row(summary, "icon_ensemble_candidate")
    combo_row = _summary_row(summary, "icon_ecmwf_feature_ml_calibrated")
    posthoc_rows = summary[summary["model_variant"].astype(str).str.startswith("icon_ecmwf_posthoc")].copy()
    best_posthoc_row = (
        {}
        if posthoc_rows.empty
        else posthoc_rows.sort_values(["mean_nll", "mae_expected", "mean_crps"]).iloc[0].to_dict()
    )
    ecmwf_row = _summary_row(summary, "ecmwf_ensemble_candidate")
    recommendation = _recommendation(icon_row, combo_row, best_posthoc_row, ecmwf_row)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "experiment": "ICON-only vs ECMWF-only vs ICON+ECMWF feature model",
        "rows": len(frame),
        "days": int(frame["target_date_local"].nunique()),
        "target_period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "variant_metadata": {name: variant["metadata"] for name, variant in variants.items()},
        "summary": json.loads(summary.to_json(orient="records")),
        "best_by_mae": best,
        "best_posthoc_by_nll": best_posthoc_row,
        "recommendation": recommendation,
        "promotion_status": "diagnostic_only_not_promoted",
        "limitations": [
            "All variants are compared on the common ICON-D2/ECMWF overlap only.",
            "The target is METAR Tmax, not official climate Tmax.",
            "The combined feature model is not deployed; this script is an offline diagnostic.",
            "A positive production decision should require improvement in MAE without worse NLL/CRPS or obvious phase instability.",
        ],
    }
    (report_dir / "lfpb_icon_ecmwf_model_comparison.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    Path("docs/lfpb_icon_ecmwf_model_comparison.md").write_text(
        _markdown(report, summary, by_phase, by_hour, by_season),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _join_common_nwp(dataset: pd.DataFrame, icon: pd.DataFrame, ecmwf: pd.DataFrame) -> pd.DataFrame:
    base = dataset.copy()
    base["target_date_local"] = base["target_date_local"].astype(str)
    base["issue_time_utc"] = pd.to_datetime(base["issue_time_utc"], utc=True)
    base = _join_latest_nwp(base, icon, "icon")
    base = _join_latest_nwp(base, ecmwf, "ecmwf")
    base = base[base["icon_model_tmax_c"].notna() & base["ecmwf_model_tmax_c"].notna()].copy()
    base["phase"] = base["local_issue_hour"].astype(int).map(_phase)
    _add_combo_features(base)
    return base.reset_index(drop=True)


def _join_latest_nwp(dataset: pd.DataFrame, nwp: pd.DataFrame, prefix: str) -> pd.DataFrame:
    nw = nwp.copy()
    nw["target_date_local"] = nw["target_date_local"].astype(str)
    nw["knowledge_time_utc"] = pd.to_datetime(nw["knowledge_time_utc"], utc=True)
    nw["model_run_time_utc"] = pd.to_datetime(nw["model_run_time_utc"], utc=True)
    nw = nw[nw["model_tmax_c"].notna()].sort_values("knowledge_time_utc")
    rows = []
    for _, row in dataset.iterrows():
        candidates = nw[
            (nw["target_date_local"] == row["target_date_local"])
            & (nw["knowledge_time_utc"] <= row["issue_time_utc"])
        ]
        if candidates.empty:
            continue
        latest = candidates.iloc[-1]
        merged = row.to_dict()
        for column in NWP_COLUMNS:
            source_column = column
            if column == "nwp_model_minus_current_max_c":
                merged[f"{prefix}_{column}"] = float(latest["model_tmax_c"]) - float(row["current_metar_max_c"])
                continue
            if column == "nwp_future_minus_current_max_c":
                future = latest.get("model_future_temp_max_c")
                merged[f"{prefix}_{column}"] = np.nan if pd.isna(future) else float(future) - float(row["current_metar_max_c"])
                continue
            if source_column in latest:
                merged[f"{prefix}_{column}"] = latest[source_column]
        merged[f"{prefix}_knowledge_time_utc"] = latest["knowledge_time_utc"].isoformat()
        merged[f"{prefix}_model_run_time_utc"] = latest["model_run_time_utc"].isoformat()
        merged[f"{prefix}_source_id"] = latest.get("source_id")
        max_knowledge = max(
            pd.Timestamp(row["max_feature_knowledge_time_utc"]),
            latest["knowledge_time_utc"],
        )
        merged["max_feature_knowledge_time_utc"] = max_knowledge.isoformat()
        merged["leakage_check_passed"] = bool(max_knowledge <= row["issue_time_utc"])
        rows.append(merged)
    out = pd.DataFrame(rows)
    return out[out["leakage_check_passed"].fillna(False).astype(bool)].reset_index(drop=True) if not out.empty else out


def _add_combo_features(frame: pd.DataFrame) -> None:
    frame["icon_minus_ecmwf_model_tmax_c"] = frame["icon_model_tmax_c"] - frame["ecmwf_model_tmax_c"]
    frame["icon_minus_ecmwf_future_temp_max_c"] = frame["icon_model_future_temp_max_c"] - frame["ecmwf_model_future_temp_max_c"]
    frame["icon_minus_ecmwf_cloud_cover_mean"] = frame["icon_model_cloud_cover_mean"] - frame["ecmwf_model_cloud_cover_mean"]
    frame["icon_minus_ecmwf_precip_sum"] = frame["icon_model_precip_sum"] - frame["ecmwf_model_precip_sum"]
    frame["icon_minus_ecmwf_shortwave_radiation_sum"] = (
        frame["icon_model_shortwave_radiation_sum"] - frame["ecmwf_model_shortwave_radiation_sum"]
    )
    frame["icon_minus_ecmwf_wind_speed_max"] = frame["icon_model_wind_speed_max"] - frame["ecmwf_model_wind_speed_max"]
    frame["abs_icon_ecmwf_model_tmax_spread_c"] = frame["icon_minus_ecmwf_model_tmax_c"].abs()
    frame["mean_icon_ecmwf_model_tmax_c"] = frame[["icon_model_tmax_c", "ecmwf_model_tmax_c"]].mean(axis=1)
    frame["max_icon_ecmwf_model_tmax_c"] = frame[["icon_model_tmax_c", "ecmwf_model_tmax_c"]].max(axis=1)
    frame["min_icon_ecmwf_model_tmax_c"] = frame[["icon_model_tmax_c", "ecmwf_model_tmax_c"]].min(axis=1)


def _fit_variants(train_core: pd.DataFrame, calibration: pd.DataFrame) -> dict[str, dict]:
    variants = {}
    for source in ["icon", "ecmwf"]:
        train_source = _single_source_view(train_core, source)
        calibration_source = _single_source_view(calibration, source)
        feature_columns = list(DEFAULT_METAR_TMAX_FEATURES) + list(NWP_COLUMNS)
        model = _fit_calibrated_model(train_source, calibration_source, feature_columns)
        train_residual = _residual_samples(train_source)
        calibration_residual = _residual_samples(train_source)
        ml_weight = _optimize_ml_weight(calibration_source, model, calibration_residual, f"{source}_ensemble_candidate")
        ensemble = IconD2MetarTmaxEnsemble(
            ml_model=model,
            residuals_by_hour=train_residual,
            ml_weight=ml_weight,
            model_version=f"lfpb_{source}_metar_tmax_experiment",
        )
        variants[f"{source}_ml_calibrated"] = {
            "type": "ml",
            "model": model,
            "source_view": source,
            "metadata": {"feature_count": len(feature_columns), "calibrated": model.calibrator is not None},
        }
        variants[f"{source}_ensemble_candidate"] = {
            "type": "ensemble",
            "model": ensemble,
            "source_view": source,
            "metadata": {"feature_count": len(feature_columns), "ml_weight": ml_weight, "residual_contexts": len(train_residual)},
        }
        variants[f"raw_{source}_residual_distribution"] = {
            "type": "residual",
            "residuals": train_residual,
            "source_view": source,
            "metadata": {"residual_contexts": len(train_residual)},
        }

    combo_feature_columns = (
        list(DEFAULT_METAR_TMAX_FEATURES)
        + [f"icon_{column}" for column in NWP_COLUMNS]
        + [f"ecmwf_{column}" for column in NWP_COLUMNS]
        + list(COMBO_DELTA_COLUMNS)
    )
    combo_model = _fit_calibrated_model(train_core, calibration, combo_feature_columns)
    variants["icon_ecmwf_feature_ml_calibrated"] = {
        "type": "ml",
        "model": combo_model,
        "source_view": "combo",
        "metadata": {"feature_count": len(combo_feature_columns), "calibrated": combo_model.calibrator is not None},
    }
    variants.update(_fit_combo_posthoc_variants(calibration, combo_model, variants["icon_ensemble_candidate"]["model"]))
    return variants


def _fit_combo_posthoc_variants(
    calibration: pd.DataFrame,
    combo_model: MetarTmaxUpsideModel,
    icon_ensemble: IconD2MetarTmaxEnsemble,
) -> dict[str, dict]:
    combo_cache = []
    icon_cache = []
    actuals = []
    for _, row in calibration.iterrows():
        combo_cache.append(combo_model.predict_distribution(row))
        icon_row = _single_source_view(pd.DataFrame([row]), "icon").iloc[0]
        icon_cache.append(icon_ensemble.predict_distribution(icon_row))
        actuals.append(float(row["final_metar_tmax_c"]))

    best_smooth = _best_posthoc_params(combo_cache, icon_cache, actuals, smooth_grid=[0.0, 0.5, 0.75, 1.0, 1.25, 1.5], icon_weight_grid=[0.0])
    best_blend = _best_posthoc_params(combo_cache, icon_cache, actuals, smooth_grid=[0.0], icon_weight_grid=np.linspace(0.0, 0.9, 10))
    best_both = _best_posthoc_params(
        combo_cache,
        icon_cache,
        actuals,
        smooth_grid=[0.0, 0.5, 0.75, 1.0, 1.25, 1.5],
        icon_weight_grid=np.linspace(0.0, 0.9, 10),
    )
    return {
        "icon_ecmwf_posthoc_smooth": {
            "type": "posthoc_combo",
            "combo_model": combo_model,
            "icon_ensemble": icon_ensemble,
            "source_view": "combo",
            "metadata": best_smooth,
        },
        "icon_ecmwf_posthoc_icon_safety_blend": {
            "type": "posthoc_combo",
            "combo_model": combo_model,
            "icon_ensemble": icon_ensemble,
            "source_view": "combo",
            "metadata": best_blend,
        },
        "icon_ecmwf_posthoc_smooth_icon_safety_blend": {
            "type": "posthoc_combo",
            "combo_model": combo_model,
            "icon_ensemble": icon_ensemble,
            "source_view": "combo",
            "metadata": best_both,
        },
    }


def _best_posthoc_params(
    combo_distributions: list[TmaxDistribution],
    icon_distributions: list[TmaxDistribution],
    actuals: list[float],
    *,
    smooth_grid,
    icon_weight_grid,
) -> dict:
    best = None
    for smooth_sigma in smooth_grid:
        for icon_weight in icon_weight_grid:
            scores = []
            maes = []
            crps_values = []
            coverages = []
            for combo_dist, icon_dist, actual in zip(combo_distributions, icon_distributions, actuals):
                dist = _apply_posthoc(combo_dist, icon_dist, smooth_sigma=float(smooth_sigma), icon_weight=float(icon_weight))
                scores.append(nll_integer_bin(dist, actual))
                maes.append(abs(dist.expected_tmax_c - actual))
                crps_values.append(crps_discrete(dist, actual))
                coverages.append(_covered(dist, actual, 0.80))
            candidate = {
                "smooth_sigma": float(smooth_sigma),
                "icon_safety_weight": float(icon_weight),
                "calibration_nll": float(np.mean(scores)),
                "calibration_mae": float(np.mean(maes)),
                "calibration_crps": float(np.mean(crps_values)),
                "calibration_coverage_80": float(np.mean(coverages)),
            }
            if best is None or candidate["calibration_nll"] < best["calibration_nll"]:
                best = candidate
    return best or {"smooth_sigma": 0.0, "icon_safety_weight": 0.0}


def _single_source_view(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out = frame.copy()
    for column in NWP_COLUMNS:
        source_column = f"{source}_{column}"
        if source_column in out.columns:
            out[column] = out[source_column]
    out["nwp_source_name"] = source
    return out


def _fit_calibrated_model(train: pd.DataFrame, calibration: pd.DataFrame, feature_columns: list[str]) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(min_rows=900, max_iter=30, feature_columns=feature_columns).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(_survival_calibration_rows(model, calibration))
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _score_holdout(test: pd.DataFrame, variants: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        for name, variant in variants.items():
            if variant["type"] in {"ml", "ensemble"}:
                feature_row = _single_source_view(pd.DataFrame([row]), variant["source_view"]).iloc[0] if variant["source_view"] in {"icon", "ecmwf"} else row
                dist = variant["model"].predict_distribution(feature_row)
            elif variant["type"] == "posthoc_combo":
                combo_dist = variant["combo_model"].predict_distribution(row)
                icon_row = _single_source_view(pd.DataFrame([row]), "icon").iloc[0]
                icon_dist = variant["icon_ensemble"].predict_distribution(icon_row)
                dist = _apply_posthoc(
                    combo_dist,
                    icon_dist,
                    smooth_sigma=float(variant["metadata"].get("smooth_sigma", 0.0)),
                    icon_weight=float(variant["metadata"].get("icon_safety_weight", 0.0)),
                )
            elif variant["type"] == "residual":
                feature_row = _single_source_view(pd.DataFrame([row]), variant["source_view"]).iloc[0]
                dist = _raw_residual_distribution(feature_row, variant["residuals"])
            else:
                raise ValueError(f"Unknown variant type: {variant['type']}")
            rows.append(_score(name, row, dist))
    return pd.DataFrame(rows)


def _time_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    dates = sorted(frame["target_date_local"].unique())
    train_end_idx = max(1, int(len(dates) * 0.60))
    calibration_end_idx = max(train_end_idx + 1, int(len(dates) * 0.80))
    train_core = frame[frame["target_date_local"].isin(dates[:train_end_idx])].copy()
    calibration = frame[frame["target_date_local"].isin(dates[train_end_idx:calibration_end_idx])].copy()
    test = frame[frame["target_date_local"].isin(dates[calibration_end_idx:])].copy()
    return train_core, calibration, test, {
        "method": "chronological_60_20_20_by_target_day_on_common_icon_ecmwf_overlap",
        "train_start": str(train_core["target_date_local"].min()),
        "train_end": str(train_core["target_date_local"].max()),
        "calibration_start": str(calibration["target_date_local"].min()),
        "calibration_end": str(calibration["target_date_local"].max()),
        "test_start": str(test["target_date_local"].min()),
        "test_end": str(test["target_date_local"].max()),
        "train_rows": len(train_core),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "train_days": int(train_core["target_date_local"].nunique()),
        "calibration_days": int(calibration["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
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


def _optimize_ml_weight(calibration: pd.DataFrame, model: MetarTmaxUpsideModel, residuals: dict[int, np.ndarray], version: str) -> float:
    best_weight = 0.0
    best_score = np.inf
    # Diagnostic speed guard: one-row survival predictions are expensive, and
    # this comparison only needs a coarse ensemble sanity check.
    calibration_sample = calibration.sort_values(["target_date_local", "issue_time_utc"]).iloc[:: max(len(calibration) // 180, 1)]
    for weight in np.linspace(0.0, 1.0, 5):
        ensemble = IconD2MetarTmaxEnsemble(
            ml_model=model,
            residuals_by_hour=residuals,
            ml_weight=float(weight),
            model_version=version,
        )
        scores = [
            nll_integer_bin(ensemble.predict_distribution(row), float(row["final_metar_tmax_c"]))
            for _, row in calibration_sample.iterrows()
        ]
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _residual_samples(train: pd.DataFrame) -> dict[int, np.ndarray]:
    train = train.copy()
    train["residual"] = pd.to_numeric(train["final_metar_tmax_c"], errors="coerce") - pd.to_numeric(
        train["model_tmax_c"],
        errors="coerce",
    )
    out = {}
    for hour, group in train.groupby("local_issue_hour"):
        values = group["residual"].dropna().to_numpy(dtype=float)
        if len(values):
            out[int(hour)] = values
    out[-1] = train["residual"].dropna().to_numpy(dtype=float)
    return out


def _raw_residual_distribution(row: pd.Series, residuals: dict[int, np.ndarray]) -> TmaxDistribution:
    samples = residuals.get(int(row["local_issue_hour"]), residuals.get(-1))
    if samples is None or len(samples) < 20:
        samples = residuals.get(-1, np.array([0.0]))
    rounded = np.rint(float(row["model_tmax_c"]) + samples).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    return TmaxDistribution(bins, probabilities).truncate_below(float(row["current_metar_max_c"]))


def _apply_posthoc(
    combo_dist: TmaxDistribution,
    icon_dist: TmaxDistribution,
    *,
    smooth_sigma: float,
    icon_weight: float,
) -> TmaxDistribution:
    dist = _smooth_distribution(combo_dist, smooth_sigma) if smooth_sigma > 0 else combo_dist
    if icon_weight > 0:
        dist = mix_distributions(dist, icon_dist, icon_weight)
    return dist


def _smooth_distribution(dist: TmaxDistribution, sigma: float) -> TmaxDistribution:
    if sigma <= 0:
        return dist
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * np.square(offsets / sigma))
    kernel = kernel / kernel.sum()
    bins = np.arange(int(dist.bins_c.min() - radius), int(dist.bins_c.max() + radius) + 1)
    probabilities = np.zeros(len(bins), dtype=float)
    index = {int(bin_c): idx for idx, bin_c in enumerate(bins)}
    for bin_c, probability in zip(dist.bins_c, dist.probabilities):
        for offset, kernel_value in zip(offsets, kernel):
            probabilities[index[int(bin_c + offset)]] += float(probability) * float(kernel_value)
    return TmaxDistribution(bins, probabilities)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "phase": _phase(int(row["local_issue_hour"])),
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
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _summary_row(summary: pd.DataFrame, model_variant: str) -> dict:
    row = summary[summary["model_variant"] == model_variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _recommendation(icon_row: dict, combo_row: dict, best_posthoc_row: dict, ecmwf_row: dict) -> dict:
    if not icon_row or not combo_row:
        return {"decision": "insufficient_rows", "reason": "Required variants are missing from the summary."}
    candidate = best_posthoc_row or combo_row
    mae_delta = float(candidate["mae_expected"]) - float(icon_row["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(icon_row["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(icon_row["mean_crps"])
    ecmwf_delta = None if not ecmwf_row else float(ecmwf_row["mae_expected"]) - float(icon_row["mae_expected"])
    if mae_delta < -0.03 and nll_delta <= 0.02 and crps_delta <= 0.02:
        decision = "consider_shadow_promotion"
        reason = "The best calibrated ICON+ECMWF candidate improved MAE materially without a clear probabilistic penalty."
    elif nll_delta > 0.25 or crps_delta > 0.02:
        decision = "do_not_use_ecmwf_in_production_yet"
        reason = "Even after post-hoc calibration, ICON+ECMWF did not preserve probabilistic quality well enough."
    elif mae_delta < -0.01:
        decision = "keep_as_shadow_only"
        reason = "The best calibrated ICON+ECMWF candidate improved MAE slightly, but the gain is too small for direct production promotion."
    else:
        decision = "do_not_use_ecmwf_in_production_yet"
        reason = "The best calibrated ICON+ECMWF candidate did not beat ICON-only on the main holdout comparison."
    return {
        "decision": decision,
        "reason": reason,
        "candidate_compared_to_icon": candidate.get("model_variant"),
        "combo_minus_icon_mae": mae_delta,
        "combo_minus_icon_nll": nll_delta,
        "combo_minus_icon_crps": crps_delta,
        "ecmwf_minus_icon_mae": ecmwf_delta,
    }


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _phase(hour: int) -> str:
    if hour < 9:
        return "before_work"
    if hour < 12:
        return "morning"
    if hour < 15:
        return "midday"
    if hour < 17:
        return "afternoon"
    return "evening"


def _season(value) -> str:
    month = pd.Timestamp(value).month
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _markdown(report: dict, summary: pd.DataFrame, by_phase: pd.DataFrame, by_hour: pd.DataFrame, by_season: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB ICON/ECMWF model comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- period: `{report['target_period'][0]}` to `{report['target_period'][1]}`",
            f"- rows: `{report['rows']}`",
            f"- days: `{report['days']}`",
            f"- split: `{report['split']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            f"- reason: {report['recommendation']['reason']}",
            "",
            "## Overall",
            "",
            _table(summary),
            "",
            "## By Phase",
            "",
            _table(by_phase),
            "",
            "## By Local Issue Hour",
            "",
            _table(by_hour),
            "",
            "## By Season",
            "",
            _table(by_season),
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
