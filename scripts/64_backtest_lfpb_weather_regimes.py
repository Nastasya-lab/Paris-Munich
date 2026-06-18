from __future__ import annotations

import argparse
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
from weather_tmax_bot.models.weather_regime import REGIME_LABELS, add_weather_regime_columns


ENHANCED_INTRADAY_FEATURES = [
    "temp_slope_since_sunrise",
    "temp_trend_last_2_metars",
    "latest_2_metar_temp_change_c",
    "cloud_cover_proxy_latest",
    "cloud_cover_proxy_trend_last_2_metars",
    "cloud_cover_proxy_trend_2h",
    "lowest_ceiling_ft_latest",
    "ceiling_trend_last_2_metars",
    "ceiling_trend_2h",
    "dewpoint_depression_latest",
    "dewpoint_depression_trend_2h",
    "pressure_tendency_1h",
    "pressure_tendency_3h",
    "wind_dir_shift_2h_deg",
    "wind_speed_trend_2h",
    "wind_direction_latest_deg",
    "wind_speed_latest_kt",
    "rain_started_after_current_max",
    "cb_tcu_appeared_after_current_max",
    "showers_appeared_after_current_max",
    "fog_or_br_recent_metar",
    "cavok_trend_last_2_metars",
    "metar_minutes_since_current_max",
    "metar_hours_since_sunrise",
    "temp_drop_after_rain_start_c",
    "temp_drop_after_cb_tcu_c",
    "wind_direction_valid_count_2h",
]

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

REGIME_FEATURE_COLUMNS = [
    *(f"weather_regime_{label}_score" for label in REGIME_LABELS),
    *(f"weather_regime_is_{label}" for label in REGIME_LABELS),
]


def main() -> None:
    args = _parse_args()
    raw = pd.read_parquet(args.dataset)
    frame = prepare_metar_tmax_dataset(raw)
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.date
    frame["season"] = frame["target_date_local"].map(_season)
    frame = frame[frame["model_tmax_c"].notna()].copy()
    frame = add_weather_regime_columns(frame)
    frame = frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)

    train, calibration, test, split = _time_split(frame)
    feature_columns = list(DEFAULT_METAR_TMAX_FEATURES) + list(ENHANCED_INTRADAY_FEATURES) + list(NWP_COLUMNS)

    icon_model = _fit_model(
        train=train,
        calibration=calibration,
        feature_columns=feature_columns,
        min_train_rows=args.min_train_rows,
        max_iter=args.max_iter,
    )
    regime_feature_model = _fit_model(
        train=train,
        calibration=calibration,
        feature_columns=feature_columns + REGIME_FEATURE_COLUMNS,
        min_train_rows=args.min_train_rows,
        max_iter=args.max_iter,
    )

    base_cal_residuals = _residual_samples_by_hour(train)
    base_weight = _optimize_ml_weight(
        calibration,
        lambda row: _base_residual_distribution(row, base_cal_residuals),
        icon_model,
    )
    base_cal_ensemble = IconD2MetarTmaxEnsemble(
        ml_model=icon_model,
        residuals_by_hour=base_cal_residuals,
        ml_weight=base_weight,
        model_version="lfpb_icon_d2_production_like_calibration_replay",
    )
    base_test_ensemble = IconD2MetarTmaxEnsemble(
        ml_model=icon_model,
        residuals_by_hour=_residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True)),
        ml_weight=base_weight,
        model_version="lfpb_icon_d2_production_like_replay",
    )
    regime_feature_weight = _optimize_ml_weight(
        calibration,
        lambda row: _base_residual_distribution(row, base_cal_residuals),
        regime_feature_model,
    )
    regime_feature_test_ensemble = IconD2MetarTmaxEnsemble(
        ml_model=regime_feature_model,
        residuals_by_hour=_residual_samples_by_hour(pd.concat([train, calibration], ignore_index=True)),
        ml_weight=regime_feature_weight,
        model_version="lfpb_icon_d2_regime_feature_replay",
    )

    regime_cal_residuals = _residual_samples_by_regime(train)
    regime_weight = _optimize_ml_weight(
        calibration,
        lambda row: _regime_residual_distribution(row, regime_cal_residuals),
        icon_model,
    )
    regime_test_residuals = _residual_samples_by_regime(pd.concat([train, calibration], ignore_index=True))

    prior_weights = _optimize_prior_weights_by_regime(
        calibration=calibration,
        base_distribution_fn=base_cal_ensemble.predict_distribution,
        prior_samples=_prior_samples(train),
        max_upside_c=icon_model.max_upside_c,
        min_rows=args.min_regime_rows,
    )
    test_prior_samples = _prior_samples(pd.concat([train, calibration], ignore_index=True))

    scored = _score_holdout(
        test=test,
        production_like_fn=base_test_ensemble.predict_distribution,
        regime_residual_fn=lambda row: mix_distributions(
            _regime_residual_distribution(row, regime_test_residuals),
            icon_model.predict_distribution(row),
            regime_weight,
        ),
        regime_feature_fn=regime_feature_test_ensemble.predict_distribution,
        regime_prior_fn=lambda row: _regime_prior_blend_distribution(
            row=row,
            base=base_test_ensemble.predict_distribution(row),
            prior_samples=test_prior_samples,
            weights=prior_weights,
            max_upside_c=icon_model.max_upside_c,
        ),
    )

    summary = _summary(scored, ["model_variant"])
    by_regime = _summary(scored, ["model_variant", "weather_regime"])
    by_hour = _summary(scored, ["model_variant", "local_issue_hour"])
    regime_counts = _regime_counts(frame, train, calibration, test)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "experiment": "offline rule-based weather regime replay for METAR Tmax",
        "dataset": args.dataset,
        "rows": len(frame),
        "days": int(frame["target_date_local"].nunique()),
        "period": [str(frame["target_date_local"].min()), str(frame["target_date_local"].max())],
        "split": split,
        "regime_labels": REGIME_LABELS,
        "regime_feature_columns": REGIME_FEATURE_COLUMNS,
        "base_ml_weight": base_weight,
        "regime_feature_ml_weight": regime_feature_weight,
        "regime_residual_ml_weight": regime_weight,
        "prior_blend_weights_by_regime": prior_weights,
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(summary, by_regime, regime_counts),
        "limitations": [
            "This is an offline replay only; production artifacts are not changed.",
            "Regimes are rule-based and use only as-of METAR/NWP feature columns.",
            "Some regimes have few holdout rows, so promotion requires stable improvement in aggregate and no large regime regression.",
        ],
    }

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(report_dir / "lfpb_weather_regime_backtest_rows.parquet", index=False)
    summary.to_csv(report_dir / "lfpb_weather_regime_backtest_summary.csv", index=False)
    by_regime.to_csv(report_dir / "lfpb_weather_regime_backtest_by_regime.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_weather_regime_backtest_by_hour.csv", index=False)
    regime_counts.to_csv(report_dir / "lfpb_weather_regime_counts.csv", index=False)
    (report_dir / "lfpb_weather_regime_backtest.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path(args.doc_path).write_text(_markdown(report, summary, by_regime, regime_counts), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


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


def _fit_model(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    feature_columns: list[str],
    min_train_rows: int,
    max_iter: int,
) -> MetarTmaxUpsideModel:
    model = MetarTmaxUpsideModel(
        min_rows=min_train_rows,
        max_iter=max_iter,
        feature_columns=feature_columns,
    ).fit(train)
    calibrator = MetarTmaxSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(
        _survival_calibration_rows(model, calibration)
    )
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _residual_samples_by_hour(frame: pd.DataFrame) -> dict[int, np.ndarray]:
    data = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    data["residual"] = data["final_metar_tmax_c"].astype(float) - data["model_tmax_c"].astype(float)
    out = {-1: data["residual"].to_numpy(dtype=float)}
    for hour, group in data.groupby("local_issue_hour"):
        out[int(hour)] = group["residual"].to_numpy(dtype=float)
    return out


def _residual_samples_by_regime(frame: pd.DataFrame) -> dict[tuple[str, int], np.ndarray]:
    data = frame.dropna(subset=["model_tmax_c", "final_metar_tmax_c"]).copy()
    data["residual"] = data["final_metar_tmax_c"].astype(float) - data["model_tmax_c"].astype(float)
    out = {("__all__", -1): data["residual"].to_numpy(dtype=float)}
    for regime, group in data.groupby("weather_regime"):
        out[(str(regime), -1)] = group["residual"].to_numpy(dtype=float)
    for (regime, hour), group in data.groupby(["weather_regime", "local_issue_hour"]):
        out[(str(regime), int(hour))] = group["residual"].to_numpy(dtype=float)
    for hour, group in data.groupby("local_issue_hour"):
        out[("__all__", int(hour))] = group["residual"].to_numpy(dtype=float)
    return out


def _base_residual_distribution(row: pd.Series, samples_by_hour: dict[int, np.ndarray]) -> TmaxDistribution:
    samples = samples_by_hour.get(int(row["local_issue_hour"]))
    if samples is None or len(samples) < 20:
        samples = samples_by_hour.get(-1, np.array([0.0]))
    return _residual_distribution(row, samples)


def _regime_residual_distribution(row: pd.Series, samples_by_context: dict[tuple[str, int], np.ndarray]) -> TmaxDistribution:
    regime = str(row.get("weather_regime", "__all__"))
    hour = int(row.get("local_issue_hour", -1))
    for key in [(regime, hour), (regime, -1), ("__all__", hour), ("__all__", -1)]:
        samples = samples_by_context.get(key)
        if samples is not None and len(samples) >= 20:
            return _residual_distribution(row, samples)
    return _residual_distribution(row, np.array([0.0]))


def _residual_distribution(row: pd.Series, samples: np.ndarray) -> TmaxDistribution:
    rounded = np.rint(float(row["model_tmax_c"]) + np.asarray(samples, dtype=float)).astype(int)
    bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
    probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
    return TmaxDistribution(bins, probabilities).truncate_below(float(row["current_metar_max_c"]))


def _prior_samples(frame: pd.DataFrame) -> dict[tuple[str, int, str], np.ndarray]:
    data = frame.dropna(subset=["remaining_upside_c"]).copy()
    out = {("__all__", -1, "all"): data["remaining_upside_c"].to_numpy(dtype=float)}
    for regime, group in data.groupby("weather_regime"):
        out[(str(regime), -1, "all")] = group["remaining_upside_c"].to_numpy(dtype=float)
    for hour, group in data.groupby("local_issue_hour"):
        out[("__all__", int(hour), "all")] = group["remaining_upside_c"].to_numpy(dtype=float)
    for (regime, hour), group in data.groupby(["weather_regime", "local_issue_hour"]):
        out[(str(regime), int(hour), "all")] = group["remaining_upside_c"].to_numpy(dtype=float)
    for (regime, hour, season), group in data.groupby(["weather_regime", "local_issue_hour", "season"]):
        out[(str(regime), int(hour), str(season))] = group["remaining_upside_c"].to_numpy(dtype=float)
    return out


def _prior_distribution(
    row: pd.Series,
    samples_by_context: dict[tuple[str, int, str], np.ndarray],
    max_upside_c: int,
) -> TmaxDistribution:
    regime = str(row.get("weather_regime", "__all__"))
    hour = int(row.get("local_issue_hour", -1))
    season = str(row.get("season", "all"))
    for key in [
        (regime, hour, season),
        (regime, hour, "all"),
        (regime, -1, "all"),
        ("__all__", hour, "all"),
        ("__all__", -1, "all"),
    ]:
        samples = samples_by_context.get(key)
        if samples is not None and len(samples) >= 20:
            upside = np.clip(samples, 0.0, max_upside_c)
            rounded = np.rint(float(row["current_metar_max_c"]) + upside).astype(int)
            bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
            probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
            return TmaxDistribution(bins, probabilities)
    return TmaxDistribution(np.array([int(round(row["current_metar_max_c"]))]), np.array([1.0]))


def _optimize_ml_weight(
    calibration: pd.DataFrame,
    residual_distribution_fn,
    model: MetarTmaxUpsideModel,
) -> float:
    cached = [
        (
            residual_distribution_fn(row),
            model.predict_distribution(row),
            float(row["final_metar_tmax_c"]),
        )
        for _, row in calibration.iterrows()
    ]
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        losses = []
        for residual_dist, ml_dist, actual in cached:
            dist = mix_distributions(residual_dist, ml_dist, float(weight))
            losses.append(nll_integer_bin(dist, actual))
        score = float(np.mean(losses))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _optimize_prior_weights_by_regime(
    *,
    calibration: pd.DataFrame,
    base_distribution_fn,
    prior_samples: dict[tuple[str, int, str], np.ndarray],
    max_upside_c: int,
    min_rows: int,
) -> dict[str, float]:
    cached = [
        {
            "weather_regime": str(row.get("weather_regime", "__default__")),
            "base": base_distribution_fn(row),
            "prior": _prior_distribution(row, prior_samples, max_upside_c),
            "actual": float(row["final_metar_tmax_c"]),
        }
        for _, row in calibration.iterrows()
    ]
    weights: dict[str, float] = {}
    global_weight = _best_prior_weight(cached)
    for regime in sorted({item["weather_regime"] for item in cached}):
        group = [item for item in cached if item["weather_regime"] == regime]
        if len(group) < min_rows:
            weights[str(regime)] = global_weight
        else:
            weights[str(regime)] = _best_prior_weight(group)
    weights["__default__"] = global_weight
    return weights


def _best_prior_weight(cached_rows: list[dict]) -> float:
    best_weight = 0.0
    best_score = np.inf
    for weight in np.linspace(0.0, 0.60, 13):
        losses = []
        for item in cached_rows:
            dist = mix_distributions(item["base"], item["prior"], float(weight))
            losses.append(nll_integer_bin(dist, item["actual"]))
        score = float(np.mean(losses))
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _regime_prior_blend_distribution(
    *,
    row: pd.Series,
    base: TmaxDistribution,
    prior_samples: dict[tuple[str, int, str], np.ndarray],
    weights: dict[str, float],
    max_upside_c: int,
) -> TmaxDistribution:
    regime = str(row.get("weather_regime", "__default__"))
    weight = float(weights.get(regime, weights.get("__default__", 0.0)))
    return mix_distributions(base, _prior_distribution(row, prior_samples, max_upside_c), weight)


def _score_holdout(
    *,
    test: pd.DataFrame,
    production_like_fn,
    regime_residual_fn,
    regime_feature_fn,
    regime_prior_fn,
) -> pd.DataFrame:
    rows = []
    for _, row in test.iterrows():
        rows.append(_score("production_like_icon_d2", row, production_like_fn(row)))
        rows.append(_score("regime_residual_icon_d2", row, regime_residual_fn(row)))
        rows.append(_score("regime_feature_icon_d2", row, regime_feature_fn(row)))
        rows.append(_score("regime_prior_blend", row, regime_prior_fn(row)))
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": str(row.get("season", "unknown")),
        "weather_regime": str(row.get("weather_regime", "unknown")),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "mae_expected": abs(dist.expected_tmax_c - actual),
        "bias_expected": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_upside_ge_1c": brier(dist.threshold_ge(int(np.ceil(current_max + 1))), bool(row["remaining_upside_c"] >= 1.0)),
        "brier_upside_ge_2c": brier(dist.threshold_ge(int(np.ceil(current_max + 2))), bool(row["remaining_upside_c"] >= 2.0)),
        "brier_upside_ge_3c": brier(dist.threshold_ge(int(np.ceil(current_max + 3))), bool(row["remaining_upside_c"] >= 3.0)),
        "coverage_80": _covered(dist, actual, 0.80),
    }


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
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _regime_counts(frame: pd.DataFrame, train: pd.DataFrame, calibration: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name, data in [("all", frame), ("train", train), ("calibration", calibration), ("test", test)]:
        counts = data["weather_regime"].value_counts().to_dict()
        for regime in REGIME_LABELS:
            rows.append({"split": split_name, "weather_regime": regime, "rows": int(counts.get(regime, 0))})
    return pd.DataFrame(rows)


def _recommendation(summary: pd.DataFrame, by_regime: pd.DataFrame, regime_counts: pd.DataFrame) -> dict:
    base = _variant_row(summary, "production_like_icon_d2")
    candidates = []
    for variant in ["regime_residual_icon_d2", "regime_feature_icon_d2", "regime_prior_blend"]:
        row = _variant_row(summary, variant)
        if not row:
            continue
        mae_delta = float(row["mae_expected"]) - float(base["mae_expected"])
        nll_delta = float(row["mean_nll"]) - float(base["mean_nll"])
        crps_delta = float(row["mean_crps"]) - float(base["mean_crps"])
        max_regime_nll_regression = _max_regime_regression(by_regime, variant, "mean_nll")
        candidates.append(
            {
                "variant": variant,
                "mae_delta": mae_delta,
                "nll_delta": nll_delta,
                "crps_delta": crps_delta,
                "max_regime_nll_regression": max_regime_nll_regression,
            }
        )
    best = min(candidates, key=lambda item: (item["nll_delta"], item["mae_delta"])) if candidates else {}
    enough_regime_rows = bool((regime_counts[regime_counts["split"] == "test"]["rows"] >= 30).sum() >= 3)
    promote = (
        bool(best)
        and best["nll_delta"] <= -0.03
        and best["mae_delta"] <= 0.03
        and best["max_regime_nll_regression"] <= 0.25
        and enough_regime_rows
    )
    return {
        "decision": "candidate_for_shadow" if promote else "do_not_promote_yet",
        "best_variant": best.get("variant"),
        "candidate_deltas_vs_production_like": candidates,
        "reason": (
            "Regime-aware replay improves probabilistic quality without major point or per-regime regression."
            if promote
            else "Regime-aware replay is useful diagnostically, but the holdout gain is not strong/stable enough for production."
        ),
    }


def _max_regime_regression(by_regime: pd.DataFrame, variant: str, metric: str) -> float:
    base = by_regime[by_regime["model_variant"] == "production_like_icon_d2"][["weather_regime", metric]]
    cand = by_regime[by_regime["model_variant"] == variant][["weather_regime", metric]]
    merged = base.merge(cand, on="weather_regime", suffixes=("_base", "_candidate"))
    if merged.empty:
        return 0.0
    return float((merged[f"{metric}_candidate"] - merged[f"{metric}_base"]).max())


def _variant_row(summary: pd.DataFrame, variant: str) -> dict:
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


def _markdown(report: dict, summary: pd.DataFrame, by_regime: pd.DataFrame, regime_counts: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB weather regime replay",
            "",
            "Offline replay only. Production artifacts were not changed.",
            "",
            "## Summary",
            "",
            "```csv",
            summary.to_csv(index=False).strip(),
            "```",
            "",
            "## Holdout rows by regime",
            "",
            "```csv",
            regime_counts[regime_counts["split"] == "test"].to_csv(index=False).strip(),
            "```",
            "",
            "## By regime",
            "",
            "```csv",
            by_regime.to_csv(index=False).strip(),
            "```",
            "",
            "## Recommendation",
            "",
            "```json",
            json.dumps(report["recommendation"], indent=2, default=str),
            "```",
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest rule-based LFPB weather-regime forecast adjustments.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--report-dir", default="data/reports")
    parser.add_argument("--doc-path", default="docs/lfpb_weather_regime_backtest.md")
    parser.add_argument("--min-train-rows", type=int, default=1200)
    parser.add_argument("--min-regime-rows", type=int, default=60)
    parser.add_argument("--max-iter", type=int, default=60)
    return parser.parse_args()


if __name__ == "__main__":
    main()
