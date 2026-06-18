from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.features.metar_upside_dataset import (
    ENHANCED_METAR_INTRADAY_FEATURES,
    _prepare_metar,
    empty_enhanced_metar_features,
)
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.intraday_ml import (
    CORE_INTRADAY_ML_FEATURES,
    IntradayMLSurvivalCalibrator,
    IntradayMLUpsideModel,
    infer_intraday_ml_context,
    prepare_intraday_ml_dataset,
)
from weather_tmax_bot.utils.time import local_day_bounds_utc


AIRPORT = "EDDM"
TIMEZONE_NAME = "Europe/Berlin"


def main() -> None:
    dataset = _load_or_build_enhanced_dataset()
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    usable = dataset[dataset["target_date_local"] <= pd.to_datetime("2025-12-30").date()].copy()
    scored, folds = _rolling_backtest(usable)
    summary = _group_summary(scored, ["model_variant"])
    by_hour = _group_summary(scored, ["model_variant", "issue_hour_utc"])
    by_season = _group_summary(scored, ["model_variant", "season"])
    by_regime = _group_summary(scored, ["model_variant", "rain_or_cb_after_max"])
    base = _row(summary, "base_intraday_ml")
    enhanced = _row(summary, "enhanced_intraday_ml")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": AIRPORT,
        "experiment": "current Munich intraday ML features vs enhanced same-day METAR features",
        "target": "official DWD daily Tmax; METAR is used only as as-of intraday signal",
        "dataset_rows": len(dataset),
        "usable_rows": len(usable),
        "days": int(usable["target_date_local"].nunique()),
        "period": [str(usable["target_date_local"].min()), str(usable["target_date_local"].max())],
        "folds": folds,
        "base_feature_count": len(CORE_INTRADAY_ML_FEATURES),
        "enhanced_feature_count": len(CORE_INTRADAY_ML_FEATURES) + len(ENHANCED_METAR_INTRADAY_FEATURES),
        "enhanced_features": ENHANCED_METAR_INTRADAY_FEATURES,
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, enhanced),
    }
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored, report_dir / "eddm_intraday_enhanced_feature_comparison_rows.parquet")
    summary.to_csv(report_dir / "eddm_intraday_enhanced_feature_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "eddm_intraday_enhanced_feature_comparison_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "eddm_intraday_enhanced_feature_comparison_by_season.csv", index=False)
    by_regime.to_csv(report_dir / "eddm_intraday_enhanced_feature_comparison_by_regime.csv", index=False)
    (report_dir / "eddm_intraday_enhanced_feature_comparison.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/eddm_intraday_enhanced_feature_comparison.md").write_text(
        _markdown(report, summary, by_hour, by_regime),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _load_or_build_enhanced_dataset() -> pd.DataFrame:
    output = Path("data/processed/intraday_ml_dataset_enhanced.parquet")
    source = Path("data/processed/training_dataset.parquet")
    metar_path = Path("data/interim/metar_iem_EDDM.parquet")
    if not source.exists():
        raise FileNotFoundError(source)
    if not metar_path.exists():
        raise FileNotFoundError(metar_path)
    dataset = pd.read_parquet(source)
    metar = pd.read_parquet(metar_path)
    enriched = _add_enhanced_metar_features(dataset, metar)
    prepared = prepare_intraday_ml_dataset(enriched)
    write_parquet(prepared, output)
    return prepared


def _add_enhanced_metar_features(dataset: pd.DataFrame, metar: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy().reset_index(drop=True)
    metar_df = _prepare_metar(metar)
    metar_df["target_date_local"] = metar_df["observation_time_utc"].dt.tz_convert(TIMEZONE_NAME).dt.date.astype(str)
    metar_df["_cloud_cover_proxy_value"] = metar_df.apply(_cloud_cover_proxy_fast, axis=1)
    metar_df["_dewpoint_depression_value"] = (
        pd.to_numeric(metar_df["temperature_c"], errors="coerce")
        - pd.to_numeric(metar_df.get("dewpoint_c"), errors="coerce")
    )
    metar_by_day = {
        day: group.sort_values("observation_time_utc").reset_index(drop=True)
        for day, group in metar_df.groupby("target_date_local", sort=False)
    }
    out["target_date_local"] = pd.to_datetime(out["target_date_local"], errors="coerce").dt.date
    out["issue_time_utc"] = pd.to_datetime(out["issue_time_utc"], utc=True, errors="coerce")
    feature_rows = [None] * len(out)
    for target_date, group in out.groupby("target_date_local", sort=False):
        day_key = target_date.isoformat()
        day_metar = metar_by_day.get(day_key, pd.DataFrame())
        day_start, day_end = local_day_bounds_utc(target_date, TIMEZONE_NAME)
        if not day_metar.empty:
            day_metar = day_metar[
                (day_metar["observation_time_utc"] >= pd.Timestamp(day_start))
                & (day_metar["observation_time_utc"] < pd.Timestamp(day_end))
            ].reset_index(drop=True)
        for index, row in group.iterrows():
            feature_rows[index] = _enhanced_for_prepared_day(
                day_metar,
                issue_utc=pd.Timestamp(row["issue_time_utc"]),
                day_start_utc=day_start,
            )
    enhanced = pd.DataFrame(feature_rows, index=out.index)
    for column in ENHANCED_METAR_INTRADAY_FEATURES:
        out[column] = enhanced[column]
    return out


def _enhanced_for_prepared_day(day_metar: pd.DataFrame, *, issue_utc: pd.Timestamp, day_start_utc: datetime) -> dict:
    if day_metar.empty:
        return empty_enhanced_metar_features()
    so_far = day_metar[day_metar["knowledge_time_utc"] <= issue_utc].copy()
    if so_far.empty:
        return empty_enhanced_metar_features()
    latest = so_far.iloc[-1]
    current_max_idx = int(so_far["temperature_c"].idxmax())
    after_current_max = so_far.loc[so_far.index >= current_max_idx].copy()
    last_2 = so_far.tail(2)
    last_2h = _window(so_far, issue_utc, 2)
    since_sunrise = so_far[so_far["observation_time_utc"] >= pd.Timestamp(day_start_utc) + pd.Timedelta(hours=5)]
    cloud = pd.to_numeric(so_far["_cloud_cover_proxy_value"], errors="coerce")
    ceiling = pd.to_numeric(so_far.get("ceiling_ft"), errors="coerce")
    dewpoint_depression = pd.to_numeric(so_far["_dewpoint_depression_value"], errors="coerce")
    wind_direction = pd.to_numeric(so_far.get("wind_direction_deg"), errors="coerce")
    return {
        "temp_slope_since_sunrise": _trend(since_sunrise, "temperature_c"),
        "temp_trend_last_2_metars": _trend(last_2, "temperature_c"),
        "latest_2_metar_temp_change_c": _trend(last_2, "temperature_c"),
        "cloud_cover_proxy_latest": _finite_or_nan(latest.get("_cloud_cover_proxy_value")),
        "cloud_cover_proxy_trend_last_2_metars": _series_trend(cloud.tail(2)),
        "cloud_cover_proxy_trend_2h": _series_trend(cloud.loc[last_2h.index]),
        "lowest_ceiling_ft_latest": _finite_or_nan(latest.get("ceiling_ft")),
        "ceiling_trend_last_2_metars": _series_trend(ceiling.tail(2)),
        "ceiling_trend_2h": _series_trend(ceiling.loc[last_2h.index]),
        "dewpoint_depression_latest": _finite_or_nan(latest.get("_dewpoint_depression_value")),
        "dewpoint_depression_trend_2h": _series_trend(dewpoint_depression.loc[last_2h.index]),
        "pressure_tendency_1h": _trend(_window(so_far, issue_utc, 1), "qnh_hpa"),
        "pressure_tendency_3h": _trend(_window(so_far, issue_utc, 3), "qnh_hpa"),
        "wind_dir_shift_2h_deg": _wind_shift(last_2h),
        "wind_speed_trend_2h": _trend(last_2h, "wind_speed_kt"),
        "wind_direction_latest_deg": _finite_or_nan(latest.get("wind_direction_deg")),
        "wind_speed_latest_kt": _finite_or_nan(latest.get("wind_speed_kt")),
        "rain_started_after_current_max": _has_weather(after_current_max, ["RA", "SHRA", "TSRA"]),
        "cb_tcu_appeared_after_current_max": _has_weather(after_current_max, [" CB", "CB", "TCU"]),
        "showers_appeared_after_current_max": _has_weather(after_current_max, ["SHRA", "SH", "VCSH"]),
        "fog_or_br_recent_metar": _has_weather(_window(so_far, issue_utc, 3), ["FG", "BR"]),
        "cavok_trend_last_2_metars": _series_trend(last_2.get("cavok", pd.Series(dtype=float)).astype(float)),
        "metar_minutes_since_current_max": _minutes_since_current_max(so_far, issue_utc),
        "metar_hours_since_sunrise": max(
            0.0,
            (issue_utc - (pd.Timestamp(day_start_utc) + pd.Timedelta(hours=5))).total_seconds() / 3600.0,
        ),
        "temp_drop_after_rain_start_c": _temp_drop_after_weather(so_far, issue_utc, ["RA", "SHRA", "TSRA"]),
        "temp_drop_after_cb_tcu_c": _temp_drop_after_weather(so_far, issue_utc, [" CB", "CB", "TCU"]),
        "wind_direction_valid_count_2h": int(wind_direction.loc[last_2h.index].dropna().shape[0]),
    }


def _window(df: pd.DataFrame, issue_utc: pd.Timestamp, hours: float) -> pd.DataFrame:
    return df[df["observation_time_utc"] >= issue_utc - pd.Timedelta(hours=hours)]


def _trend(df: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.iloc[-1] - values.iloc[0])


def _series_trend(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.iloc[-1] - clean.iloc[0])


def _wind_shift(df: pd.DataFrame) -> float:
    values = pd.to_numeric(df.get("wind_direction_deg"), errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return float("nan")
    diff = abs(values[-1] - values[0]) % 360
    return float(min(diff, 360 - diff))


def _minutes_since_current_max(so_far: pd.DataFrame, issue_utc: pd.Timestamp) -> float:
    idx = so_far["temperature_c"].idxmax()
    max_time = pd.Timestamp(so_far.loc[idx, "observation_time_utc"])
    return float((issue_utc - max_time).total_seconds() / 60.0)


def _temp_drop_after_weather(so_far: pd.DataFrame, issue_utc: pd.Timestamp, codes: list[str]) -> float:
    mask = so_far.get("raw_metar", pd.Series("", index=so_far.index)).fillna("").astype(str).apply(
        lambda text: any(code in text for code in codes)
    )
    if not mask.any():
        return 0.0
    first_weather = so_far.loc[mask, "observation_time_utc"].min()
    after = so_far[(so_far["observation_time_utc"] >= first_weather) & (so_far["observation_time_utc"] <= issue_utc)]
    if after.empty:
        return 0.0
    return float(pd.to_numeric(after["temperature_c"], errors="coerce").max() - pd.to_numeric(after["temperature_c"], errors="coerce").iloc[-1])


def _has_weather(df: pd.DataFrame, codes: list[str]) -> bool:
    text = " ".join(df.get("raw_metar", pd.Series(dtype=str)).fillna("").astype(str).tolist())
    return any(code in text for code in codes)


def _cloud_cover_proxy_fast(row: pd.Series) -> float:
    if bool(row.get("cavok", False)):
        return 0.0
    text = " ".join(str(row.get(column, "") or "") for column in ["cloud_layers", "raw_metar"])
    if "OVC" in text:
        return 8.0
    if "BKN" in text:
        return 6.0
    if "SCT" in text:
        return 4.0
    if "FEW" in text:
        return 2.0
    if "NSC" in text or "SKC" in text or "CLR" in text:
        return 0.0
    return float("nan")


def _finite_or_nan(value) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def _rolling_backtest(dataset: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    base_features = list(CORE_INTRADAY_ML_FEATURES)
    enhanced_features = base_features + list(ENHANCED_METAR_INTRADAY_FEATURES)
    return _rolling_backtest_with_feature_sets(
        dataset,
        {
            "base_intraday_ml": base_features,
            "enhanced_intraday_ml": enhanced_features,
        },
    )


def _rolling_backtest_with_feature_sets(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[dict]]:
    rows = []
    folds = []
    for fold_start in pd.date_range("2025-08-01", "2025-12-01", freq="MS").date:
        fold_end = (pd.Timestamp(fold_start) + pd.offsets.MonthEnd(1)).date()
        calibration_start = (pd.Timestamp(fold_start) - pd.Timedelta(days=90)).date()
        train_core = dataset[dataset["target_date_local"] < calibration_start].copy()
        calibration = dataset[
            (dataset["target_date_local"] >= calibration_start) & (dataset["target_date_local"] < fold_start)
        ].copy()
        test = dataset[(dataset["target_date_local"] >= fold_start) & (dataset["target_date_local"] <= fold_end)].copy()
        if len(train_core) < 300 or len(calibration) < 100 or test.empty:
            folds.append(
                {
                    "fold_start": fold_start.isoformat(),
                    "fold_end": fold_end.isoformat(),
                    "status": "skipped",
                    "train_core_rows": len(train_core),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                }
            )
            continue
        models = {
            name: _fit_calibrated(train_core, calibration, features)
            for name, features in feature_sets.items()
        }
        for _, row in test.iterrows():
            for name, model in models.items():
                rows.append(_score(name, row, model.predict_distribution(row.to_dict()), fold_start))
        folds.append(
            {
                "fold_start": fold_start.isoformat(),
                "fold_end": fold_end.isoformat(),
                "status": "evaluated",
                "train_core_rows": len(train_core),
                "calibration_rows": len(calibration),
                "test_rows": len(test),
            }
        )
    if not rows:
        raise ValueError("no evaluable EDDM enhanced intraday folds")
    return pd.DataFrame(rows), folds


def _fit_calibrated(train: pd.DataFrame, calibration: pd.DataFrame, features: list[str]) -> IntradayMLUpsideModel:
    model = IntradayMLUpsideModel(feature_columns=features, max_iter=40).fit(train)
    calibrator = IntradayMLSurvivalCalibrator(max_upside_c=model.max_upside_c).fit(
        _survival_calibration_rows(model, calibration)
    )
    model.calibrator = calibrator if calibrator.fitted else None
    return model


def _survival_calibration_rows(model: IntradayMLUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        survival = model.predict_upside_survival(row.to_dict())
        remaining_upside = float(row["remaining_upside_c"])
        out = {
            "target_date_local": str(row["target_date_local"]),
            "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
            "issue_hour_utc": int(row["issue_hour_utc"]),
            "remaining_upside_c": remaining_upside,
        }
        out.update(infer_intraday_ml_context(row))
        for threshold in range(1, model.max_upside_c + 1):
            out[f"raw_probability_upside_ge_{threshold}c"] = float(survival[threshold])
            out[f"actual_upside_ge_{threshold}c"] = float(remaining_upside >= threshold)
        rows.append(out)
    return pd.DataFrame(rows)


def _score(model_variant: str, row: pd.Series, prediction: tuple[TmaxDistribution, dict], fold_start) -> dict:
    dist, details = prediction
    actual = float(row["tmax_c"])
    return {
        "model_variant": model_variant,
        "fold_start": fold_start.isoformat(),
        "target_date_local": str(row["target_date_local"]),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "season": _season(row["target_date_local"]),
        "weather_regime": str(row.get("weather_regime", "unknown")),
        "advection_regime": _advection_regime(row),
        "rain_or_cb_after_max": bool(
            row.get("rain_started_after_current_max", False) or row.get("cb_tcu_appeared_after_current_max", False)
        ),
        "actual_tmax_c": actual,
        "observed_max_so_far_from_metar": float(row["observed_max_so_far_from_metar"]),
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


def _recommendation(base: dict, enhanced: dict) -> dict:
    mae_delta = float(enhanced["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(enhanced["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(enhanced["mean_crps"]) - float(base["mean_crps"])
    false_upside_delta = float(enhanced["mean_false_upside_probability"]) - float(
        base["mean_false_upside_probability"]
    )
    checks = {
        "mae_improves_at_least_0_02c": mae_delta <= -0.02,
        "nll_not_materially_worse": nll_delta <= 0.03,
        "crps_not_materially_worse": crps_delta <= 0.005,
        "false_upside_not_worse_by_3pp": false_upside_delta <= 0.03,
    }
    decision = "promote_to_main_model" if all(checks.values()) else "do_not_promote_yet"
    return {
        "decision": decision,
        "checks": checks,
        "enhanced_minus_base_mae": mae_delta,
        "enhanced_minus_base_nll": nll_delta,
        "enhanced_minus_base_crps": crps_delta,
        "enhanced_minus_base_false_upside_probability": false_upside_delta,
    }


def _row(summary: pd.DataFrame, variant: str) -> dict:
    row = summary[summary["model_variant"] == variant]
    return {} if row.empty else row.iloc[0].to_dict()


def _advection_regime(row: pd.Series) -> str:
    if bool(row.get("adv_any_frontal_passage_signal", False)):
        return "frontal_passage"
    if bool(row.get("adv_any_cold_advection_signal", False)):
        return "cold_advection"
    if bool(row.get("adv_any_warm_advection_signal", False)):
        return "warm_advection"
    if bool(row.get("adv_any_north_sector", False)):
        return "north_sector"
    if bool(row.get("adv_any_south_sector", False)):
        return "south_sector"
    return "neutral_or_missing"


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


def _markdown(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# EDDM enhanced intraday METAR feature comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- target: {report['target']}",
            f"- period: `{report['period'][0]}` to `{report['period'][1]}`",
            f"- rows: `{report['usable_rows']}`",
            f"- days: `{report['days']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## By UTC Issue Hour",
            "",
            _table(by_hour),
            "",
            "## Rain/CB After Current Max",
            "",
            _table(by_regime),
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
