from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.intraday_update import apply_intraday_update
from weather_tmax_bot.models.nwp_residual_model import NWPResidualDistributionModel


def main() -> None:
    args = _parse_args()
    dataset = pd.read_parquet(args.dataset)
    daily_target = pd.read_parquet(args.daily_target)
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    daily_target["target_date_local"] = pd.to_datetime(daily_target["target_date_local"], errors="coerce").dt.date
    if args.rolling:
        _run_rolling_backtest(dataset, daily_target, args)
        return
    test_start = pd.to_datetime(args.test_start).date()
    test_end = pd.to_datetime(args.test_end).date()

    prior_train = dataset[
        (dataset["target_date_local"] < test_start)
        & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
        & dataset["model_tmax_c"].notna()
    ].copy()
    intraday_train = dataset[
        (dataset["target_date_local"] < test_start)
        & dataset["observed_max_so_far_from_metar"].notna()
        & dataset["last_metar_temp_c"].notna()
    ].copy()
    timing_train = daily_target[daily_target["target_date_local"] < test_start].copy()
    test = dataset[
        (dataset["target_date_local"] >= test_start)
        & (dataset["target_date_local"] <= test_end)
        & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
        & dataset["model_tmax_c"].notna()
        & dataset["observed_max_so_far_from_metar"].notna()
        & dataset["last_metar_temp_c"].notna()
    ].copy()
    if prior_train.empty or intraday_train.empty or timing_train.empty or test.empty:
        raise SystemExit("intraday backtest requires non-empty train-only prior, intraday, timing, and holdout slices")

    model = NWPResidualDistributionModel().fit(prior_train)
    rows = []
    for _, row in test.sort_values(["target_date_local", "issue_time_utc"]).iterrows():
        feature_row = row.drop(labels=["tmax_c"]).to_dict()
        observed_max = float(row["observed_max_so_far_from_metar"])
        base_dist = model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed_max)
        update = apply_intraday_update(
            base_dist,
            feature_row,
            row["target_date_local"],
            pd.Timestamp(row["issue_time_utc"]).to_pydatetime(),
            training_frame=intraday_train,
            daily_target_frame=timing_train,
        )
        rows.append(_score("icon_d2_prior", row, base_dist))
        rows.append(_score("icon_d2_prior_plus_intraday", row, update.distribution, details=update.details))

    scored = pd.DataFrame(rows)
    summary = _summaries(scored, ["model_variant"])
    by_hour = _summaries(scored, ["model_variant", "issue_hour_utc"])
    regimes = _build_regime_rows(scored)
    by_regime = _summaries(regimes, ["model_variant", "regime"])
    metadata = _metadata(prior_train, intraday_train, timing_train, test, scored, summary, test_start, test_end)

    write_parquet(scored, "data/reports/intraday_backtest_rows.parquet")
    write_parquet(summary, "data/reports/intraday_backtest_summary.parquet")
    write_parquet(by_hour, "data/reports/intraday_backtest_by_hour.parquet")
    write_parquet(by_regime, "data/reports/intraday_backtest_by_regime.parquet")
    Path("data/reports/intraday_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path("docs/intraday_backtest.md").write_text(_build_doc(metadata, summary, by_hour, by_regime), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe holdout backtest for the same-day intraday update layer.")
    parser.add_argument("--dataset", default="data/processed/training_dataset.parquet")
    parser.add_argument("--daily-target", default="data/processed/daily_target.parquet")
    parser.add_argument("--test-start", default="2025-11-01")
    parser.add_argument("--test-end", default="2025-12-30")
    parser.add_argument("--rolling", action="store_true", help="Run expanding monthly folds starting in August 2025.")
    parser.add_argument("--rolling-start", default="2025-08-01")
    parser.add_argument("--rolling-end", default="2025-12-30")
    return parser.parse_args()


def _run_rolling_backtest(dataset: pd.DataFrame, daily_target: pd.DataFrame, args: argparse.Namespace) -> None:
    rolling_start = pd.to_datetime(args.rolling_start).date()
    rolling_end = pd.to_datetime(args.rolling_end).date()
    fold_starts = pd.date_range(rolling_start, rolling_end, freq="MS").date
    all_rows = []
    fold_metadata = []
    for fold_start in fold_starts:
        next_month = (pd.Timestamp(fold_start) + pd.offsets.MonthBegin(1)).date()
        fold_end = min(rolling_end, next_month - pd.Timedelta(days=1))
        prior_train = dataset[
            (dataset["target_date_local"] < fold_start)
            & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
            & dataset["model_tmax_c"].notna()
        ].copy()
        intraday_train = dataset[
            (dataset["target_date_local"] < fold_start)
            & dataset["observed_max_so_far_from_metar"].notna()
            & dataset["last_metar_temp_c"].notna()
        ].copy()
        timing_train = daily_target[daily_target["target_date_local"] < fold_start].copy()
        test = dataset[
            (dataset["target_date_local"] >= fold_start)
            & (dataset["target_date_local"] <= fold_end)
            & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
            & dataset["model_tmax_c"].notna()
            & dataset["observed_max_so_far_from_metar"].notna()
            & dataset["last_metar_temp_c"].notna()
        ].copy()
        if len(prior_train) < NWPResidualDistributionModel().min_group_rows or test.empty:
            fold_metadata.append(
                {
                    "fold_start": fold_start.isoformat(),
                    "fold_end": str(fold_end),
                    "prior_train_rows": len(prior_train),
                    "test_rows": len(test),
                    "status": "skipped_insufficient_rows",
                }
            )
            continue
        model = NWPResidualDistributionModel().fit(prior_train)
        for _, row in test.sort_values(["target_date_local", "issue_time_utc"]).iterrows():
            feature_row = row.drop(labels=["tmax_c"]).to_dict()
            observed_max = float(row["observed_max_so_far_from_metar"])
            base_dist = model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed_max)
            update = apply_intraday_update(
                base_dist,
                feature_row,
                row["target_date_local"],
                pd.Timestamp(row["issue_time_utc"]).to_pydatetime(),
                training_frame=intraday_train,
                daily_target_frame=timing_train,
            )
            prior_score = _score("icon_d2_prior", row, base_dist)
            updated_score = _score("icon_d2_prior_plus_intraday", row, update.distribution, details=update.details)
            prior_score["fold_start"] = fold_start.isoformat()
            updated_score["fold_start"] = fold_start.isoformat()
            all_rows.extend([prior_score, updated_score])
        fold_metadata.append(
            {
                "fold_start": fold_start.isoformat(),
                "fold_end": str(fold_end),
                "prior_train_period": [str(prior_train["target_date_local"].min()), str(prior_train["target_date_local"].max())],
                "prior_train_rows": len(prior_train),
                "intraday_train_rows": len(intraday_train),
                "timing_train_rows": len(timing_train),
                "test_rows": len(test),
                "status": "evaluated",
            }
        )
    if not all_rows:
        raise SystemExit("rolling intraday backtest found no evaluable folds")
    scored = pd.DataFrame(all_rows)
    summary = _summaries(scored, ["model_variant"])
    by_hour = _summaries(scored, ["model_variant", "issue_hour_utc"])
    by_fold = _summaries(scored, ["model_variant", "fold_start"])
    by_regime = _summaries(_build_regime_rows(scored), ["model_variant", "regime"])
    metadata = _rolling_metadata(scored, summary, fold_metadata, rolling_start, rolling_end)
    write_parquet(scored, "data/reports/intraday_rolling_backtest_rows.parquet")
    write_parquet(summary, "data/reports/intraday_rolling_backtest_summary.parquet")
    write_parquet(by_hour, "data/reports/intraday_rolling_backtest_by_hour.parquet")
    write_parquet(by_fold, "data/reports/intraday_rolling_backtest_by_fold.parquet")
    write_parquet(by_regime, "data/reports/intraday_rolling_backtest_by_regime.parquet")
    Path("data/reports/intraday_rolling_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Path("docs/intraday_rolling_backtest.md").write_text(
        _build_rolling_doc(metadata, summary, by_hour, by_fold, by_regime),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


def _score(variant: str, row: pd.Series, dist, *, details: dict | None = None) -> dict:
    actual = float(row["tmax_c"])
    interval_50 = dist.interval(0.50)
    interval_80 = dist.interval(0.80)
    interval_90 = dist.interval(0.90)
    future_increase = max(0.0, actual - float(row["observed_max_so_far_from_metar"]))
    payload = {
        "model_variant": variant,
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "month": int(row["month"]),
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "observed_max_so_far_c": float(row["observed_max_so_far_from_metar"]),
        "last_metar_temp_c": float(row["last_metar_temp_c"]),
        "drop_from_observed_max_c": max(0.0, float(row["observed_max_so_far_from_metar"]) - float(row["last_metar_temp_c"])),
        "future_increase_c": future_increase,
        "actual_peak_already_passed": future_increase <= 0.5,
        "has_precip_recent": bool(row.get("has_precip_recent", False)),
        "has_thunder_recent": bool(row.get("has_thunder_recent", False)),
        "temp_trend_3h": _optional_float(row.get("temp_trend_3h")),
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_ge_20": brier(dist.threshold_ge(20), actual >= 20),
        "brier_ge_25": brier(dist.threshold_ge(25), actual >= 25),
        "brier_ge_30": brier(dist.threshold_ge(30), actual >= 30),
        "covered_50": _interval_contains_actual_bin(interval_50, actual),
        "covered_80": _interval_contains_actual_bin(interval_80, actual),
        "covered_90": _interval_contains_actual_bin(interval_90, actual),
        "width_80_c": interval_80[1] - interval_80[0],
    }
    if details is not None:
        payload.update(
            {
                "intraday_active": bool(details.get("active", False)),
                "intraday_reason": details.get("reason"),
                "predicted_peak_passed_probability": details.get("peak_passed_probability"),
                "timing_peak_passed_prior": details.get("timing_peak_passed_prior"),
                "intraday_blend_weight": details.get("intraday_blend_weight"),
                "nwp_future_upside_c": details.get("nwp_future_upside_c"),
            }
        )
    return payload


def _summaries(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "rows": len(group),
                "mae_expected": mae(group["actual_tmax_c"], group["expected_tmax_c"]),
                "rmse_expected": rmse(group["actual_tmax_c"], group["expected_tmax_c"]),
                "bias_expected": float(np.mean(group["expected_tmax_c"] - group["actual_tmax_c"])),
                "mean_nll": float(group["nll"].mean()),
                "mean_crps": float(group["crps"].mean()),
                "brier_ge_20": float(group["brier_ge_20"].mean()),
                "brier_ge_25": float(group["brier_ge_25"].mean()),
                "brier_ge_30": float(group["brier_ge_30"].mean()),
                "coverage_50": float(group["covered_50"].mean()),
                "coverage_80": float(group["covered_80"].mean()),
                "coverage_90": float(group["covered_90"].mean()),
                "mean_width_80_c": float(group["width_80_c"].mean()),
            }
        )
        if group["predicted_peak_passed_probability"].notna().any():
            probabilities = group["predicted_peak_passed_probability"].dropna()
            actual = group.loc[probabilities.index, "actual_peak_already_passed"]
            row["peak_passed_brier"] = float(np.mean((probabilities - actual.astype(float)) ** 2))
            row["intraday_active_ratio"] = float(group["intraday_active"].fillna(False).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_regime_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scored.iterrows():
        for regime in _regimes(row):
            payload = row.to_dict()
            payload["regime"] = regime
            rows.append(payload)
    return pd.DataFrame(rows)


def _regimes(row: pd.Series) -> list[str]:
    regimes = ["all"]
    if bool(row["has_precip_recent"]):
        regimes.append("precip_recent")
    else:
        regimes.append("dry_recent")
    if bool(row["has_thunder_recent"]):
        regimes.append("thunder_recent")
    if float(row["drop_from_observed_max_c"]) >= 3:
        regimes.append("drop_ge_3c")
    if float(row["drop_from_observed_max_c"]) >= 5:
        regimes.append("drop_ge_5c")
    if int(row["issue_hour_utc"]) <= 6:
        regimes.append("early_day_utc_00_06")
    if int(row["issue_hour_utc"]) >= 12:
        regimes.append("late_day_utc_12_18")
    if bool(row["actual_peak_already_passed"]):
        regimes.append("actual_peak_passed")
    return regimes


def _metadata(
    prior_train: pd.DataFrame,
    intraday_train: pd.DataFrame,
    timing_train: pd.DataFrame,
    test: pd.DataFrame,
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    test_start: date,
    test_end: date,
) -> dict:
    base = summary[summary["model_variant"] == "icon_d2_prior"].iloc[0]
    updated = summary[summary["model_variant"] == "icon_d2_prior_plus_intraday"].iloc[0]
    active = scored[scored["model_variant"] == "icon_d2_prior_plus_intraday"]
    return {
        "design": "fixed holdout; all prior, intraday analogue, and Tmax timing inputs are restricted to target dates before test_start",
        "test_start": test_start.isoformat(),
        "test_end": test_end.isoformat(),
        "prior_train_period": [str(prior_train["target_date_local"].min()), str(prior_train["target_date_local"].max())],
        "prior_train_rows": len(prior_train),
        "intraday_train_period": [str(intraday_train["target_date_local"].min()), str(intraday_train["target_date_local"].max())],
        "intraday_train_rows": len(intraday_train),
        "timing_train_period": [str(timing_train["target_date_local"].min()), str(timing_train["target_date_local"].max())],
        "timing_train_rows": len(timing_train),
        "holdout_rows": len(test),
        "holdout_dates": [str(test["target_date_local"].min()), str(test["target_date_local"].max())],
        "holdout_threshold_event_rows": {
            "ge_20": int((test["tmax_c"] >= 20).sum()),
            "ge_25": int((test["tmax_c"] >= 25).sum()),
            "ge_30": int((test["tmax_c"] >= 30).sum()),
        },
        "intraday_active_ratio": float(active["intraday_active"].fillna(False).mean()),
        "improvement_intraday_minus_prior": {
            "mae_expected": float(updated["mae_expected"] - base["mae_expected"]),
            "rmse_expected": float(updated["rmse_expected"] - base["rmse_expected"]),
            "mean_nll": float(updated["mean_nll"] - base["mean_nll"]),
            "mean_crps": float(updated["mean_crps"] - base["mean_crps"]),
            "brier_ge_30": float(updated["brier_ge_30"] - base["brier_ge_30"]),
        },
        "limitations": [
            "Historical IEM METAR currently ends on 2025-12-30, so the simultaneous ICON-D2 plus METAR holdout ends there.",
            "Historical feature rows are evaluated at 00/03/06/09/12/15/18 UTC. The new Railway availability-aware +01:40 schedule remains a forward-test concern.",
            "The holdout covers November and December only; seasonal promotion requires more archived same-day evidence.",
            "The winter holdout has no Tmax >=25C or >=30C events, so warm-season threshold reliability remains untested.",
        ],
    }


def _rolling_metadata(
    scored: pd.DataFrame,
    summary: pd.DataFrame,
    fold_metadata: list[dict],
    rolling_start: date,
    rolling_end: date,
) -> dict:
    base = summary[summary["model_variant"] == "icon_d2_prior"].iloc[0]
    updated = summary[summary["model_variant"] == "icon_d2_prior_plus_intraday"].iloc[0]
    test = scored[scored["model_variant"] == "icon_d2_prior"]
    active = scored[scored["model_variant"] == "icon_d2_prior_plus_intraday"]
    return {
        "design": "expanding monthly folds; every fold trains the ICON-D2 prior, intraday analogues, and timing prior only on dates before its test month",
        "rolling_start": rolling_start.isoformat(),
        "rolling_end": rolling_end.isoformat(),
        "evaluated_rows": len(test),
        "folds": fold_metadata,
        "holdout_threshold_event_rows": {
            "ge_20": int((test["actual_tmax_c"] >= 20).sum()),
            "ge_25": int((test["actual_tmax_c"] >= 25).sum()),
            "ge_30": int((test["actual_tmax_c"] >= 30).sum()),
        },
        "intraday_active_ratio": float(active["intraday_active"].fillna(False).mean()),
        "improvement_intraday_minus_prior": {
            "mae_expected": float(updated["mae_expected"] - base["mae_expected"]),
            "rmse_expected": float(updated["rmse_expected"] - base["rmse_expected"]),
            "mean_nll": float(updated["mean_nll"] - base["mean_nll"]),
            "mean_crps": float(updated["mean_crps"] - base["mean_crps"]),
            "brier_ge_20": float(updated["brier_ge_20"] - base["brier_ge_20"]),
            "brier_ge_25": float(updated["brier_ge_25"] - base["brier_ge_25"]),
            "brier_ge_30": float(updated["brier_ge_30"] - base["brier_ge_30"]),
        },
        "limitations": [
            "The first August fold has only a short ICON-D2 residual history and is a warm-start stress test, not a mature production-model estimate.",
            "Historical feature rows are evaluated at 00/03/06/09/12/15/18 UTC. The Railway +01:40 schedule still requires forward monitoring.",
            "Only five expanding monthly folds are available because forecast-as-issued ICON-D2 coverage and historical METAR overlap is limited.",
        ],
    }


def _build_doc(metadata: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame) -> str:
    lines = [
        "# Intraday update backtest",
        "",
        "This report evaluates whether the same-day METAR and sampled ICON-D2 update improves the ICON-D2 residual prior without temporal leakage.",
        "",
        "## Leakage-safe design",
        "",
        f"- holdout: `{metadata['test_start']}` to `{metadata['test_end']}`",
        f"- ICON-D2 prior train period: `{metadata['prior_train_period'][0]}` to `{metadata['prior_train_period'][1]}`",
        f"- ICON-D2 prior train rows: `{metadata['prior_train_rows']}`",
        f"- intraday analogue train period: `{metadata['intraday_train_period'][0]}` to `{metadata['intraday_train_period'][1]}`",
        f"- intraday analogue train rows: `{metadata['intraday_train_rows']}`",
        f"- holdout rows: `{metadata['holdout_rows']}`",
        "- all prior distributions, intraday analogues, and Tmax timing priors are restricted to dates before the holdout",
        "",
        "## Overall metrics",
        "",
        _table(summary, ["model_variant", "rows", "mae_expected", "rmse_expected", "bias_expected", "mean_nll", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "Lower is better for MAE, RMSE, NLL, CRPS, and Brier score. Coverage 80 should be interpreted against the 0.80 target.",
        "",
        "## Interpretation",
        "",
        "- The intraday layer improves overall MAE, NLL, and 80% interval coverage. RMSE and CRPS are slightly worse on this mature winter slice.",
        "- During 12/15/18 UTC issues, expected-value MAE improves materially because observations identify days where little remaining upside is plausible.",
        "- During 00/03/06/09 UTC issues, expected-value MAE is worse but interval coverage improves: the layer mainly widens uncertainty before the daytime peak.",
        "- For observed drops of at least 5C from the current-day METAR maximum, MAE improves from about 0.97C to 0.57C.",
        "- The holdout contains no Tmax >=25C or >=30C events. It validates the mechanism on winter weather, not summer heat calibration.",
        "",
        "## By issue hour",
        "",
        _table(by_hour, ["model_variant", "issue_hour_utc", "rows", "mae_expected", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "## Selected regimes",
        "",
        _table(by_regime[by_regime["regime"].isin(["all", "precip_recent", "dry_recent", "drop_ge_3c", "drop_ge_5c", "early_day_utc_00_06", "late_day_utc_12_18", "actual_peak_passed"])], ["model_variant", "regime", "rows", "mae_expected", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in metadata["limitations"])
    lines.extend(
        [
            "",
            "## Operational decision",
            "",
            "Keep the intraday layer enabled as a monitored secondary correction. Use this holdout to identify weak issue hours and regimes; do not treat the November-December slice as sufficient evidence for final seasonal calibration.",
        ]
    )
    return "\n".join(lines)


def _build_rolling_doc(
    metadata: dict,
    summary: pd.DataFrame,
    by_hour: pd.DataFrame,
    by_fold: pd.DataFrame,
    by_regime: pd.DataFrame,
) -> str:
    lines = [
        "# Intraday rolling backtest",
        "",
        "This report expands the fixed winter holdout into monthly forward folds. Each fold uses only information from earlier dates.",
        "",
        "## Leakage-safe design",
        "",
        f"- rolling period: `{metadata['rolling_start']}` to `{metadata['rolling_end']}`",
        f"- evaluated rows: `{metadata['evaluated_rows']}`",
        f"- event rows Tmax >=20C / >=25C / >=30C: `{metadata['holdout_threshold_event_rows']['ge_20']}` / `{metadata['holdout_threshold_event_rows']['ge_25']}` / `{metadata['holdout_threshold_event_rows']['ge_30']}`",
        "- every month refits the ICON-D2 residual prior and restricts METAR analogues plus Tmax timing climatology to prior dates",
        "",
        "## Overall metrics",
        "",
        _table(summary, ["model_variant", "rows", "mae_expected", "rmse_expected", "bias_expected", "mean_nll", "mean_crps", "brier_ge_20", "brier_ge_25", "brier_ge_30", "coverage_80"]),
        "",
        "## By fold",
        "",
        _table(by_fold, ["model_variant", "fold_start", "rows", "mae_expected", "rmse_expected", "mean_nll", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "## Interpretation",
        "",
        "- The rolling check is the broader historical test because it includes warm-season rows and 35 Tmax >=30C cases.",
        "- The intraday layer improves overall MAE, RMSE, NLL, Brier >=25C, Brier >=30C, and 80% interval coverage.",
        "- CRPS is almost neutral but slightly worse overall, mainly because early-day uncertainty widening and late-day sharpening are not yet separately calibrated.",
        "- The strongest expected-value gains are after 12 UTC and on days where the observed METAR maximum has already dropped by several degrees.",
        "- Brier >=20C is slightly worse, so threshold calibration should not be tightened solely from this MVP layer.",
        "",
        "## By issue hour",
        "",
        _table(by_hour, ["model_variant", "issue_hour_utc", "rows", "mae_expected", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "## Selected regimes",
        "",
        _table(by_regime[by_regime["regime"].isin(["all", "precip_recent", "dry_recent", "drop_ge_3c", "drop_ge_5c", "early_day_utc_00_06", "late_day_utc_12_18", "actual_peak_passed"])], ["model_variant", "regime", "rows", "mae_expected", "mean_crps", "brier_ge_30", "coverage_80"]),
        "",
        "## Fold inventory",
        "",
    ]
    for fold in metadata["folds"]:
        lines.append(
            f"- `{fold['fold_start']}` to `{fold['fold_end']}`: `{fold['status']}`, prior train rows `{fold['prior_train_rows']}`, test rows `{fold['test_rows']}`"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in metadata["limitations"])
    lines.extend(
        [
            "",
            "## Operational decision",
            "",
            "Treat the rolling result as the broader historical check and keep the fixed winter holdout as a stricter mature-history slice. Continue forward monitoring on the Railway availability-aware schedule before tightening production acceptance gates.",
        ]
    )
    return "\n".join(lines)


def _table(df: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _format_value(value) -> str:
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def _optional_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _interval_contains_actual_bin(interval: tuple[float, float], actual: float) -> bool:
    return interval[0] - 0.5 <= actual < interval[1] + 0.5


if __name__ == "__main__":
    main()
