from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.intraday_survival_prior import (
    adjust_upside_probability,
    build_daily_first_metar_max,
    build_seasonal_hourly_survival_table,
    lookup_survival_prior,
    season_for_month,
)
from weather_tmax_bot.models.distribution import TmaxDistribution

BINS = np.arange(-35, 46)
FORMULAS = (
    ("cap_blend", 0.25),
    ("cap_blend", 0.50),
    ("cap_blend", 0.75),
    ("cap_blend", 1.00),
    ("multiply", 0.25),
    ("multiply", 0.50),
    ("multiply", 0.75),
    ("multiply", 1.00),
)
GATES = {
    "all": lambda prior, local_hour: True,
    "survival_le_005": lambda prior, local_hour: prior <= 0.05,
    "survival_le_010": lambda prior, local_hour: prior <= 0.10,
    "survival_le_020": lambda prior, local_hour: prior <= 0.20,
    "survival_le_035": lambda prior, local_hour: prior <= 0.35,
    "local_ge17": lambda prior, local_hour: local_hour >= 17,
}
FOLD_STARTS = pd.date_range("2025-08-01", "2025-12-01", freq="MS").date


def main() -> None:
    seasonal = _load_seasonal_weight_analysis()
    dataset = pd.read_parquet("data/processed/training_dataset.parquet")
    truth = pd.read_parquet("data/processed/daily_target.parquet")
    metar = pd.read_parquet("data/interim/metar_iem_EDDM.parquet")
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"], errors="coerce").dt.date
    daily_metar = build_daily_first_metar_max(metar)

    rows = []
    fold_inventory = []
    for fold_start in FOLD_STARTS:
        fold_end = min(pd.Timestamp("2025-12-30").date(), (pd.Timestamp(fold_start) + pd.offsets.MonthBegin(1)).date() - pd.Timedelta(days=1))
        records = seasonal._build_fold_records(dataset, truth, fold_start, fold_end)
        survival = build_seasonal_hourly_survival_table(daily_metar, train_before=fold_start)
        fold_rows = _score_fold(records, survival)
        rows.extend(fold_rows)
        fold_inventory.append(
            {
                "fold_start": fold_start.isoformat(),
                "fold_end": str(fold_end),
                "test_rows": len(records),
                "survival_train_days": int((daily_metar["target_date_local"] < fold_start).sum()),
            }
        )

    scored = pd.DataFrame(rows)
    summary = _summaries(scored, ["model_variant"])
    by_hour = _summaries(scored, ["model_variant", "local_hour_floor"])
    by_season = _summaries(scored, ["model_variant", "season"])
    by_regime = _summaries(_regime_rows(scored), ["model_variant", "regime"])
    full_survival = build_seasonal_hourly_survival_table(daily_metar, train_before=pd.Timestamp("2026-01-01").date())
    recommendation = _recommend(summary, by_regime)
    report = {
        "design": (
            "Shadow-only expanding monthly backtest. For each August-December 2025 fold, the ICON-D2 residual prior, "
            "intraday analogues, and seasonal METAR Tmax timing survival table use only dates before the fold."
        ),
        "folds": fold_inventory,
        "evaluated_base_rows": int((scored["model_variant"] == "current_dynamic").sum()),
        "formula_definitions": {
            "cap_blend": "Move strength * max(0, current_upside_probability - seasonal_survival_prior) into the observed Tmax bin.",
            "multiply": "Multiply current upside probability by seasonal_survival_prior ** strength and move removed mass into the observed Tmax bin.",
        },
        "selection_rule": "Prefer lower CRPS, then lower NLL; inspect late-day rows separately before any production promotion.",
        "recommendation": recommendation,
        "illustrative_summer_17_local": _illustrative_case(full_survival),
        "limitations": [
            "Historical evaluation is limited to August-December 2025 because honest forecast-as-issued ICON-D2 overlap starts in late July 2025.",
            "Historical feature rows run at 00/03/06/09/12/15/18 UTC; Railway METAR events arrive around :20/:50 and require continued forward shadow monitoring.",
            "METAR temperatures are integer-rounded. The survival prior measures first attainment of the rounded daily METAR maximum, not the exact DWD 10-minute Tmax time.",
            "This report does not change production forecasts.",
        ],
    }

    Path("data/reports").mkdir(parents=True, exist_ok=True)
    write_parquet(scored, "data/reports/intraday_survival_prior_rows.parquet")
    write_parquet(summary, "data/reports/intraday_survival_prior_summary.parquet")
    write_parquet(by_hour, "data/reports/intraday_survival_prior_by_hour.parquet")
    write_parquet(by_season, "data/reports/intraday_survival_prior_by_season.parquet")
    write_parquet(by_regime, "data/reports/intraday_survival_prior_by_regime.parquet")
    write_parquet(full_survival, "data/reports/metar_seasonal_hourly_survival_prior.parquet")
    Path("data/reports/intraday_survival_prior_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("docs/intraday_survival_prior_analysis.md").write_text(
        _build_doc(report, summary, by_hour, by_regime, full_survival),
        encoding="utf-8",
    )
    print(json.dumps({"recommendation": recommendation, "illustrative_summer_17_local": report["illustrative_summer_17_local"]}, indent=2))


def _load_seasonal_weight_analysis():
    path = Path("scripts/40_analyze_intraday_seasonal_weights.py")
    spec = importlib.util.spec_from_file_location("intraday_seasonal_weight_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _score_fold(records: list[dict], survival: pd.DataFrame) -> list[dict]:
    rows = []
    for record in records:
        current = _distribution_from_record(record)
        prior = lookup_survival_prior(survival, month=pd.Timestamp(record["target_date_local"]).month, local_hour=record["local_hour"])
        rows.append(_score_record(record, current, "current_dynamic", prior, current.threshold_ge(int(np.ceil(record["observed_max_so_far_c"])) + 1)))
        for gate_name, gate in GATES.items():
            for formula, strength in FORMULAS:
                active = bool(gate(prior, record["local_hour"]))
                adjusted = adjust_upside_probability(
                    current,
                    observed_max_so_far_c=record["observed_max_so_far_c"],
                    survival_prior=prior,
                    formula=formula,
                    strength=strength if active else 0.0,
                )
                rows.append(
                    _score_record(
                        record,
                        adjusted.distribution,
                        f"{gate_name}__{formula}_{int(strength * 100):03d}",
                        prior,
                        adjusted.adjusted_upside_probability,
                        adjustment_applied=active,
                    )
                )
    return rows


def _distribution_from_record(record: dict) -> TmaxDistribution:
    probabilities = (1.0 - record["current_weight"]) * record["base_probs"] + record["current_weight"] * record["pure_probs"]
    return TmaxDistribution(BINS, probabilities)


def _score_record(
    record: dict,
    dist: TmaxDistribution,
    variant: str,
    survival_prior: float,
    predicted_upside: float,
    adjustment_applied: bool = False,
) -> dict:
    actual = float(record["actual"])
    observed = float(record["observed_max_so_far_c"])
    actual_bin = int(np.rint(actual))
    actual_prob = float(dist.probabilities[dist.bins_c == actual_bin].sum())
    cdf = np.cumsum(dist.probabilities)
    obs_cdf = dist.bins_c >= actual
    interval_80 = dist.interval(0.80)
    local_hour_floor = int(np.floor(record["local_hour"]))
    return {
        "model_variant": variant,
        "fold_start": record["fold_start"],
        "target_date_local": record["target_date_local"],
        "issue_hour_utc": int(record["issue_hour_utc"]),
        "local_hour": float(record["local_hour"]),
        "local_hour_floor": local_hour_floor,
        "season": season_for_month(pd.Timestamp(record["target_date_local"]).month),
        "actual_tmax_c": actual,
        "observed_max_so_far_c": observed,
        "actual_upside": bool(actual >= np.ceil(observed) + 1),
        "predicted_upside_probability": float(predicted_upside),
        "survival_prior": float(survival_prior),
        "adjustment_applied": bool(adjustment_applied),
        "expected_tmax_c": dist.expected_tmax_c,
        "absolute_error": abs(dist.expected_tmax_c - actual),
        "squared_error": (dist.expected_tmax_c - actual) ** 2,
        "nll": float(-np.log(max(actual_prob, 1e-12))),
        "crps": float(np.mean((cdf - obs_cdf) ** 2)),
        "brier_upside": float((predicted_upside - bool(actual >= np.ceil(observed) + 1)) ** 2),
        "covered_80": bool(interval_80[0] - 0.5 <= actual < interval_80[1] + 0.5),
        "width_80_c": float(interval_80[1] - interval_80[0]),
    }


def _summaries(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        row.update(
            {
                "rows": int(len(group)),
                "days": int(group["target_date_local"].nunique()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "nll": float(group["nll"].mean()),
                "crps": float(group["crps"].mean()),
                "brier_upside": float(group["brier_upside"].mean()),
                "coverage80": float(group["covered_80"].mean()),
                "width80_c": float(group["width_80_c"].mean()),
                "mean_predicted_upside_probability": float(group["predicted_upside_probability"].mean()),
                "actual_upside_rate": float(group["actual_upside"].mean()),
                "mean_survival_prior": float(group["survival_prior"].mean()),
                "adjustment_applied_ratio": float(group["adjustment_applied"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _regime_rows(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scored.iterrows():
        regimes = ["all"]
        if row["local_hour"] >= 17:
            regimes.append("late_local_ge17")
        if row["local_hour"] >= 17 and row["season"] == "summer_JJA":
            regimes.append("summer_late_local_ge17")
        if row["local_hour"] >= 16 and row["season"] in {"summer_JJA", "autumn_SON"}:
            regimes.append("warm_half_late_local_ge16")
        for regime in regimes:
            payload = row.to_dict()
            payload["regime"] = regime
            rows.append(payload)
    return pd.DataFrame(rows)


def _recommend(summary: pd.DataFrame, by_regime: pd.DataFrame) -> dict:
    baseline = summary[summary["model_variant"] == "current_dynamic"].iloc[0]
    promotion_candidates = summary[
        (summary["model_variant"] != "current_dynamic")
        & (summary["adjustment_applied_ratio"] > 0)
        & (summary["crps"] <= baseline["crps"])
    ].sort_values(["crps", "nll"])
    eligible_for_production_promotion = not promotion_candidates.empty
    candidates = promotion_candidates
    if not eligible_for_production_promotion:
        candidates = summary[
            (summary["model_variant"] != "current_dynamic") & (summary["adjustment_applied_ratio"] > 0)
        ].sort_values(["crps", "nll"])
    best = candidates.iloc[0]
    late = by_regime[by_regime["regime"] == "late_local_ge17"].set_index("model_variant")
    best_late = late.loc[best["model_variant"]]
    baseline_late = late.loc["current_dynamic"]
    return {
        "best_shadow_candidate": str(best["model_variant"]),
        "candidate_adjustment_applied_ratio": float(best["adjustment_applied_ratio"]),
        "eligible_for_production_promotion": eligible_for_production_promotion,
        "overall_delta_vs_current": _metric_deltas(best, baseline),
        "late_local_ge17_delta_vs_current": _metric_deltas(best_late, baseline_late),
        "decision": (
            "Keep production unchanged. No tested candidate improves overall CRPS; "
            "run the least harmful late-day correction in forward shadow mode before considering promotion."
            if not eligible_for_production_promotion
            else "Keep production unchanged. Run the selected formula in forward shadow mode before promotion."
        ),
    }


def _metric_deltas(candidate: pd.Series, baseline: pd.Series) -> dict:
    return {
        metric: float(candidate[metric] - baseline[metric])
        for metric in ("mae", "rmse", "nll", "crps", "brier_upside", "coverage80", "mean_predicted_upside_probability")
    }


def _illustrative_case(full_survival: pd.DataFrame) -> dict:
    prior = lookup_survival_prior(full_survival, month=7, local_hour=17.83)
    current_upside = 0.25
    values = {"current_upside_probability": current_upside, "summer_survival_prior_after_17_local": prior}
    base = TmaxDistribution(np.array([23, 24]), np.array([1 - current_upside, current_upside]))
    for formula, strength in FORMULAS:
        adjusted = adjust_upside_probability(
            base,
            observed_max_so_far_c=23.0,
            survival_prior=prior,
            formula=formula,
            strength=strength,
        )
        values[f"{formula}_{int(strength * 100):03d}"] = adjusted.adjusted_upside_probability
    return values


def _build_doc(report: dict, summary: pd.DataFrame, by_hour: pd.DataFrame, by_regime: pd.DataFrame, survival: pd.DataFrame) -> str:
    selected = report["recommendation"]["best_shadow_candidate"]
    late = by_regime[by_regime["regime"].isin(["late_local_ge17", "summer_late_local_ge17", "warm_half_late_local_ge16"])]
    return "\n".join(
        [
            "# Intraday seasonal survival prior analysis",
            "",
            "This is a shadow-only historical experiment. Production forecasting logic is unchanged.",
            "",
            "## Design",
            "",
            report["design"],
            "",
            "The seasonal survival prior is `P(first rounded METAR Tmax attainment is still ahead | season, local hour)`.",
            "",
            "## Seasonal hourly survival prior",
            "",
            _table(survival, ["season", "local_hour", "training_days", "peak_ahead_days", "survival_prior"]),
            "",
            "## Overall candidate comparison",
            "",
            _table(summary, ["model_variant", "rows", "mae", "rmse", "nll", "crps", "brier_upside", "coverage80", "mean_predicted_upside_probability", "actual_upside_rate"]),
            "",
            "## Late-day comparison",
            "",
            _table(late, ["regime", "model_variant", "rows", "mae", "nll", "crps", "brier_upside", "mean_predicted_upside_probability", "actual_upside_rate"]),
            "",
            "## By local hour",
            "",
            _table(by_hour[by_hour["model_variant"].isin(["current_dynamic", selected])], ["model_variant", "local_hour_floor", "rows", "mae", "nll", "crps", "brier_upside", "mean_predicted_upside_probability", "actual_upside_rate"]),
            "",
            "## Recommendation",
            "",
            f"Least harmful historical shadow candidate by CRPS then NLL: `{selected}`.",
            "",
            f"Eligible for production promotion from this experiment alone: `{report['recommendation']['eligible_for_production_promotion']}`.",
            "",
            report["recommendation"]["decision"],
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
        ]
    )


def _table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format(value) -> str:
    return f"{float(value):.4f}" if isinstance(value, (float, np.floating)) else str(value)


if __name__ == "__main__":
    main()
