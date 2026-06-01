from __future__ import annotations

import json
from itertools import combinations_with_replacement
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import weather_tmax_bot.models.intraday_update as intraday
from weather_tmax_bot.models.nwp_residual_model import NWPResidualDistributionModel


BINS = np.arange(-35, 46)
LOCAL_TZ = ZoneInfo("Europe/Berlin")
GROUPS = ((0, 3), (6, 9), (12,), (15, 18))
GROUP_LABELS = ("utc_00_03", "utc_06_09", "utc_12", "utc_15_18")
CHALLENGER_COOL = (0.0, 0.0, 0.25, 0.55)


def main() -> None:
    dataset = pd.read_parquet("data/processed/training_dataset.parquet")
    truth = pd.read_parquet("data/processed/daily_target.parquet")
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"], errors="coerce").dt.date

    scenarios = {
        "warm": {
            "validation": ("2025-08-01", "2025-08-31"),
            "test": ("2025-09-01", "2025-09-30"),
            "description": "Warm-season cold-start profile: fit on August 2025, test on September 2025.",
        },
        "cool": {
            "validation": ("2025-09-01", "2025-10-31"),
            "test": ("2025-11-01", "2025-12-30"),
            "description": "Cool-season profile: fit on September-October 2025, test on November-December 2025.",
        },
    }
    report = {"scenarios": {}, "global_notes": _global_notes(dataset)}
    for name, spec in scenarios.items():
        validation_records = _build_monthly_records(dataset, truth, *spec["validation"])
        test_records = _build_monthly_records(dataset, truth, *spec["test"])
        validation = _to_arrays(validation_records)
        test = _to_arrays(test_records)
        optimized = _optimize_profiles(validation)
        scenario_report = {
            "description": spec["description"],
            "validation_period": spec["validation"],
            "test_period": spec["test"],
            "validation_inventory": _inventory(validation_records),
            "test_inventory": _inventory(test_records),
            "optimized_on_validation": optimized,
            "test_summary": _scenario_test_summary(test, optimized),
            "test_by_issue_hour": _by_issue_hour(test, optimized),
            "test_regimes": _regime_summary(test, optimized),
            "late_override_sweep": _late_override_sweep(validation, test, optimized["crps"]["profile"]),
            "paired_day_bootstrap": {
                "crps_profile_minus_current": _paired_bootstrap(test, _weights_from_profile(test, optimized["crps"]["profile"]), test["current_weight"]),
                "mae_profile_minus_current": _paired_bootstrap(test, _weights_from_profile(test, optimized["mae"]["profile"]), test["current_weight"]),
                "crps_profile_minus_prior": _paired_bootstrap(test, _weights_from_profile(test, optimized["crps"]["profile"]), np.zeros(len(test["actual"]))),
                "mae_profile_minus_prior": _paired_bootstrap(test, _weights_from_profile(test, optimized["mae"]["profile"]), np.zeros(len(test["actual"]))),
            },
        }
        report["scenarios"][name] = scenario_report

    Path("data/reports").mkdir(parents=True, exist_ok=True)
    Path("data/reports/intraday_seasonal_weight_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("docs/intraday_seasonal_weight_analysis.md").write_text(_build_doc(report), encoding="utf-8")
    print(json.dumps(_console_summary(report), indent=2))


def _build_monthly_records(dataset: pd.DataFrame, truth: pd.DataFrame, start_date: str, end_date: str) -> list[dict]:
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    records: list[dict] = []
    for fold_start in pd.date_range(start, end, freq="MS").date:
        fold_end = min(end, (pd.Timestamp(fold_start) + pd.offsets.MonthBegin(1)).date() - pd.Timedelta(days=1))
        records.extend(_build_fold_records(dataset, truth, fold_start, fold_end))
    return records


def _build_fold_records(dataset: pd.DataFrame, truth: pd.DataFrame, start_date, end_date) -> list[dict]:
    prior_train = dataset[
        (dataset["target_date_local"] < start_date)
        & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
        & dataset["model_tmax_c"].notna()
    ].copy()
    analogue_raw = dataset[
        (dataset["target_date_local"] < start_date)
        & dataset["observed_max_so_far_from_metar"].notna()
        & dataset["last_metar_temp_c"].notna()
    ].copy()
    timing_raw = truth[truth["target_date_local"] < start_date].copy()
    test = dataset[
        (dataset["target_date_local"] >= start_date)
        & (dataset["target_date_local"] <= end_date)
        & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
        & dataset["model_tmax_c"].notna()
        & dataset["observed_max_so_far_from_metar"].notna()
        & dataset["last_metar_temp_c"].notna()
    ].copy()
    if len(prior_train) < NWPResidualDistributionModel().min_group_rows or test.empty:
        return []

    model = NWPResidualDistributionModel().fit(prior_train)
    original_load = intraday._load_intraday_training
    original_timing = intraday._timing_peak_passed_prior
    original_weight = intraday._intraday_blend_weight
    analogue = original_load("unused", training_frame=analogue_raw)
    timing_cache = {}
    intraday._load_intraday_training = lambda *args, **kwargs: analogue
    intraday._intraday_blend_weight = lambda *, local_hour, peak_probability: 1.0

    def cached_timing(*, daily_target_path, target_date, local_hour, daily_target_frame=None):
        key = (target_date.month, round(float(local_hour), 6))
        if key not in timing_cache:
            timing_cache[key] = original_timing(
                daily_target_path=daily_target_path,
                target_date=target_date,
                local_hour=local_hour,
                daily_target_frame=timing_raw,
            )
        return timing_cache[key]

    intraday._timing_peak_passed_prior = cached_timing
    rows = []
    try:
        for _, row in test.sort_values(["target_date_local", "issue_time_utc"]).iterrows():
            feature_row = row.drop(labels=["tmax_c"]).to_dict()
            observed_max = float(row["observed_max_so_far_from_metar"])
            issue_time = pd.Timestamp(row["issue_time_utc"]).to_pydatetime()
            base = model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed_max)
            pure = intraday.apply_intraday_update(
                base,
                feature_row,
                row["target_date_local"],
                issue_time,
                training_frame=analogue_raw,
                daily_target_frame=timing_raw,
            )
            local_issue = issue_time.astimezone(LOCAL_TZ)
            peak_probability = float(pure.details.get("peak_passed_probability") or 0.0)
            current_weight = original_weight(
                local_hour=local_issue.hour + local_issue.minute / 60,
                peak_probability=peak_probability,
            )
            rows.append(
                {
                    "fold_start": str(start_date),
                    "target_date_local": str(row["target_date_local"]),
                    "issue_hour_utc": int(row["issue_hour_utc"]),
                    "local_hour": local_issue.hour + local_issue.minute / 60,
                    "actual": float(row["tmax_c"]),
                    "observed_max_so_far_c": observed_max,
                    "base_probs": _align(base),
                    "pure_probs": _align(pure.distribution),
                    "current_weight": float(current_weight),
                    "peak_probability": peak_probability,
                    "drop_from_observed_max_c": max(0.0, observed_max - float(row["last_metar_temp_c"])),
                    "has_precip_recent": bool(row.get("has_precip_recent", False)),
                    "nwp_future_upside_c": pure.details.get("nwp_future_upside_c"),
                    "prior_train_rows": int(len(prior_train)),
                    "analogue_train_rows": int(len(analogue_raw)),
                }
            )
    finally:
        intraday._load_intraday_training = original_load
        intraday._timing_peak_passed_prior = original_timing
        intraday._intraday_blend_weight = original_weight
    return rows


def _align(distribution) -> np.ndarray:
    out = np.zeros(len(BINS), dtype=float)
    out[distribution.bins_c - BINS[0]] = distribution.probabilities
    return out


def _to_arrays(records: list[dict]) -> dict:
    return {
        "records": records,
        "date": np.array([record["target_date_local"] for record in records]),
        "issue_hour_utc": np.array([record["issue_hour_utc"] for record in records], dtype=int),
        "actual": np.array([record["actual"] for record in records], dtype=float),
        "observed_max_so_far_c": np.array([record["observed_max_so_far_c"] for record in records], dtype=float),
        "base": np.stack([record["base_probs"] for record in records]) if records else np.empty((0, len(BINS))),
        "pure": np.stack([record["pure_probs"] for record in records]) if records else np.empty((0, len(BINS))),
        "current_weight": np.array([record["current_weight"] for record in records], dtype=float),
        "peak_probability": np.array([record["peak_probability"] for record in records], dtype=float),
        "drop_from_observed_max_c": np.array([record["drop_from_observed_max_c"] for record in records], dtype=float),
        "has_precip_recent": np.array([record["has_precip_recent"] for record in records], dtype=bool),
        "nwp_future_upside_c": np.array([
            np.nan if record["nwp_future_upside_c"] is None else float(record["nwp_future_upside_c"])
            for record in records
        ]),
    }


def _optimize_profiles(validation: dict) -> dict:
    grid = np.round(np.arange(0.0, 1.0, 0.05), 2)
    profiles = [tuple(float(value) for value in profile) for profile in combinations_with_replacement(grid, len(GROUPS))]
    results = {}
    for objective in ("crps", "mae", "nll"):
        best = None
        for profile in profiles:
            metrics = _metrics(validation, _weights_from_profile(validation, profile))
            key = (metrics[objective], metrics["nll"], metrics["mae"])
            if best is None or key < best[0]:
                best = (key, profile, metrics)
        results[objective] = {"profile": _profile_payload(best[1]), "validation_metrics": best[2]}
    return results


def _weights_from_profile(arrays: dict, profile_payload: dict | tuple[float, ...]) -> np.ndarray:
    profile = tuple(profile_payload[label] for label in GROUP_LABELS) if isinstance(profile_payload, dict) else profile_payload
    weights = np.zeros(len(arrays["actual"]), dtype=float)
    for group, value in zip(GROUPS, profile):
        weights[np.isin(arrays["issue_hour_utc"], group)] = value
    return weights


def _profile_payload(profile: tuple[float, ...]) -> dict:
    return {label: float(value) for label, value in zip(GROUP_LABELS, profile)}


def _metrics(arrays: dict, weights: np.ndarray) -> dict:
    if len(arrays["actual"]) == 0:
        return {}
    probabilities = (1.0 - weights[:, None]) * arrays["base"] + weights[:, None] * arrays["pure"]
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    cdf = np.cumsum(probabilities, axis=1)
    actual = arrays["actual"]
    expected = probabilities @ BINS
    actual_index = np.clip(np.rint(actual).astype(int) - BINS[0], 0, len(BINS) - 1)
    probability_actual = probabilities[np.arange(len(actual)), actual_index]
    lower_80 = BINS[(cdf >= 0.1).argmax(axis=1)]
    upper_80 = BINS[(cdf >= 0.9).argmax(axis=1)]
    obs_cdf = BINS[None, :] >= actual[:, None]
    return {
        "rows": int(len(actual)),
        "days": int(len(set(arrays["date"].tolist()))),
        "mae": float(np.mean(np.abs(expected - actual))),
        "rmse": float(np.sqrt(np.mean((expected - actual) ** 2))),
        "bias": float(np.mean(expected - actual)),
        "nll": float(np.mean(-np.log(np.clip(probability_actual, 1e-12, None)))),
        "crps": float(np.mean((cdf - obs_cdf) ** 2)),
        "brier_ge20": float(np.mean((probabilities[:, BINS >= 20].sum(axis=1) - (actual >= 20)) ** 2)),
        "brier_ge25": float(np.mean((probabilities[:, BINS >= 25].sum(axis=1) - (actual >= 25)) ** 2)),
        "brier_ge30": float(np.mean((probabilities[:, BINS >= 30].sum(axis=1) - (actual >= 30)) ** 2)),
        "coverage80": float(np.mean((lower_80 - 0.5 <= actual) & (actual < upper_80 + 0.5))),
        "mean_width80_c": float(np.mean(upper_80 - lower_80)),
        "mean_weight": float(np.mean(weights)),
    }


def _scenario_test_summary(test: dict, optimized: dict) -> dict:
    summary = {
        "prior": _metrics(test, np.zeros(len(test["actual"]))),
        "current_dynamic": _metrics(test, test["current_weight"]),
        "fixed_cool_challenger_from_previous_analysis": _metrics(test, _weights_from_profile(test, CHALLENGER_COOL)),
    }
    for objective, payload in optimized.items():
        summary[f"optimized_{objective}"] = _metrics(test, _weights_from_profile(test, payload["profile"]))
    return summary


def _by_issue_hour(test: dict, optimized: dict) -> dict:
    variants = {
        "prior": np.zeros(len(test["actual"])),
        "current_dynamic": test["current_weight"],
        "optimized_crps": _weights_from_profile(test, optimized["crps"]["profile"]),
        "optimized_mae": _weights_from_profile(test, optimized["mae"]["profile"]),
    }
    output = {}
    for name, weights in variants.items():
        rows = []
        for hour in sorted(set(test["issue_hour_utc"].tolist())):
            mask = test["issue_hour_utc"] == hour
            rows.append(
                {
                    "issue_hour_utc": int(hour),
                    "approx_local_winter_hour": int(hour + 1),
                    "approx_local_summer_hour": int(hour + 2),
                    "metrics": _metrics(_slice_arrays(test, mask), weights[mask]),
                }
            )
        output[name] = rows
    return output


def _regime_summary(test: dict, optimized: dict) -> dict:
    variants = {
        "prior": np.zeros(len(test["actual"])),
        "current_dynamic": test["current_weight"],
        "optimized_crps": _weights_from_profile(test, optimized["crps"]["profile"]),
        "optimized_mae": _weights_from_profile(test, optimized["mae"]["profile"]),
    }
    regimes = {
        "all": np.ones(len(test["actual"]), dtype=bool),
        "hot_ge25": test["actual"] >= 25,
        "hot_ge30": test["actual"] >= 30,
        "late_utc_12_18": test["issue_hour_utc"] >= 12,
        "drop_ge3": test["drop_from_observed_max_c"] >= 3,
        "drop_ge5": test["drop_from_observed_max_c"] >= 5,
        "precip_recent": test["has_precip_recent"],
        "late_drop_precip": (test["issue_hour_utc"] >= 12) & (test["drop_from_observed_max_c"] >= 3) & test["has_precip_recent"],
    }
    output = {}
    for regime_name, mask in regimes.items():
        output[regime_name] = {
            "rows": int(mask.sum()),
            "days": int(len(set(test["date"][mask].tolist()))) if mask.any() else 0,
            "variants": {name: _metrics(_slice_arrays(test, mask), weights[mask]) for name, weights in variants.items()} if mask.any() else {},
        }
    return output


def _late_override_sweep(validation: dict, test: dict, base_profile: dict) -> dict:
    candidates = []
    conditions = {
        "late_drop_ge3_precip": lambda arrays: (arrays["issue_hour_utc"] >= 12)
        & (arrays["drop_from_observed_max_c"] >= 3)
        & arrays["has_precip_recent"],
        "late_drop_ge5": lambda arrays: (arrays["issue_hour_utc"] >= 12) & (arrays["drop_from_observed_max_c"] >= 5),
        "late_peak_ge85": lambda arrays: (arrays["issue_hour_utc"] >= 12) & (arrays["peak_probability"] >= 0.85),
        "late_low_nwp_upside_drop_ge3": lambda arrays: (arrays["issue_hour_utc"] >= 12)
        & (arrays["drop_from_observed_max_c"] >= 3)
        & np.isfinite(arrays["nwp_future_upside_c"])
        & (arrays["nwp_future_upside_c"] <= 1.0),
    }
    base_validation_weights = _weights_from_profile(validation, base_profile)
    base_test_weights = _weights_from_profile(test, base_profile)
    for condition_name, condition in conditions.items():
        for override_weight in (0.65, 0.75, 0.85, 0.95):
            validation_weights = base_validation_weights.copy()
            validation_mask = condition(validation)
            validation_weights[validation_mask] = np.maximum(validation_weights[validation_mask], override_weight)
            test_weights = base_test_weights.copy()
            test_mask = condition(test)
            test_weights[test_mask] = np.maximum(test_weights[test_mask], override_weight)
            candidates.append(
                {
                    "condition": condition_name,
                    "override_weight": override_weight,
                    "validation_trigger_rows": int(validation_mask.sum()),
                    "test_trigger_rows": int(test_mask.sum()),
                    "validation_metrics": _metrics(validation, validation_weights),
                    "test_metrics": _metrics(test, test_weights),
                }
            )
    candidates.sort(key=lambda item: (item["validation_metrics"]["crps"], item["validation_metrics"]["nll"]))
    return {
        "base_profile_metrics_validation": _metrics(validation, base_validation_weights),
        "base_profile_metrics_test": _metrics(test, base_test_weights),
        "best_by_validation_crps": candidates[0] if candidates else None,
        "top_candidates": candidates[:6],
    }


def _paired_bootstrap(test: dict, candidate_weights: np.ndarray, baseline_weights: np.ndarray, n: int = 5000) -> dict:
    candidate_frame = _loss_frame(test, candidate_weights)
    baseline_frame = _loss_frame(test, baseline_weights)
    days = sorted(set(candidate_frame["date"].tolist()))
    by_day_candidate = {day: candidate_frame[candidate_frame["date"] == day] for day in days}
    by_day_baseline = {day: baseline_frame[baseline_frame["date"] == day] for day in days}
    rng = np.random.default_rng(20260601)
    values = {key: [] for key in ("mae", "rmse", "nll", "crps", "coverage80")}
    for _ in range(n):
        sample = rng.choice(days, size=len(days), replace=True)
        candidate = pd.concat([by_day_candidate[day] for day in sample], ignore_index=True)
        baseline = pd.concat([by_day_baseline[day] for day in sample], ignore_index=True)
        candidate_metrics = _loss_metrics(candidate)
        baseline_metrics = _loss_metrics(baseline)
        for key in values:
            values[key].append(candidate_metrics[key] - baseline_metrics[key])
    return {
        key: {
            "delta": float(np.mean(metric_values)),
            "ci95": [float(np.quantile(metric_values, 0.025)), float(np.quantile(metric_values, 0.975))],
        }
        for key, metric_values in values.items()
    }


def _loss_frame(test: dict, weights: np.ndarray) -> pd.DataFrame:
    probabilities = (1.0 - weights[:, None]) * test["base"] + weights[:, None] * test["pure"]
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    actual = test["actual"]
    cdf = np.cumsum(probabilities, axis=1)
    expected = probabilities @ BINS
    actual_index = np.clip(np.rint(actual).astype(int) - BINS[0], 0, len(BINS) - 1)
    lower_80 = BINS[(cdf >= 0.1).argmax(axis=1)]
    upper_80 = BINS[(cdf >= 0.9).argmax(axis=1)]
    return pd.DataFrame(
        {
            "date": test["date"],
            "absolute_error": np.abs(expected - actual),
            "squared_error": (expected - actual) ** 2,
            "nll": -np.log(np.clip(probabilities[np.arange(len(actual)), actual_index], 1e-12, None)),
            "crps": np.mean((cdf - (BINS[None, :] >= actual[:, None])) ** 2, axis=1),
            "coverage80": ((lower_80 - 0.5 <= actual) & (actual < upper_80 + 0.5)).astype(float),
        }
    )


def _loss_metrics(frame: pd.DataFrame) -> dict:
    return {
        "mae": frame["absolute_error"].mean(),
        "rmse": np.sqrt(frame["squared_error"].mean()),
        "nll": frame["nll"].mean(),
        "crps": frame["crps"].mean(),
        "coverage80": frame["coverage80"].mean(),
    }


def _slice_arrays(arrays: dict, mask: np.ndarray) -> dict:
    return {
        key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(mask) else value)
        for key, value in arrays.items()
        if key != "records"
    }


def _inventory(records: list[dict]) -> dict:
    if not records:
        return {"rows": 0}
    actual = np.array([record["actual"] for record in records])
    return {
        "rows": len(records),
        "days": len(set(record["target_date_local"] for record in records)),
        "ge20_rows": int((actual >= 20).sum()),
        "ge25_rows": int((actual >= 25).sum()),
        "ge30_rows": int((actual >= 30).sum()),
        "prior_train_rows_min": int(min(record["prior_train_rows"] for record in records)),
        "prior_train_rows_max": int(max(record["prior_train_rows"] for record in records)),
    }


def _global_notes(dataset: pd.DataFrame) -> dict:
    eligible = dataset[
        (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
        & dataset["model_tmax_c"].notna()
        & dataset["observed_max_so_far_from_metar"].notna()
        & dataset["last_metar_temp_c"].notna()
    ].copy()
    return {
        "eligible_start": str(eligible["target_date_local"].min()) if not eligible.empty else None,
        "eligible_end": str(eligible["target_date_local"].max()) if not eligible.empty else None,
        "eligible_rows": int(len(eligible)),
        "warning": "Open-Meteo Single Runs ICON-D2 temperature coverage before late July 2025 is not usable in the local archive.",
    }


def _build_doc(report: dict) -> str:
    lines = [
        "# Intraday seasonal weight analysis",
        "",
        "This is a research-only report. Production intraday weights were not changed.",
        "",
        "## Data Limits",
        "",
        f"- eligible period: `{report['global_notes']['eligible_start']}` to `{report['global_notes']['eligible_end']}`",
        f"- eligible rows: `{report['global_notes']['eligible_rows']}`",
        f"- warning: {report['global_notes']['warning']}",
        "",
    ]
    for scenario_name, scenario in report["scenarios"].items():
        lines.extend(_scenario_doc(scenario_name, scenario))
    lines.extend(
        [
            "## Recommendation",
            "",
            "- Do not replace production weights with a single year-round profile.",
            "- Keep the current aggressive warm-season behavior as a useful baseline, but evaluate a CRPS-oriented warm challenger in shadow mode.",
            "- For cool months, prefer zero morning influence and moderate afternoon weights as the next challenger.",
            "- Implement seasonal profiles only after a shadow-mode comparison on live 2026 summer data or a larger issued archive.",
        ]
    )
    return "\n".join(lines)


def _scenario_doc(name: str, scenario: dict) -> list[str]:
    lines = [
        f"## {name.title()} Scenario",
        "",
        scenario["description"],
        "",
        f"- validation period: `{scenario['validation_period'][0]}` to `{scenario['validation_period'][1]}`",
        f"- test period: `{scenario['test_period'][0]}` to `{scenario['test_period'][1]}`",
        f"- validation inventory: `{scenario['validation_inventory']}`",
        f"- test inventory: `{scenario['test_inventory']}`",
        "",
        "Optimized validation profiles:",
        "",
        "| objective | utc_00_03 | utc_06_09 | utc_12 | utc_15_18 | validation MAE | validation CRPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for objective, payload in scenario["optimized_on_validation"].items():
        profile = payload["profile"]
        metrics = payload["validation_metrics"]
        lines.append(
            f"| {objective} | {profile['utc_00_03']:.2f} | {profile['utc_06_09']:.2f} | {profile['utc_12']:.2f} | {profile['utc_15_18']:.2f} | {metrics['mae']:.4f} | {metrics['crps']:.5f} |"
        )
    lines.extend(
        [
            "",
            "Test summary:",
            "",
            "| variant | rows | MAE | RMSE | NLL | CRPS | Brier >=30 | coverage80 | mean weight |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, metrics in scenario["test_summary"].items():
        lines.append(
            f"| {variant} | {metrics['rows']} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | {metrics['nll']:.4f} | {metrics['crps']:.5f} | {metrics['brier_ge30']:.5f} | {metrics['coverage80']:.4f} | {metrics['mean_weight']:.4f} |"
        )
    override = scenario["late_override_sweep"]["best_by_validation_crps"]
    if override:
        lines.extend(
            [
                "",
                "Best late override candidate by validation CRPS:",
                "",
                f"- condition: `{override['condition']}`",
                f"- override weight: `{override['override_weight']}`",
                f"- validation trigger rows: `{override['validation_trigger_rows']}`",
                f"- test trigger rows: `{override['test_trigger_rows']}`",
            ]
        )
    lines.append("")
    return lines


def _console_summary(report: dict) -> dict:
    return {
        name: {
            "validation_inventory": scenario["validation_inventory"],
            "test_inventory": scenario["test_inventory"],
            "optimized_profiles": {
                objective: payload["profile"] for objective, payload in scenario["optimized_on_validation"].items()
            },
            "test_summary": {
                variant: {
                    key: metrics[key]
                    for key in ("mae", "rmse", "nll", "crps", "brier_ge30", "coverage80", "mean_weight")
                }
                for variant, metrics in scenario["test_summary"].items()
            },
            "best_override": scenario["late_override_sweep"]["best_by_validation_crps"],
        }
        for name, scenario in report["scenarios"].items()
    }


if __name__ == "__main__":
    main()
