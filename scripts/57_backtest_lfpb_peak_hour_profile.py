from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.distribution import TmaxDistribution


THRESHOLDS = tuple(range(1, 13))


def main() -> None:
    args = _parse_args()
    dataset = _prepare_dataset(pd.read_parquet(args.dataset))
    metar = _prepare_metar(pd.read_parquet(args.metar))
    metadata = _load_json(args.metadata)
    model = joblib.load(args.model)

    split = metadata.get("split") or {}
    test_start = split.get("test_start")
    test_end = split.get("test_end")
    if not test_start or not test_end:
        raise ValueError("metadata split must include test_start and test_end")

    train_metar = metar[metar["target_date_local"] < test_start].copy()
    daily_peak = build_daily_peak_profile_input(train_metar, timezone=args.timezone, min_obs_count=args.min_obs_count)
    hourly_profile_rows = build_hourly_remaining_upside_profile(
        train_metar,
        timezone=args.timezone,
        min_obs_count=args.min_obs_count,
        max_threshold=max(THRESHOLDS),
    )
    profile = summarize_hourly_profile(hourly_profile_rows)

    test = dataset[(dataset["target_date_local"] >= test_start) & (dataset["target_date_local"] <= test_end)].copy()
    if test.empty:
        raise ValueError("no holdout rows found")

    variants = _variant_grid()
    scored_rows = []
    for _, row in test.iterrows():
        base = model.predict_distribution(row)
        scored_rows.append(_score("base", row, base))
        for variant in variants:
            adjusted = apply_peak_hour_profile_adjustment(
                base,
                row,
                profile=profile,
                strength=variant["strength"],
                mode=variant["mode"],
                use_context_caps=variant["use_context_caps"],
            )
            scored_rows.append(_score(variant["name"], row, adjusted))

    scored = pd.DataFrame(scored_rows)
    summary = _group_summary(scored, ["model_variant"]).sort_values("mae_expected")
    by_hour = _group_summary(scored, ["model_variant", "local_issue_hour"])
    by_season = _group_summary(scored, ["model_variant", "season"])
    peak_hour_summary = summarize_peak_hours(daily_peak)
    profile_table = profile_to_table(profile)
    recommendation = _recommend(summary, by_hour)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(output / "lfpb_peak_hour_profile_backtest_rows.parquet", index=False)
    summary.to_csv(output / "lfpb_peak_hour_profile_backtest_summary.csv", index=False)
    by_hour.to_csv(output / "lfpb_peak_hour_profile_backtest_by_hour.csv", index=False)
    by_season.to_csv(output / "lfpb_peak_hour_profile_backtest_by_season.csv", index=False)
    peak_hour_summary.to_csv(output / "lfpb_peak_hour_profile_peak_hour_summary.csv", index=False)
    profile_table.to_csv(output / "lfpb_peak_hour_profile_survival_table.csv", index=False)

    report = {
        "analysis": "LFPB historical peak-hour profile and intraday survival replay.",
        "test_period": [test_start, test_end],
        "test_rows": int(len(test)),
        "test_days": int(test["target_date_local"].nunique()),
        "profile_training_period": [
            str(train_metar["target_date_local"].min()),
            str(train_metar["target_date_local"].max()),
        ],
        "profile_training_days": int(hourly_profile_rows["target_date_local"].nunique()),
        "leakage_policy": "Peak-hour and remaining-upside profile is built only from METAR days before holdout test_start.",
        "recommendation": recommendation,
        "summary": json.loads(summary.to_json(orient="records")),
        "created_outputs": {
            "rows": str(output / "lfpb_peak_hour_profile_backtest_rows.parquet"),
            "summary": str(output / "lfpb_peak_hour_profile_backtest_summary.csv"),
            "by_hour": str(output / "lfpb_peak_hour_profile_backtest_by_hour.csv"),
            "peak_hour_summary": str(output / "lfpb_peak_hour_profile_peak_hour_summary.csv"),
            "survival_table": str(output / "lfpb_peak_hour_profile_survival_table.csv"),
        },
    }
    (output / "lfpb_peak_hour_profile_backtest.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def build_daily_peak_profile_input(metar: pd.DataFrame, *, timezone: str, min_obs_count: int) -> pd.DataFrame:
    rows = []
    for target_date, day in metar.groupby("target_date_local", sort=True):
        day = day.sort_values("local_time")
        if len(day) < min_obs_count:
            continue
        tmax = float(day["temperature_c"].max())
        first = day[day["temperature_c"] == tmax].iloc[0]
        rows.append(
            {
                "target_date_local": target_date,
                "season": _season(pd.Timestamp(target_date).month),
                "tmax_c": tmax,
                "first_max_time_local": first["local_time"].isoformat(),
                "first_max_hour_local": int(first["local_time"].hour),
                "obs_count": int(len(day)),
            }
        )
    return pd.DataFrame(rows)


def build_hourly_remaining_upside_profile(
    metar: pd.DataFrame,
    *,
    timezone: str,
    min_obs_count: int,
    max_threshold: int,
) -> pd.DataFrame:
    rows = []
    for target_date, day in metar.groupby("target_date_local", sort=True):
        day = day.sort_values("local_time")
        if len(day) < min_obs_count:
            continue
        final_tmax = float(day["temperature_c"].max())
        season = _season(pd.Timestamp(target_date).month)
        for hour in range(24):
            issue_local = _safe_local_timestamp(target_date, hour, timezone)
            so_far = day[day["local_time"] <= issue_local]
            if so_far.empty:
                continue
            current_max = float(so_far["temperature_c"].max())
            latest_temp = float(so_far.iloc[-1]["temperature_c"])
            remaining = max(0.0, final_tmax - current_max)
            row = {
                "target_date_local": target_date,
                "season": season,
                "local_hour": hour,
                "final_tmax_c": final_tmax,
                "current_max_c": current_max,
                "latest_temp_c": latest_temp,
                "drop_from_current_max_c": current_max - latest_temp,
                "remaining_upside_c": remaining,
                "obs_count": int(len(day)),
            }
            for threshold in range(1, max_threshold + 1):
                row[f"upside_ge_{threshold}c"] = bool(remaining >= threshold)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_hourly_profile(hourly_rows: pd.DataFrame) -> dict:
    profile = {}
    for (season, hour), group in hourly_rows.groupby(["season", "local_hour"], dropna=False):
        key = (str(season), int(hour))
        profile[key] = {
            "rows": int(len(group)),
            "mean_remaining_upside_c": float(group["remaining_upside_c"].mean()),
            "survival": {
                threshold: float(group[f"upside_ge_{threshold}c"].mean())
                for threshold in THRESHOLDS
                if f"upside_ge_{threshold}c" in group.columns
            },
        }
    return profile


def summarize_peak_hours(daily_peak: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in sorted(daily_peak["season"].dropna().unique()):
        group = daily_peak[daily_peak["season"] == season]
        for hour in range(24):
            count = int((group["first_max_hour_local"] == hour).sum())
            rows.append(
                {
                    "season": season,
                    "local_hour": hour,
                    "days": int(len(group)),
                    "first_peak_count": count,
                    "first_peak_share": float(count / len(group)) if len(group) else np.nan,
                    "peak_after_hour_share": float((group["first_max_hour_local"] > hour).mean()) if len(group) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def profile_to_table(profile: dict) -> pd.DataFrame:
    rows = []
    for (season, hour), item in sorted(profile.items()):
        row = {
            "season": season,
            "local_hour": hour,
            "rows": item["rows"],
            "mean_remaining_upside_c": item["mean_remaining_upside_c"],
        }
        for threshold, value in item["survival"].items():
            row[f"p_upside_ge_{threshold}c"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def apply_peak_hour_profile_adjustment(
    distribution: TmaxDistribution,
    row: pd.Series,
    *,
    profile: dict,
    strength: float,
    mode: str,
    use_context_caps: bool,
) -> TmaxDistribution:
    if not _variant_active(row, mode):
        return distribution
    observed_bin = int(np.ceil(float(row["current_metar_max_c"])))
    base = _truncate_below_observed_bin(distribution, observed_bin)
    max_upside = max(1, int(base.bins_c.max() - observed_bin))
    original = _distribution_to_survival(base, observed_bin, max_upside)
    priors = _lookup_profile(profile, str(row["season"]), float(row["local_issue_hour"]), max_upside)
    caps = _context_caps(row, max_upside) if use_context_caps else {threshold: 1.0 for threshold in range(1, max_upside + 1)}
    target = {threshold: min(priors.get(threshold, 1.0), caps.get(threshold, 1.0)) for threshold in range(1, max_upside + 1)}
    effective_strength = float(np.clip(strength * _phase_strength(float(row["local_issue_hour"])), 0.0, 1.0))
    adjusted = {
        threshold: original[threshold] - effective_strength * max(0.0, original[threshold] - target[threshold])
        for threshold in range(1, max_upside + 1)
    }
    values = np.minimum.accumulate(np.clip([adjusted[t] for t in range(1, max_upside + 1)], 0.0, 1.0))
    return _survival_to_distribution({idx + 1: float(value) for idx, value in enumerate(values)}, observed_bin)


def _variant_active(row: pd.Series, mode: str) -> bool:
    hour = float(row["local_issue_hour"])
    if mode == "full":
        return True
    if mode == "after18":
        return hour >= 18
    if mode == "after16_strong_or_after18":
        if hour >= 18:
            return True
        if hour < 16:
            return False
        future_delta = _optional_float(row, "nwp_future_minus_current_max_c")
        trend_3h = _optional_float(row, "temp_trend_3h")
        return bool(
            float(row.get("drop_from_current_max_c", 0.0) or 0.0) >= 1.0
            or bool(row.get("has_rain_recent_metar", False))
            or (future_delta is not None and future_delta <= 0.5)
            or (trend_3h is not None and trend_3h <= 0.0)
        )
    raise ValueError(f"unknown peak-hour profile mode: {mode}")


def _lookup_profile(profile: dict, season: str, local_hour: float, max_upside: int) -> dict[int, float]:
    hour = int(np.floor(local_hour))
    candidates = [
        profile.get((season, hour)),
        profile.get((season, min(hour + 1, 23))),
    ]
    if candidates[0] is None:
        available = [item for (profile_season, _), item in profile.items() if profile_season == season]
        candidates[0] = available[0] if available else None
    if candidates[0] is None:
        return {threshold: 1.0 for threshold in range(1, max_upside + 1)}
    lower = candidates[0]["survival"]
    upper = candidates[1]["survival"] if candidates[1] is not None else lower
    weight = float(np.clip(local_hour - hour, 0.0, 1.0))
    return {
        threshold: float((1.0 - weight) * lower.get(threshold, 0.0) + weight * upper.get(threshold, 0.0))
        for threshold in range(1, max_upside + 1)
    }


def _context_caps(row: pd.Series, max_upside: int) -> dict[int, float]:
    future_delta = _optional_float(row, "nwp_future_minus_current_max_c")
    drop = float(row.get("drop_from_current_max_c", 0.0) or 0.0)
    trend_1h = _optional_float(row, "temp_trend_1h")
    trend_3h = _optional_float(row, "temp_trend_3h")
    rain_recent = bool(row.get("has_rain_recent_metar", False))
    hour = float(row.get("local_issue_hour", 12.0))
    caps = {}
    for threshold in range(1, max_upside + 1):
        nwp_cap = 1.0 if future_delta is None else _sigmoid((future_delta - (threshold - 0.35)) / 0.70)
        multiplier = 1.0
        if drop >= 1.0:
            multiplier *= float(np.exp(-0.22 * min(drop, 6.0)))
        if trend_1h is not None and trend_1h <= -1.0:
            multiplier *= 0.80
        if trend_3h is not None and trend_3h <= -2.0:
            multiplier *= 0.75
        if rain_recent:
            multiplier *= 0.70 if drop >= 1.0 else 0.85
        if hour >= 18:
            multiplier *= 0.75
        caps[threshold] = float(np.clip(nwp_cap * multiplier, 0.0, 1.0))
    return caps


def _variant_grid() -> list[dict]:
    variants = []
    for strength in (0.25, 0.50, 0.75, 0.92):
        for mode in ("full", "after18", "after16_strong_or_after18"):
            for use_context in (False, True):
                variants.append(
                    {
                        "name": f"peak_profile_{mode}_{'context' if use_context else 'prior'}_s{strength:.2f}",
                        "mode": mode,
                        "use_context_caps": use_context,
                        "strength": strength,
                    }
                )
    return variants


def _score(model_variant: str, row: pd.Series, dist: TmaxDistribution) -> dict:
    actual = float(row["final_metar_tmax_c"])
    current_max = float(row["current_metar_max_c"])
    return {
        "model_variant": model_variant,
        "target_date_local": str(row["target_date_local"]),
        "local_issue_hour": int(row["local_issue_hour"]),
        "season": row.get("season"),
        "actual_metar_tmax_c": actual,
        "current_metar_max_c": current_max,
        "remaining_upside_c": float(row["remaining_upside_c"]),
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "probability_actual_integer_bin": _probability_for_bin(dist, int(round(actual))),
        "probability_ge_current_plus_1c": dist.threshold_ge(int(np.ceil(current_max + 1))),
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
                "mean_probability_actual_integer_bin": float(group["probability_actual_integer_bin"].mean()),
                "mean_probability_ge_current_plus_1c": float(group["probability_ge_current_plus_1c"].mean()),
                "brier_upside_ge_1c": float(group["brier_upside_ge_1c"].mean()),
                "brier_upside_ge_2c": float(group["brier_upside_ge_2c"].mean()),
                "brier_upside_ge_3c": float(group["brier_upside_ge_3c"].mean()),
                "coverage_80": float(group["coverage_80"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _recommend(summary: pd.DataFrame, by_hour: pd.DataFrame) -> dict:
    base = summary[summary["model_variant"] == "base"].iloc[0]
    candidates = summary[summary["model_variant"] != "base"].copy()
    candidates["mae_delta"] = candidates["mae_expected"] - float(base["mae_expected"])
    candidates["nll_delta"] = candidates["mean_nll"] - float(base["mean_nll"])
    candidates["brier_ge1_delta"] = candidates["brier_upside_ge_1c"] - float(base["brier_upside_ge_1c"])
    safe = candidates[
        (candidates["mae_delta"] < 0)
        & (candidates["nll_delta"] <= 0.02)
        & (candidates["brier_ge1_delta"] <= 0.01)
    ].copy()
    if safe.empty:
        best = candidates.sort_values(["mae_delta", "mean_nll"]).iloc[0]
        return {
            "production_recommendation": "do_not_promote_peak_hour_profile_yet",
            "reason": "No candidate improved MAE while keeping NLL and +1C Brier within safety gates.",
            "best_research_candidate": json.loads(best.to_json()),
        }
    best = safe.sort_values(["mae_expected", "mean_nll"]).iloc[0]
    return {
        "production_recommendation": "promote_only_as_time_gated_post_processor",
        "recommended_variant": best["model_variant"],
        "reason": "Candidate improves MAE and keeps probabilistic metrics within safety gates on the holdout.",
        "metrics": json.loads(best.to_json()),
    }


def _prepare_metar(metar: pd.DataFrame) -> pd.DataFrame:
    frame = metar.copy()
    frame["observation_time_utc"] = pd.to_datetime(frame["observation_time_utc"], utc=True, errors="coerce")
    frame["temperature_c"] = pd.to_numeric(frame["temperature_c"], errors="coerce")
    frame = frame.dropna(subset=["observation_time_utc", "temperature_c"]).copy()
    frame["local_time"] = frame["observation_time_utc"].dt.tz_convert("Europe/Paris")
    frame["target_date_local"] = frame["local_time"].dt.date.astype(str)
    return frame.sort_values("observation_time_utc")


def _safe_local_timestamp(target_date: str, hour: int, timezone: str) -> pd.Timestamp:
    naive = pd.Timestamp(f"{target_date} {hour:02d}:00:00")
    localized = pd.DatetimeIndex([naive]).tz_localize(
        timezone,
        nonexistent="shift_forward",
        ambiguous="NaT",
    )[0]
    if pd.isna(localized):
        localized = pd.DatetimeIndex([naive]).tz_localize(
            timezone,
            nonexistent="shift_forward",
            ambiguous=True,
        )[0]
    return pd.Timestamp(localized)


def _prepare_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    frame["target_date_local"] = frame["target_date_local"].astype(str)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["season"] = pd.to_datetime(frame["target_date_local"], errors="coerce").dt.month.map(_season)
    frame = frame[frame["leakage_check_passed"].fillna(False).astype(bool)].copy()
    frame = frame.dropna(subset=["final_metar_tmax_c", "current_metar_max_c", "model_tmax_c"])
    return frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)


def _distribution_to_survival(dist: TmaxDistribution, observed_bin: int, max_upside: int) -> dict[int, float]:
    return {
        threshold: float(dist.probabilities[dist.bins_c >= observed_bin + threshold].sum())
        for threshold in range(1, max_upside + 1)
    }


def _survival_to_distribution(survival: dict[int, float], observed_bin: int) -> TmaxDistribution:
    max_upside = max(survival)
    values = np.minimum.accumulate(np.clip([survival[t] for t in range(1, max_upside + 1)], 0.0, 1.0))
    probs = np.empty(max_upside + 1, dtype=float)
    probs[0] = 1.0 - values[0]
    if max_upside > 1:
        probs[1:-1] = values[:-1] - values[1:]
    probs[-1] = values[-1]
    return TmaxDistribution(np.arange(observed_bin, observed_bin + max_upside + 1), probs)


def _truncate_below_observed_bin(distribution: TmaxDistribution, observed_bin: int) -> TmaxDistribution:
    probs = distribution.probabilities.copy()
    below = distribution.bins_c < observed_bin
    removed = float(probs[below].sum())
    probs[below] = 0.0
    mask = distribution.bins_c == observed_bin
    if mask.any():
        probs[mask] += removed
        return TmaxDistribution(distribution.bins_c, probs)
    bins = np.arange(observed_bin, max(int(distribution.bins_c.max()), observed_bin) + 1)
    new_probs = np.zeros(len(bins), dtype=float)
    lookup = {int(bin_c): float(prob) for bin_c, prob in zip(distribution.bins_c, probs)}
    for idx, bin_c in enumerate(bins):
        new_probs[idx] = lookup.get(int(bin_c), 0.0)
    new_probs[0] += removed
    return TmaxDistribution(bins, new_probs)


def _phase_strength(hour: float) -> float:
    if hour < 10:
        return 0.05
    if hour < 12:
        return 0.12
    if hour < 14:
        return 0.22
    if hour < 16:
        return 0.38
    if hour < 18:
        return 0.65
    return 0.95


def _probability_for_bin(dist: TmaxDistribution, actual_bin: int) -> float:
    mask = dist.bins_c == actual_bin
    return float(dist.probabilities[mask].sum()) if mask.any() else 0.0


def _covered(dist: TmaxDistribution, actual: float, mass: float) -> bool:
    low, high = dist.interval(mass)
    return bool(low <= actual <= high)


def _optional_float(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    return float(value)


def _season(month: int | float | None) -> str:
    if month is None or pd.isna(month):
        return "unknown"
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest LFPB peak-hour profile as an intraday post-processor.")
    parser.add_argument("--dataset", default="data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")
    parser.add_argument("--metar", default="data/interim/metar_iem_LFPB.parquet")
    parser.add_argument("--model", default="data/models/lfpb_metar_tmax_icon_d2_v1.joblib")
    parser.add_argument("--metadata", default="data/reports/lfpb_icon_d2_metar_tmax_training.json")
    parser.add_argument("--timezone", default="Europe/Paris")
    parser.add_argument("--min-obs-count", type=int, default=36)
    parser.add_argument("--output-dir", default="data/reports")
    return parser.parse_args()


if __name__ == "__main__":
    main()
