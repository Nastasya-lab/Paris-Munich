from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_intraday_survival import apply_metar_intraday_survival_layer


DATASET_PATH = Path("data/processed/metar_upside_dataset_LFPB_icon_d2_spatial_advection.parquet")
PRODUCTION_MODEL_PATH = Path("data/models/lfpb_metar_tmax_icon_d2_v1.joblib")
OUTPUT_MODEL_PATH = Path("data/models/lfpb_after_dip_remaining_upside_candidate_v1.joblib")
REPORT_DIR = Path("data/reports")
MAX_UPSIDE_C = 8


FEATURE_COLUMNS = [
    "local_issue_hour",
    "current_metar_max_c",
    "latest_metar_temp_c",
    "drop_from_current_max_c",
    "temp_trend_1h",
    "temp_trend_3h",
    "temp_trend_6h",
    "temp_trend_last_2_metars",
    "latest_2_metar_temp_change_c",
    "temp_slope_since_sunrise",
    "has_rain_recent_metar",
    "rain_started_after_current_max",
    "cb_tcu_appeared_after_current_max",
    "showers_appeared_after_current_max",
    "cloud_cover_proxy_latest",
    "cloud_cover_proxy_trend_2h",
    "dewpoint_depression_latest",
    "dewpoint_depression_trend_2h",
    "pressure_tendency_1h",
    "pressure_tendency_3h",
    "wind_dir_shift_2h_deg",
    "wind_speed_trend_2h",
    "metar_minutes_since_current_max",
    "temp_drop_after_rain_start_c",
    "temp_drop_after_cb_tcu_c",
    "rain_mm_last_30m",
    "rain_mm_last_1h",
    "rain_mm_last_3h",
    "rain_max_6min_last_3h",
    "model_tmax_c",
    "model_future_temp_max_c",
    "nwp_model_minus_current_max_c",
    "nwp_future_minus_current_max_c",
    "model_future_cloud_cover_mean",
    "model_future_precip_sum",
    "model_future_shortwave_radiation_sum",
    "model_future_wind_speed_max",
    "model_future_gust_max",
    "spatial_available_station_count",
    "spatial_latest_temp_mean_c",
    "spatial_latest_temp_max_c",
    "spatial_current_max_mean_c",
    "spatial_current_max_max_c",
    "spatial_latest_minus_lfpb_latest_mean_c",
    "spatial_max_minus_lfpb_current_max_mean_c",
    "spatial_any_neighbor_above_lfpb_latest",
    "spatial_any_neighbor_above_lfpb_current_max",
    "adv_available_station_count",
    "adv_mean_wind_speed_latest_kt",
    "adv_mean_temp_trend_1h",
    "adv_mean_temp_trend_3h",
    "adv_mean_dewpoint_trend_3h",
    "adv_mean_pressure_tendency_3h",
    "adv_any_cold_advection_signal",
    "adv_any_warm_advection_signal",
    "adv_any_frontal_passage_signal",
    "adv_neighbor_mean_minus_lfpb_temp_trend_1h",
]


@dataclass
class AfterDipUpsideCandidate:
    feature_columns: list[str]
    imputer: SimpleImputer
    threshold_models: dict[int, HistGradientBoostingClassifier | None]
    constant_probabilities: dict[int, float]
    max_upside_c: int = MAX_UPSIDE_C
    model_version: str = "lfpb_after_dip_remaining_upside_candidate_v1"

    def predict_survival_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self.imputer.transform(_numeric_features(frame, self.feature_columns))
        cols = {}
        for threshold in range(1, self.max_upside_c + 1):
            model = self.threshold_models.get(threshold)
            if model is None:
                probs = np.full(len(frame), self.constant_probabilities.get(threshold, 0.0), dtype=float)
            else:
                probs = model.predict_proba(X)[:, 1].astype(float)
            cols[threshold] = probs
        matrix = np.column_stack([cols[t] for t in range(1, self.max_upside_c + 1)])
        matrix = np.minimum.accumulate(np.clip(matrix, 0.0, 1.0), axis=1)
        return pd.DataFrame(
            matrix,
            index=frame.index,
            columns=[f"probability_upside_ge_{threshold}c" for threshold in range(1, self.max_upside_c + 1)],
        )

    def predict_distribution_frame(self, frame: pd.DataFrame) -> list[TmaxDistribution]:
        survival = self.predict_survival_frame(frame)
        out = []
        for index, row in frame.iterrows():
            surv = {
                threshold: float(survival.loc[index, f"probability_upside_ge_{threshold}c"])
                for threshold in range(1, self.max_upside_c + 1)
            }
            probs = _survival_to_probabilities(surv, self.max_upside_c)
            observed_bin = int(np.rint(float(row["current_metar_max_c"])))
            out.append(TmaxDistribution(np.arange(observed_bin, observed_bin + self.max_upside_c + 1), probs))
        return out


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = _load_dataset(DATASET_PATH)
    dataset = _add_context_flags(dataset)
    risk = dataset[dataset["after_dip_risk_context"]].copy()
    train, test = _time_split(risk)

    candidate = _fit_candidate(train)
    joblib.dump(candidate, OUTPUT_MODEL_PATH)

    candidate_rows = _score_distributions(test, candidate.predict_distribution_frame(test), "after_dip_ml_candidate")
    prior_rows = _score_distributions(test, _prior_distributions(train, test), "historical_context_prior")
    production_rows = _score_production(test)
    scored = pd.concat([production_rows, prior_rows, candidate_rows], ignore_index=True)

    summary = _summary(scored)
    by_hour = _by_group(scored, "local_issue_hour")
    by_context = _by_group(scored, "risk_context")
    context_counts = _context_counts(dataset)
    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "model_path": str(OUTPUT_MODEL_PATH),
        "rows_total": int(len(dataset)),
        "days_total": int(dataset["target_date_local"].nunique()),
        "risk_rows_total": int(len(risk)),
        "risk_days_total": int(risk["target_date_local"].nunique()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_days": int(train["target_date_local"].nunique()),
        "test_days": int(test["target_date_local"].nunique()),
        "feature_columns": FEATURE_COLUMNS,
        "summary": json.loads(summary.to_json(orient="records")),
        "by_hour": json.loads(by_hour.to_json(orient="records")),
        "by_context": json.loads(by_context.to_json(orient="records")),
        "context_counts": context_counts,
        "recommendation": _recommendation(summary, len(test)),
    }

    scored.to_parquet(REPORT_DIR / "lfpb_after_dip_upside_candidate_rows.parquet", index=False)
    summary.to_csv(REPORT_DIR / "lfpb_after_dip_upside_candidate_summary.csv", index=False)
    by_hour.to_csv(REPORT_DIR / "lfpb_after_dip_upside_candidate_by_hour.csv", index=False)
    by_context.to_csv(REPORT_DIR / "lfpb_after_dip_upside_candidate_by_context.csv", index=False)
    (REPORT_DIR / "lfpb_after_dip_upside_candidate.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(report, REPORT_DIR / "lfpb_after_dip_upside_candidate.md")
    print(json.dumps(report, indent=2, default=str))


def _load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame["target_date_local"] = frame["target_date_local"].astype(str)
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["remaining_upside_c"] = pd.to_numeric(frame["remaining_upside_c"], errors="coerce").clip(lower=0.0)
    frame = frame.dropna(subset=["issue_time_utc", "remaining_upside_c", "current_metar_max_c"])
    if "leakage_check_passed" in frame:
        frame = frame[frame["leakage_check_passed"].fillna(False).astype(bool)].copy()
    return frame.sort_values(["target_date_local", "issue_time_utc"]).reset_index(drop=True)


def _add_context_flags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    hour = _num(out, "local_issue_hour")
    convective = (
        _bool(out, "has_rain_recent_metar")
        | _bool(out, "rain_started_after_current_max")
        | _bool(out, "cb_tcu_appeared_after_current_max")
        | _bool(out, "showers_appeared_after_current_max")
    )
    dip = (
        (_num(out, "drop_from_current_max_c") >= 1.0)
        | (_num(out, "temp_drop_after_rain_start_c") >= 1.0)
        | (_num(out, "temp_drop_after_cb_tcu_c") >= 1.0)
    )
    rebound = (
        (_num(out, "temp_trend_1h") >= 1.5)
        | (_num(out, "temp_trend_last_2_metars") >= 1.5)
        | (_num(out, "latest_2_metar_temp_change_c") >= 1.5)
        | (_num(out, "temp_trend_3h") >= 2.0)
    )
    recent_max = _num(out, "metar_minutes_since_current_max").fillna(999.0) <= 45.0
    midday = hour.between(11.5, 16.0)
    out["convective_signal"] = convective
    out["dip_signal"] = dip
    out["rebound_signal"] = rebound
    out["recent_or_new_max_signal"] = recent_max
    out["after_dip_risk_context"] = midday & (convective | dip | rebound)
    out["risk_context"] = np.select(
        [
            midday & convective & rebound,
            midday & convective & dip,
            midday & rebound,
            midday & dip,
            midday & convective,
        ],
        ["convective_rebound", "convective_dip", "rebound_only", "dip_only", "convective_only"],
        default="other_midday_risk",
    )
    return out


def _time_split(frame: pd.DataFrame, test_fraction: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = np.array(sorted(frame["target_date_local"].unique()))
    split = max(1, int(np.floor(len(days) * (1.0 - test_fraction))))
    train_days = set(days[:split])
    train = frame[frame["target_date_local"].isin(train_days)].copy()
    test = frame[~frame["target_date_local"].isin(train_days)].copy()
    if train.empty or test.empty:
        raise ValueError("not enough after-dip risk rows for time split")
    return train, test


def _fit_candidate(train: pd.DataFrame) -> AfterDipUpsideCandidate:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X = imputer.fit_transform(_numeric_features(train, FEATURE_COLUMNS))
    upside = train["remaining_upside_c"].to_numpy(dtype=float)
    models = {}
    constants = {}
    for threshold in range(1, MAX_UPSIDE_C + 1):
        y = (upside >= threshold).astype(int)
        if np.unique(y).size < 2:
            models[threshold] = None
            constants[threshold] = float(y.mean())
            continue
        model = HistGradientBoostingClassifier(
            learning_rate=0.045,
            max_iter=80,
            max_leaf_nodes=9,
            min_samples_leaf=18,
            l2_regularization=1.5,
            random_state=173,
        )
        model.fit(X, y)
        models[threshold] = model
    return AfterDipUpsideCandidate(FEATURE_COLUMNS, imputer, models, constants)


def _score_production(test: pd.DataFrame) -> pd.DataFrame:
    if not PRODUCTION_MODEL_PATH.exists():
        return pd.DataFrame()
    model = joblib.load(PRODUCTION_MODEL_PATH)
    distributions = []
    for _, row in test.iterrows():
        raw = model.predict_distribution(row.to_dict())
        adjusted = apply_metar_intraday_survival_layer(raw, row.to_dict(), historical_dataset=DATASET_PATH and pd.read_parquet(DATASET_PATH))
        distributions.append(adjusted.distribution)
    return _score_distributions(test, distributions, "current_production_with_survival")


def _prior_distributions(train: pd.DataFrame, test: pd.DataFrame) -> list[TmaxDistribution]:
    priors = {}
    fallback = train["remaining_upside_c"].to_numpy(dtype=float)
    for key, group in train.groupby(["local_issue_hour", "risk_context"]):
        priors[key] = group["remaining_upside_c"].to_numpy(dtype=float)
    by_hour = {hour: group["remaining_upside_c"].to_numpy(dtype=float) for hour, group in train.groupby("local_issue_hour")}
    out = []
    for _, row in test.iterrows():
        samples = priors.get((row["local_issue_hour"], row["risk_context"]))
        if samples is None or len(samples) < 20:
            samples = by_hour.get(row["local_issue_hour"])
        if samples is None or len(samples) == 0:
            samples = fallback
        rounded = np.rint(float(row["current_metar_max_c"]) + np.clip(samples, 0.0, MAX_UPSIDE_C)).astype(int)
        bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
        probs = np.array([(rounded == bin_c).mean() for bin_c in bins], dtype=float)
        out.append(TmaxDistribution(bins, probs))
    return out


def _score_distributions(frame: pd.DataFrame, distributions: list[TmaxDistribution], variant: str) -> pd.DataFrame:
    rows = []
    for (_, row), dist in zip(frame.iterrows(), distributions, strict=True):
        actual = float(row["final_metar_tmax_c"])
        actual_bin = int(np.rint(actual))
        current_max = float(row["current_metar_max_c"])
        p_actual = _prob_at(dist, actual_bin)
        rows.append(
            {
                "variant": variant,
                "target_date_local": row["target_date_local"],
                "issue_time_utc": row["issue_time_utc"],
                "local_issue_hour": float(row["local_issue_hour"]),
                "risk_context": row["risk_context"],
                "final_metar_tmax_c": actual,
                "current_metar_max_c": current_max,
                "remaining_upside_c": float(row["remaining_upside_c"]),
                "expected_tmax_c": float(dist.expected_tmax_c),
                "most_likely_integer_c": int(dist.most_likely_integer_c),
                "abs_error_c": abs(float(dist.expected_tmax_c) - actual),
                "mode_abs_error_c": abs(int(dist.most_likely_integer_c) - actual_bin),
                "p_actual_bin": p_actual,
                "nll_actual_bin": -float(np.log(max(p_actual, 1e-6))),
                "brier_upside_ge_1c": _brier(dist.threshold_ge(int(np.ceil(current_max + 1))), row["remaining_upside_c"] >= 1.0),
                "brier_upside_ge_2c": _brier(dist.threshold_ge(int(np.ceil(current_max + 2))), row["remaining_upside_c"] >= 2.0),
                "brier_upside_ge_3c": _brier(dist.threshold_ge(int(np.ceil(current_max + 3))), row["remaining_upside_c"] >= 3.0),
                "p_upside_ge_1c": float(dist.threshold_ge(int(np.ceil(current_max + 1)))),
                "p_upside_ge_2c": float(dist.threshold_ge(int(np.ceil(current_max + 2)))),
                "p_upside_ge_3c": float(dist.threshold_ge(int(np.ceil(current_max + 3)))),
            }
        )
    return pd.DataFrame(rows)


def _summary(scored: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(scored, ["variant"])


def _by_group(scored: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return _aggregate(scored, ["variant", group_col])


def _aggregate(scored: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        scored.groupby(group_cols, dropna=False)
        .agg(
            rows=("target_date_local", "count"),
            distinct_days=("target_date_local", "nunique"),
            mae_expected_c=("abs_error_c", "mean"),
            mae_mode_c=("mode_abs_error_c", "mean"),
            nll_actual_bin=("nll_actual_bin", "mean"),
            mean_p_actual_bin=("p_actual_bin", "mean"),
            brier_upside_ge_1c=("brier_upside_ge_1c", "mean"),
            brier_upside_ge_2c=("brier_upside_ge_2c", "mean"),
            brier_upside_ge_3c=("brier_upside_ge_3c", "mean"),
            avg_p_upside_ge_1c=("p_upside_ge_1c", "mean"),
            actual_upside_ge_1c=("remaining_upside_c", lambda s: float((s >= 1.0).mean())),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def _context_counts(frame: pd.DataFrame) -> list[dict]:
    return json.loads(
        frame.groupby(["after_dip_risk_context", "risk_context"], dropna=False)
        .agg(rows=("target_date_local", "count"), days=("target_date_local", "nunique"))
        .reset_index()
        .to_json(orient="records")
    )


def _recommendation(summary: pd.DataFrame, test_rows: int) -> str:
    if test_rows < 100:
        return "do_not_promote_small_holdout_collect_more_cases"
    pivot = summary.set_index("variant")
    if {"after_dip_ml_candidate", "current_production_with_survival"}.issubset(pivot.index):
        cand = pivot.loc["after_dip_ml_candidate"]
        prod = pivot.loc["current_production_with_survival"]
        if cand["brier_upside_ge_1c"] < prod["brier_upside_ge_1c"] and cand["nll_actual_bin"] <= prod["nll_actual_bin"]:
            return "promising_shadow_candidate_not_production_until_more_live_cases"
    return "do_not_promote_candidate_keep_rule_guard"


def _numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in frame.columns:
            if frame[column].dtype == bool:
                out[column] = frame[column].astype(float)
            else:
                out[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            out[column] = np.nan
    return out


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    return frame[column].fillna(False).astype(bool)


def _survival_to_probabilities(survival: dict[int, float], max_upside_c: int) -> np.ndarray:
    values = np.minimum.accumulate(
        np.clip([survival.get(threshold, 0.0) for threshold in range(1, max_upside_c + 1)], 0.0, 1.0)
    )
    probs = np.empty(max_upside_c + 1, dtype=float)
    probs[0] = 1.0 - values[0]
    probs[1:-1] = values[:-1] - values[1:]
    probs[-1] = values[-1]
    return probs / probs.sum()


def _prob_at(distribution: TmaxDistribution, actual_bin: int) -> float:
    mask = distribution.bins_c == actual_bin
    return float(distribution.probabilities[mask][0]) if mask.any() else 0.0


def _brier(probability: float, event: bool) -> float:
    return float((float(probability) - float(bool(event))) ** 2)


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# LFPB after-dip remaining-upside candidate",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        "## Dataset",
        "",
        f"- Total rows: `{report['rows_total']}`",
        f"- Total days: `{report['days_total']}`",
        f"- Risk rows: `{report['risk_rows_total']}`",
        f"- Risk days: `{report['risk_days_total']}`",
        f"- Train/test rows: `{report['train_rows']}` / `{report['test_rows']}`",
        f"- Train/test days: `{report['train_days']}` / `{report['test_days']}`",
        "",
        "## Summary",
        "",
        _markdown_table(pd.DataFrame(report["summary"])),
        "",
        "## By Hour",
        "",
        _markdown_table(pd.DataFrame(report["by_hour"])),
        "",
        "## Recommendation",
        "",
        f"`{report['recommendation']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
