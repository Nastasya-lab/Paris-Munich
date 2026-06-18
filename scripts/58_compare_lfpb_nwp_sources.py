from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


NWP_VALUE_COLUMNS = [
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
]


def main() -> None:
    dataset_path = Path("data/processed/metar_upside_dataset_LFPB.parquet")
    icon_path = Path("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
    ecmwf_path = Path("data/forecasts/open_meteo_single_runs_ecmwf_ifs_LFPB.parquet")
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(dataset_path)
    icon = pd.read_parquet(icon_path)
    ecmwf = pd.read_parquet(ecmwf_path)

    joined = _join_two_sources(dataset, icon, ecmwf)
    if joined.empty:
        raise ValueError("No common leakage-safe ICON/ECMWF rows found for LFPB")

    scored = _score(joined)
    overall = _summary(scored, ["source"])
    by_phase = _summary(scored, ["phase", "source"])
    by_hour = _summary(scored, ["local_issue_hour", "source"])
    by_season = _summary(scored, ["season", "source"])
    winners_by_phase = _winner_summary(scored, ["phase"])
    suggested_weights = _suggested_weights(scored)

    joined.to_parquet(report_dir / "lfpb_icon_ecmwf_common_rows.parquet", index=False)
    scored.to_parquet(report_dir / "lfpb_icon_ecmwf_source_scores.parquet", index=False)
    overall.to_csv(report_dir / "lfpb_icon_ecmwf_overall.csv", index=False)
    by_phase.to_csv(report_dir / "lfpb_icon_ecmwf_by_phase.csv", index=False)
    by_hour.to_csv(report_dir / "lfpb_icon_ecmwf_by_hour.csv", index=False)
    by_season.to_csv(report_dir / "lfpb_icon_ecmwf_by_season.csv", index=False)
    winners_by_phase.to_csv(report_dir / "lfpb_icon_ecmwf_winners_by_phase.csv", index=False)
    suggested_weights.to_csv(report_dir / "lfpb_icon_ecmwf_suggested_weights.csv", index=False)

    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": "LFPB",
        "target": "daily maximum temperature reported by METAR",
        "sources_compared": {
            "icon": "open_meteo.single_run.icon_d2",
            "ecmwf": "open_meteo.single_run.ecmwf_ifs",
        },
        "rows": len(joined),
        "days": int(joined["target_date_local"].nunique()),
        "target_period": [str(joined["target_date_local"].min()), str(joined["target_date_local"].max())],
        "phase_definition_local_hour": {
            "before_work": "00-08",
            "morning": "09-11",
            "midday": "12-14",
            "afternoon": "15-16",
            "evening": "17-23",
        },
        "overall": json.loads(overall.to_json(orient="records")),
        "by_phase": json.loads(by_phase.to_json(orient="records")),
        "winners_by_phase": json.loads(winners_by_phase.to_json(orient="records")),
        "suggested_rule_weights": json.loads(suggested_weights.to_json(orient="records")),
        "limitations": [
            "This diagnostic compares raw NWP daily Tmax guidance, not a fully trained probabilistic model.",
            "Rows are included only when both ICON-D2 and ECMWF IFS are available as-of the issue time.",
            "The comparison uses METAR Tmax target, not official national-climate Tmax.",
            "Suggested weights are a diagnostic baseline; final production weights should be validated out-of-time.",
        ],
    }
    (report_dir / "lfpb_icon_ecmwf_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("docs/lfpb_icon_ecmwf_diagnostic.md").write_text(_markdown(report, overall, by_phase, winners_by_phase, suggested_weights), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _join_two_sources(dataset: pd.DataFrame, icon: pd.DataFrame, ecmwf: pd.DataFrame) -> pd.DataFrame:
    base = dataset.copy()
    base["target_date_local"] = base["target_date_local"].astype(str)
    base["issue_time_utc"] = pd.to_datetime(base["issue_time_utc"], utc=True)
    base["target_date_dt"] = pd.to_datetime(base["target_date_local"], errors="coerce")
    base["season"] = base["target_date_dt"].dt.month.map(_season_from_month)

    icon_joined = _join_latest_nwp(base, icon, "icon")
    both = _join_latest_nwp(icon_joined, ecmwf, "ecmwf")
    both = both[both["icon_model_tmax_c"].notna() & both["ecmwf_model_tmax_c"].notna()].copy()
    both["phase"] = both["local_issue_hour"].astype(int).map(_phase)
    both["icon_minus_ecmwf_tmax_c"] = both["icon_model_tmax_c"] - both["ecmwf_model_tmax_c"]
    both["nwp_spread_abs_c"] = both["icon_minus_ecmwf_tmax_c"].abs()
    return both.reset_index(drop=True)


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
        for column in NWP_VALUE_COLUMNS:
            if column in latest:
                merged[f"{prefix}_{column}"] = latest[column]
        merged[f"{prefix}_knowledge_time_utc"] = latest["knowledge_time_utc"].isoformat()
        merged[f"{prefix}_model_run_time_utc"] = latest["model_run_time_utc"].isoformat()
        merged[f"{prefix}_source_id"] = latest.get("source_id")
        rows.append(merged)
    return pd.DataFrame(rows)


def _score(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        actual = float(row["final_metar_tmax_c"])
        for source in ["icon", "ecmwf"]:
            prediction = float(row[f"{source}_model_tmax_c"])
            error = prediction - actual
            rounded_error = int(round(prediction)) - int(round(actual))
            rows.append(
                {
                    "target_date_local": row["target_date_local"],
                    "issue_time_utc": row["issue_time_utc"].isoformat(),
                    "local_issue_hour": int(row["local_issue_hour"]),
                    "phase": row["phase"],
                    "season": row["season"],
                    "source": source,
                    "actual_metar_tmax_c": actual,
                    "predicted_tmax_c": prediction,
                    "error_c": error,
                    "abs_error_c": abs(error),
                    "rounded_error_c": rounded_error,
                    "exact_integer_hit": abs(rounded_error) == 0,
                    "within_1c_integer": abs(rounded_error) <= 1,
                    "nwp_spread_abs_c": float(row["nwp_spread_abs_c"]),
                    "current_metar_max_c": float(row["current_metar_max_c"]),
                    "source_knowledge_time_utc": row[f"{source}_knowledge_time_utc"],
                    "source_model_run_time_utc": row[f"{source}_model_run_time_utc"],
                }
            )
    scored = pd.DataFrame(rows)
    pair_key = ["target_date_local", "issue_time_utc"]
    pivot = scored.pivot_table(index=pair_key, columns="source", values="abs_error_c", aggfunc="first").reset_index()
    pivot["winner"] = np.where(
        pivot["icon"] < pivot["ecmwf"],
        "icon",
        np.where(pivot["ecmwf"] < pivot["icon"], "ecmwf", "tie"),
    )
    return scored.merge(pivot[pair_key + ["winner"]], on=pair_key, how="left")


def _summary(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in scored.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append(
            {
                **dict(zip(columns, keys)),
                "rows": len(group),
                "days": int(group["target_date_local"].nunique()),
                "mae": float(group["abs_error_c"].mean()),
                "rmse": float(np.sqrt(np.mean(np.square(group["error_c"])))),
                "bias": float(group["error_c"].mean()),
                "median_abs_error": float(group["abs_error_c"].median()),
                "exact_integer_hit_rate": float(group["exact_integer_hit"].mean()),
                "within_1c_integer_rate": float(group["within_1c_integer"].mean()),
                "mean_abs_icon_ecmwf_spread": float(group["nwp_spread_abs_c"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _winner_summary(scored: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    pairs = scored.drop_duplicates(["target_date_local", "issue_time_utc"]).copy()
    rows = []
    for keys, group in pairs.groupby(columns, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        total = len(group)
        rows.append(
            {
                **dict(zip(columns, keys)),
                "rows": total,
                "days": int(group["target_date_local"].nunique()),
                "icon_wins": int((group["winner"] == "icon").sum()),
                "ecmwf_wins": int((group["winner"] == "ecmwf").sum()),
                "ties": int((group["winner"] == "tie").sum()),
                "icon_win_rate": float((group["winner"] == "icon").mean()),
                "ecmwf_win_rate": float((group["winner"] == "ecmwf").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(columns).reset_index(drop=True)


def _suggested_weights(scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = scored.pivot_table(
        index=["target_date_local", "issue_time_utc", "phase"],
        columns="source",
        values=["predicted_tmax_c", "actual_metar_tmax_c"],
        aggfunc="first",
    ).reset_index()
    pairs.columns = ["_".join([str(part) for part in col if part]) for col in pairs.columns]
    for phase, group in pairs.groupby("phase"):
        best = None
        for weight_icon in np.linspace(0, 1, 21):
            pred = weight_icon * group["predicted_tmax_c_icon"] + (1 - weight_icon) * group["predicted_tmax_c_ecmwf"]
            actual = group["actual_metar_tmax_c_icon"]
            mae = float(np.mean(np.abs(pred - actual)))
            if best is None or mae < best["mae"]:
                best = {"phase": phase, "weight_icon": float(weight_icon), "weight_ecmwf": float(1 - weight_icon), "mae": mae}
        if best:
            rows.append({**best, "rows": len(group), "days": int(group["target_date_local"].nunique())})
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


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


def _season_from_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def _markdown(report: dict, overall: pd.DataFrame, by_phase: pd.DataFrame, winners: pd.DataFrame, weights: pd.DataFrame) -> str:
    return "\n".join(
        [
            "# LFPB ICON-D2 vs ECMWF IFS diagnostic",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- rows: `{report['rows']}`",
            f"- days: `{report['days']}`",
            f"- period: `{report['target_period'][0]}` to `{report['target_period'][1]}`",
            "",
            "## Overall",
            "",
            _table(overall),
            "",
            "## By Phase",
            "",
            _table(by_phase),
            "",
            "## Winners By Phase",
            "",
            _table(winners),
            "",
            "## Simple Weight Baseline",
            "",
            _table(weights),
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
