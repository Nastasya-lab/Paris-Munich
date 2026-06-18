from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.metar_upside_dataset import build_metar_remaining_upside_dataset
from weather_tmax_bot.utils.time import local_day_bounds_utc

app = typer.Typer()

MIN_URLS = [
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MIN/MN_95_previous-2020-2024.csv.gz",
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/MIN/MN_95_latest-2025-2026.csv.gz",
]
STATION_ID = "95088001"


@app.command()
def main(
    metar_path: str = typer.Option("data/interim/metar_iem_LFPB.parquet"),
    target_path: str = typer.Option("data/processed/metar_tmax_target_LFPB.parquet"),
    output_path: str = typer.Option("data/processed/metar_upside_dataset_LFPB.parquet"),
    report_dir: str = typer.Option("data/reports"),
    include_rain: bool = typer.Option(True),
) -> None:
    metar = pd.read_parquet(metar_path)
    target = pd.read_parquet(target_path)
    rain = _read_min6_rain() if include_rain else pd.DataFrame()
    dataset = build_metar_remaining_upside_dataset(
        metar,
        target,
        airport_icao="LFPB",
        timezone_name="Europe/Paris",
    )
    if include_rain:
        dataset = _add_fast_rain_features(dataset, rain)
    write_parquet(dataset, output_path)

    report_output = Path(report_dir)
    report_output.mkdir(parents=True, exist_ok=True)
    hour_summary = _hourly_summary(dataset)
    hour_summary.to_csv(report_output / "lfpb_metar_upside_by_hour.csv", index=False)
    baseline = _phase_prior_leave_one_year_out(dataset)
    baseline.to_csv(report_output / "lfpb_metar_upside_phase_prior_baseline.csv", index=False)
    report = _report(dataset, hour_summary, baseline)
    (report_output / "lfpb_metar_upside_dataset_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    _write_markdown(report, report_output / "lfpb_metar_upside_dataset_report.md")
    print(json.dumps(report, indent=2, default=str))


def _read_min6_rain() -> pd.DataFrame:
    frames = []
    for url in MIN_URLS:
        for chunk in pd.read_csv(
            url,
            sep=";",
            compression="gzip",
            dtype=str,
            usecols=["NUM_POSTE", "AAAAMMJJHHMN", "RR", "QRR"],
            chunksize=500000,
        ):
            sub = chunk[chunk["NUM_POSTE"].eq(STATION_ID)].copy()
            if not sub.empty:
                frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=["observation_time_utc", "rr_mm"])
    df = pd.concat(frames, ignore_index=True)
    df["observation_time_utc"] = pd.to_datetime(df["AAAAMMJJHHMN"], format="%Y%m%d%H%M", utc=True, errors="coerce")
    df["rr_mm"] = pd.to_numeric(df["RR"], errors="coerce").fillna(0.0)
    return df[["observation_time_utc", "rr_mm"]].dropna(subset=["observation_time_utc"])


def _add_fast_rain_features(dataset: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    out = dataset.copy()
    if rain.empty:
        out["rain_6min_missing"] = True
        for column in ["rain_mm_last_30m", "rain_mm_last_1h", "rain_mm_last_3h", "rain_mm_since_midnight", "rain_max_6min_last_3h"]:
            out[column] = 0.0
        return out
    rain_df = rain.sort_values("observation_time_utc").copy()
    rain_df["observation_time_utc"] = pd.to_datetime(rain_df["observation_time_utc"], utc=True)
    times = rain_df["observation_time_utc"].astype("int64").to_numpy()
    rr = pd.to_numeric(rain_df["rr_mm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cumsum = np.concatenate([[0.0], np.cumsum(rr)])

    def sum_between(start_ns: int, end_ns: int) -> float:
        left = int(np.searchsorted(times, start_ns, side="left"))
        right = int(np.searchsorted(times, end_ns, side="right"))
        return float(cumsum[right] - cumsum[left])

    issue_times = pd.to_datetime(out["issue_time_utc"], utc=True)
    rows = []
    for issue, target_date in zip(issue_times, out["target_date_local"], strict=False):
        issue_ns = int(issue.value)
        day_start, _ = local_day_bounds_utc(pd.Timestamp(str(target_date)).date(), "Europe/Paris")
        day_start_ns = int(pd.Timestamp(day_start).value)
        last_3h_start_ns = int((issue - pd.Timedelta(hours=3)).value)
        left = int(np.searchsorted(times, last_3h_start_ns, side="left"))
        right = int(np.searchsorted(times, issue_ns, side="right"))
        max_3h = float(np.nanmax(rr[left:right])) if right > left else 0.0
        rows.append(
            {
                "rain_6min_missing": False,
                "rain_mm_last_30m": sum_between(int((issue - pd.Timedelta(minutes=30)).value), issue_ns),
                "rain_mm_last_1h": sum_between(int((issue - pd.Timedelta(hours=1)).value), issue_ns),
                "rain_mm_last_3h": sum_between(last_3h_start_ns, issue_ns),
                "rain_mm_since_midnight": sum_between(day_start_ns, issue_ns),
                "rain_max_6min_last_3h": max_3h,
            }
        )
    rain_features = pd.DataFrame(rows, index=out.index)
    for column in rain_features.columns:
        out[column] = rain_features[column]
    return out


def _hourly_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    grouped = (
        df.groupby("local_issue_hour", dropna=False)
        .agg(
            rows=("target_date_local", "count"),
            mean_remaining_upside_c=("remaining_upside_c", "mean"),
            median_remaining_upside_c=("remaining_upside_c", "median"),
            p_upside_ge_1c=("upside_ge_1c", "mean"),
            p_upside_ge_2c=("upside_ge_2c", "mean"),
            p_upside_ge_3c=("upside_ge_3c", "mean"),
            persistence_mae_c=("remaining_upside_c", "mean"),
        )
        .reset_index()
    )
    return grouped


def _phase_prior_leave_one_year_out(dataset: pd.DataFrame) -> pd.DataFrame:
    df = dataset.copy()
    df["date"] = pd.to_datetime(df["target_date_local"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].map(_season)
    grouped = {
        key: group.copy()
        for key, group in df.groupby(["local_issue_hour", "season"], sort=False)
    }
    grouped_hour = {key: group.copy() for key, group in df.groupby("local_issue_hour", sort=False)}
    median_cache: dict[tuple[float, str, int], float] = {}
    fallback_cache: dict[tuple[float, int], float] = {}
    rows = []
    for _, row in df.iterrows():
        key = (float(row["local_issue_hour"]), str(row["season"]), int(row["year"]))
        if key not in median_cache:
            group = grouped.get((row["local_issue_hour"], row["season"]), pd.DataFrame())
            train = group[group["year"] != row["year"]] if not group.empty else group
            median_cache[key] = float(train["remaining_upside_c"].median()) if not train.empty else float("nan")
        pred_upside = median_cache[key]
        if np.isnan(pred_upside):
            fallback_key = (float(row["local_issue_hour"]), int(row["year"]))
            if fallback_key not in fallback_cache:
                group = grouped_hour.get(row["local_issue_hour"], pd.DataFrame())
                train = group[group["year"] != row["year"]] if not group.empty else group
                fallback_cache[fallback_key] = float(train["remaining_upside_c"].median()) if not train.empty else 0.0
            pred_upside = fallback_cache[fallback_key]
        pred_final = float(row["current_metar_max_c"] + pred_upside)
        actual = float(row["final_metar_tmax_c"])
        rows.append(
            {
                "target_date_local": row["target_date_local"],
                "local_issue_hour": row["local_issue_hour"],
                "season": row["season"],
                "actual_final_metar_tmax_c": actual,
                "current_metar_max_c": float(row["current_metar_max_c"]),
                "persistence_error_c": float(row["current_metar_max_c"] - actual),
                "phase_prior_predicted_upside_c": pred_upside,
                "phase_prior_predicted_final_c": pred_final,
                "phase_prior_error_c": pred_final - actual,
            }
        )
    out = pd.DataFrame(rows)
    out["persistence_abs_error_c"] = out["persistence_error_c"].abs()
    out["phase_prior_abs_error_c"] = out["phase_prior_error_c"].abs()
    return out


def _report(dataset: pd.DataFrame, hour_summary: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    by_hour = (
        baseline.groupby("local_issue_hour")
        .agg(
            rows=("target_date_local", "count"),
            persistence_mae_c=("persistence_abs_error_c", "mean"),
            phase_prior_mae_c=("phase_prior_abs_error_c", "mean"),
            phase_prior_win_rate=("phase_prior_abs_error_c", lambda s: np.nan),
        )
        .reset_index()
    )
    wins = []
    for hour, group in baseline.groupby("local_issue_hour"):
        wins.append(
            {
                "local_issue_hour": float(hour),
                "phase_prior_win_rate": float((group["phase_prior_abs_error_c"] < group["persistence_abs_error_c"]).mean()),
            }
        )
    win_frame = pd.DataFrame(wins)
    by_hour = by_hour.drop(columns=["phase_prior_win_rate"]).merge(win_frame, on="local_issue_hour", how="left")
    return {
        "airport": "LFPB",
        "dataset_rows": int(len(dataset)),
        "target_days": int(dataset["target_date_local"].nunique()) if not dataset.empty else 0,
        "issue_hours": sorted(float(v) for v in dataset["local_issue_hour"].dropna().unique()),
        "leakage_check_pass_rate": float(dataset["leakage_check_passed"].mean()) if not dataset.empty else None,
        "rain_feature_missing_rate": float(dataset["rain_6min_missing"].mean()) if "rain_6min_missing" in dataset else None,
        "overall_persistence_mae_c": float(baseline["persistence_abs_error_c"].mean()) if not baseline.empty else None,
        "overall_phase_prior_mae_c": float(baseline["phase_prior_abs_error_c"].mean()) if not baseline.empty else None,
        "phase_prior_win_rate": float((baseline["phase_prior_abs_error_c"] < baseline["persistence_abs_error_c"]).mean())
        if not baseline.empty
        else None,
        "by_hour": json.loads(by_hour.to_json(orient="records")) if not by_hour.empty else [],
        "upside_by_hour": json.loads(hour_summary.to_json(orient="records")) if not hour_summary.empty else [],
        "recommendation": "train_ordinal_remaining_upside_model" if len(dataset) > 5000 else "collect_more_rows",
    }


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# LFPB METAR remaining-upside dataset",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Rows: `{report['dataset_rows']}`",
        f"- Target days: `{report['target_days']}`",
        f"- Issue hours: `{report['issue_hours']}`",
        f"- Leakage check pass rate: `{report['leakage_check_pass_rate']:.1%}`",
        f"- Rain feature missing rate: `{report['rain_feature_missing_rate']:.1%}`",
        f"- Persistence MAE: `{report['overall_persistence_mae_c']:.2f} C`",
        f"- Phase-prior MAE: `{report['overall_phase_prior_mae_c']:.2f} C`",
        f"- Phase-prior win rate: `{report['phase_prior_win_rate']:.1%}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "## By Local Issue Hour",
        "",
        "| hour | persistence MAE | phase-prior MAE | phase-prior win rate |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["by_hour"]:
        lines.append(
            f"| {row['local_issue_hour']:.0f} | {row['persistence_mae_c']:.2f} | {row['phase_prior_mae_c']:.2f} | {row['phase_prior_win_rate']:.1%} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    app()
