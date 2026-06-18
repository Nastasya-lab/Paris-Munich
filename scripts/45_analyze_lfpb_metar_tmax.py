from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import typer

from weather_tmax_bot.data.metar import metar_records_from_raw
from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.build_metar_target import build_daily_metar_tmax

app = typer.Typer()

AIRPORT = "LFPB"
TIMEZONE = "Europe/Paris"
STATION_ID = "95088001"
IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
MF_DAILY_URLS = [
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_95_previous-1950-2024_RR-T-Vent.csv.gz",
    "https://object.files.data.gouv.fr/meteofrance/data/synchro_ftp/BASE/QUOT/Q_95_latest-2025-2026_RR-T-Vent.csv.gz",
]


@app.command()
def main(
    start_date: str = typer.Option("2020-01-01"),
    end_date: str = typer.Option("2026-06-06"),
    output_dir: str = typer.Option("data/reports"),
    raw_metar_path: str = typer.Option("data/interim/metar_iem_LFPB.parquet"),
    target_path: str = typer.Option("data/processed/metar_tmax_target_LFPB.parquet"),
    fetch_metar: bool = typer.Option(True),
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    metar = _fetch_or_read_metar(start_date, end_date, raw_metar_path, fetch=fetch_metar)
    target = build_daily_metar_tmax(
        metar,
        airport_icao=AIRPORT,
        timezone_name=TIMEZONE,
        source_id="iem.metar.archive.LFPB",
        expected_reports_per_day=48,
    )
    write_parquet(target, target_path)

    official = _read_official_daily_tx()
    joined = target.merge(official, on="target_date_local", how="inner")
    joined["official_minus_metar_c"] = joined["official_tx_c"] - joined["metar_tmax_c"]
    joined["metar_tmax_int"] = joined["metar_tmax_c"].round().astype("Int64")
    joined["official_tx_int"] = joined["official_tx_c"].round().astype("Int64")
    joined.to_csv(output / "lfpb_metar_vs_official_tx.csv", index=False)

    baseline = _climatology_leave_one_year_out(joined)
    baseline.to_csv(output / "lfpb_metar_tmax_climatology_baseline.csv", index=False)

    report = _build_report(metar, target, joined, baseline, start_date, end_date)
    (output / "lfpb_metar_tmax_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, output / "lfpb_metar_tmax_analysis.md")
    print(json.dumps(report, indent=2, default=str))


def _fetch_or_read_metar(start_date: str, end_date: str, path: str | Path, *, fetch: bool) -> pd.DataFrame:
    p = Path(path)
    if not fetch and p.exists():
        return pd.read_parquet(p)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    params = [
        ("station", AIRPORT),
        ("data", "metar"),
        ("year1", str(start.year)),
        ("month1", str(start.month)),
        ("day1", str(start.day)),
        ("year2", str(end.year)),
        ("month2", str(end.month)),
        ("day2", str(end.day)),
        ("tz", "Etc/UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("elev", "yes"),
        ("missing", "M"),
        ("trace", "T"),
        ("direct", "yes"),
        ("report_type", "1"),
        ("report_type", "2"),
    ]
    response = requests.get(IEM_URL, params=params, timeout=120)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), comment="#")
    if frame.empty:
        raise RuntimeError("IEM returned no LFPB METAR rows")
    raw_col = "metar" if "metar" in frame.columns else "raw"
    rows = []
    for _, row in frame.dropna(subset=[raw_col, "valid"]).iterrows():
        rows.append((str(row[raw_col]), pd.Timestamp(row["valid"], tz="UTC").to_pydatetime()))
    metar = metar_records_from_raw(rows, source_id="iem.metar.archive.LFPB")
    write_parquet(metar, p)
    return metar


def _read_official_daily_tx() -> pd.DataFrame:
    frames = []
    for url in MF_DAILY_URLS:
        for chunk in pd.read_csv(
            url,
            compression="gzip",
            sep=";",
            dtype=str,
            usecols=["NUM_POSTE", "AAAAMMJJ", "TX", "QTX", "HTX"],
            chunksize=300000,
        ):
            sub = chunk[chunk["NUM_POSTE"].eq(STATION_ID)].copy()
            if not sub.empty:
                frames.append(sub)
    if not frames:
        raise RuntimeError("No Meteo-France daily TX rows found for LFPB")
    df = pd.concat(frames, ignore_index=True)
    df["target_date_local"] = pd.to_datetime(df["AAAAMMJJ"], format="%Y%m%d").dt.date.astype(str)
    df["official_tx_c"] = pd.to_numeric(df["TX"], errors="coerce")
    df["official_qtx"] = pd.to_numeric(df["QTX"], errors="coerce")
    df["official_htx"] = df["HTX"]
    return df[["target_date_local", "official_tx_c", "official_qtx", "official_htx"]]


def _climatology_leave_one_year_out(target: pd.DataFrame, window_days: int = 15) -> pd.DataFrame:
    df = target.copy()
    df["date"] = pd.to_datetime(df["target_date_local"])
    df["year"] = df["date"].dt.year
    df["doy"] = df["date"].dt.dayofyear
    rows = []
    for _, row in df.iterrows():
        distance = np.minimum((df["doy"] - row["doy"]).abs(), 366 - (df["doy"] - row["doy"]).abs())
        train = df[(df["year"] != row["year"]) & (distance <= window_days)]["metar_tmax_c"].dropna()
        if train.empty:
            prediction = float(df[df["year"] != row["year"]]["metar_tmax_c"].median())
        else:
            prediction = float(train.median())
        rows.append(
            {
                "target_date_local": row["target_date_local"],
                "actual_metar_tmax_c": float(row["metar_tmax_c"]),
                "climatology_median_c": prediction,
                "error_c": prediction - float(row["metar_tmax_c"]),
                "abs_error_c": abs(prediction - float(row["metar_tmax_c"])),
            }
        )
    return pd.DataFrame(rows)


def _build_report(metar: pd.DataFrame, target: pd.DataFrame, joined: pd.DataFrame, baseline: pd.DataFrame, start: str, end: str) -> dict:
    good = target[target["quality_flags"].eq("ok")]
    diff = joined["official_minus_metar_c"].dropna()
    return {
        "airport": AIRPORT,
        "target_type": "metar_tmax",
        "period_requested": [start, end],
        "metar_rows": int(len(metar)),
        "metar_temperature_nonnull_rate": float(pd.to_numeric(metar["temperature_c"], errors="coerce").notna().mean()),
        "target_days": int(len(target)),
        "target_ok_days": int(len(good)),
        "target_low_coverage_days": int((target["quality_flags"] != "ok").sum()),
        "median_metar_reports_per_day": float(target["metar_obs_count"].median()),
        "paired_official_days": int(len(joined)),
        "official_minus_metar_mean_c": float(diff.mean()),
        "official_minus_metar_median_c": float(diff.median()),
        "official_minus_metar_p95_abs_c": float(diff.abs().quantile(0.95)),
        "official_equals_metar_integer_rate": float((joined["official_tx_int"] == joined["metar_tmax_int"]).mean()),
        "official_within_1c_of_metar_rate": float((joined["official_tx_c"] - joined["metar_tmax_c"]).abs().le(1.0).mean()),
        "climatology_baseline_mae_c": float(baseline["abs_error_c"].mean()),
        "climatology_baseline_exact_integer_rate": float(
            (baseline["climatology_median_c"].round().astype(int) == baseline["actual_metar_tmax_c"].round().astype(int)).mean()
        ),
        "recommendation": _recommendation(target, joined, baseline),
    }


def _recommendation(target: pd.DataFrame, joined: pd.DataFrame, baseline: pd.DataFrame) -> str:
    if len(target) < 700:
        return "collect_more_metar_history_before_training"
    if target["quality_flags"].eq("ok").mean() < 0.8:
        return "metar_coverage_too_sparse_review_target_quality"
    if float(baseline["abs_error_c"].mean()) > 4.0:
        return "climatology_baseline_weak_but_training_still_possible"
    return "proceed_to_remaining_upside_dataset"


def _write_markdown(report: dict, path: Path) -> None:
    lines = [
        "# LFPB METAR Tmax analysis",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary",
        "",
        f"- METAR rows: `{report['metar_rows']}`",
        f"- Target days: `{report['target_days']}`; ok days: `{report['target_ok_days']}`",
        f"- Median METAR reports/day: `{report['median_metar_reports_per_day']:.1f}`",
        f"- Paired official TX days: `{report['paired_official_days']}`",
        f"- Official TX minus METAR Tmax: mean `{report['official_minus_metar_mean_c']:.2f} C`, median `{report['official_minus_metar_median_c']:.2f} C`, 95% abs `{report['official_minus_metar_p95_abs_c']:.2f} C`",
        f"- Official rounded TX equals rounded METAR Tmax: `{report['official_equals_metar_integer_rate']:.1%}`",
        f"- Official TX within 1 C of METAR Tmax: `{report['official_within_1c_of_metar_rate']:.1%}`",
        f"- Leave-one-year-out climatology MAE: `{report['climatology_baseline_mae_c']:.2f} C`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
        "## Interpretation",
        "",
        "This target predicts the maximum temperature that appears in LFPB METAR/SPECI reports during the local day.",
        "It is intentionally different from the official Meteo-France daily TX target.",
        "The next step is a remaining-upside dataset where each issue row predicts how much higher the final METAR Tmax can go above the observed METAR max so far.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    app()
