from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.awc import AWCAdapter
from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.data.open_meteo import fetch_open_meteo_live_extract
from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.spatial_metar import SPATIAL_STATIONS_BY_AIRPORT
from weather_tmax_bot.temporal.freshness_gate import evaluate_freshness_gate


def refresh_operational_data(
    airport: str = "EDDM",
    target_date_local: date | None = None,
    refresh_awc: bool = True,
    refresh_nwp: bool = True,
    root: str | Path = ".",
) -> dict:
    if airport != "EDDM":
        raise ValueError("MVP operational refresh currently supports EDDM only")
    root = Path(root)
    target = target_date_local or date.today()
    summary = {"airport": airport, "target_date_local": target.isoformat(), "sources": {}}
    if refresh_awc:
        awc = refresh_awc_live(airport=airport, root=root)
        summary["sources"]["awc"] = awc
        spatial_stations = SPATIAL_STATIONS_BY_AIRPORT.get(airport.upper(), ())
        if spatial_stations:
            summary["sources"]["spatial_awc"] = refresh_spatial_awc_live(airport=airport, root=root)
    if refresh_nwp:
        nwp = refresh_open_meteo_nwp(airport=airport, target_date_local=target, root=root)
        summary["sources"]["open_meteo_nwp"] = nwp
    summary["freshness_gate"] = evaluate_freshness_gate(root=root, fail_on_missing=False, fail_on_stale=True)
    return summary


def refresh_awc_live(airport: str = "EDDM", root: str | Path = ".") -> dict:
    root = Path(root)
    adapter = AWCAdapter()
    metar = adapter.fetch_latest_metar(airport, hours=30)
    taf = adapter.fetch_latest_taf(airport)
    metar_path = root / f"data/forecasts/awc_metar_live_{airport}.parquet"
    taf_path = root / f"data/forecasts/awc_taf_live_{airport}.parquet"
    _append_dedup(metar, metar_path)
    _append_dedup(taf, taf_path)
    return {
        "metar_rows_fetched": len(metar),
        "taf_rows_fetched": len(taf),
        "metar_archive_rows": _rows(metar_path),
        "taf_archive_rows": _rows(taf_path),
    }


def refresh_spatial_awc_live(airport: str = "EDDM", root: str | Path = ".") -> dict:
    root = Path(root)
    stations = SPATIAL_STATIONS_BY_AIRPORT.get(airport.upper(), ())
    refreshed = {}
    for station in stations:
        try:
            refreshed[station] = refresh_awc_live(airport=station, root=root)
        except Exception as exc:
            refreshed[station] = {"error": str(exc)}
    return refreshed


def refresh_open_meteo_nwp(airport: str, target_date_local: date, root: str | Path = ".") -> dict:
    if airport != "EDDM":
        raise ValueError("MVP Open-Meteo refresh currently knows EDDM coordinates only")
    root = Path(root)
    rows = fetch_open_meteo_live_extract(
        airport_icao=airport,
        latitude=48.3538,
        longitude=11.7861,
        target_date_local=target_date_local,
        timezone_name="Europe/Berlin",
    )
    if rows.empty:
        return {"rows_fetched": 0, "archive_rows": _rows(root / "data/forecasts/open_meteo_archive.parquet")}
    archive_path = root / "data/forecasts/open_meteo_archive.parquet"
    NWPArchive(archive_path).append_extract(rows)
    return {"rows_fetched": len(rows), "archive_rows": _rows(archive_path)}


def _append_dedup(rows: pd.DataFrame, path: Path) -> None:
    if path.exists() and not rows.empty:
        rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
        if "ingest_time_utc" in rows.columns:
            rows = rows.sort_values("ingest_time_utc")
        subset = [col for col in ("raw_record_hash",) if col in rows.columns]
        if subset:
            # Preserve the earliest operational retrieval: it is the true
            # knowledge time for an unchanged live report.
            rows = rows.drop_duplicates(subset=subset, keep="first")
    rows = _normalize_refresh_frame(rows)
    write_parquet(rows, path)


def _rows(path: Path) -> int:
    return 0 if not path.exists() else len(pd.read_parquet(path))


def _normalize_refresh_frame(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for col in ("visibility", "weather_codes", "cloud_layers", "ceiling_ft"):
        if col in rows.columns:
            rows[col] = rows[col].astype("string")
    for col in ("temperature_c", "dewpoint_c", "qnh_hpa", "wind_direction_deg", "wind_speed_kt", "gust_kt"):
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows
