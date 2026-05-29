from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from weather_tmax_bot.utils.geo import haversine_km


DWD_10MIN_STATION_LIST_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/10_minutes/air_temperature/historical/zehn_min_tu_Beschreibung_Stationen.txt"
)


@dataclass(frozen=True)
class StationSelection:
    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    distance_km: float
    source_url: str
    rationale: str
    date_from: date | None = None
    date_to: date | None = None


def known_eddm_station() -> StationSelection:
    return StationSelection(
        station_id="01262",
        name="Muenchen-Flughafen",
        latitude=48.3477,
        longitude=11.8134,
        elevation_m=446.0,
        distance_km=round(haversine_km(48.3538, 11.7861, 48.3477, 11.8134), 2),
        date_from=date(1992, 5, 19),
        date_to=date(2026, 5, 28),
        source_url=DWD_10MIN_STATION_LIST_URL,
        rationale="Closest official DWD station matching Munich Airport and named Muenchen-Flughafen.",
    )


def fetch_station_metadata(cache_path: str | Path = "data/raw/dwd_10min_station_metadata.txt") -> Path:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DWD_10MIN_STATION_LIST_URL, timeout=30)
    response.raise_for_status()
    path.write_text(response.text, encoding="latin-1")
    return path


def parse_station_metadata_text(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 8 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "station_id": parts[0].zfill(5),
                    "date_from": pd.to_datetime(parts[1], format="%Y%m%d").date(),
                    "date_to": pd.to_datetime(parts[2], format="%Y%m%d").date(),
                    "elevation_m": float(parts[3]),
                    "latitude": float(parts[4]),
                    "longitude": float(parts[5]),
                    "name": parts[6],
                    "state": parts[7],
                }
            )
        except ValueError:
            continue
    return pd.DataFrame(rows)


def read_station_metadata(path: str | Path) -> pd.DataFrame:
    raw = Path(path).read_bytes()
    text = raw.decode("latin-1")
    return parse_station_metadata_text(text)


def select_nearest_station(stations: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    df = stations.copy()
    df["distance_km"] = df.apply(lambda r: haversine_km(lat, lon, r["latitude"], r["longitude"]), axis=1)
    return df.sort_values("distance_km").reset_index(drop=True)


def select_station_for_airport(
    stations: pd.DataFrame,
    airport_lat: float,
    airport_lon: float,
    preferred_name_contains: str = "Flughafen",
    max_distance_km: float = 50.0,
) -> StationSelection:
    ranked = select_nearest_station(stations, airport_lat, airport_lon)
    nearby = ranked[ranked["distance_km"] <= max_distance_km].copy()
    if nearby.empty:
        raise ValueError("no DWD station within max distance")
    preferred = nearby[nearby["name"].str.contains(preferred_name_contains, case=False, na=False)]
    chosen = preferred.iloc[0] if not preferred.empty else nearby.iloc[0]
    return StationSelection(
        station_id=chosen["station_id"],
        name=str(chosen["name"]).replace("\xfc", "ue"),
        latitude=float(chosen["latitude"]),
        longitude=float(chosen["longitude"]),
        elevation_m=float(chosen["elevation_m"]),
        distance_km=float(round(chosen["distance_km"], 2)),
        date_from=chosen["date_from"],
        date_to=chosen["date_to"],
        source_url=DWD_10MIN_STATION_LIST_URL,
        rationale="Nearest official DWD 10-minute air-temperature station with airport-name preference.",
    )
