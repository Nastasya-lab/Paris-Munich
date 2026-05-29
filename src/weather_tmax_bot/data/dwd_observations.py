from __future__ import annotations

from datetime import date, timedelta
import re
from pathlib import Path
from urllib.parse import urljoin
from zipfile import ZipFile

import pandas as pd
import requests

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.utils.validation import DataAvailabilityError


DWD_10MIN_HIST_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/10_minutes/air_temperature/historical/"
)
DWD_10MIN_RECENT_BASE = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/10_minutes/air_temperature/recent/"
)


class DWDAdapter:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)

    def fetch_observations(self, airport: str, start: date, end: date, station_id: str = "01262") -> pd.DataFrame:
        archives = self.discover_station_archives(station_id=station_id, start=start, end=end)
        if not archives:
            raise DataAvailabilityError(f"no DWD 10-minute archives discovered for station {station_id}")
        frames = []
        for url in archives:
            filename = url.rsplit("/", 1)[-1]
            path = self.data_dir / "raw" / "dwd" / filename
            if not path.exists():
                self.download_file(url, path)
            frames.append(self.parse_10min_air_temperature_zip(path, station_id=station_id))
        obs = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["station_id", "observation_time_utc"])
        mask = (obs["observation_time_utc"].dt.date >= start) & (obs["observation_time_utc"].dt.date <= end)
        out = obs.loc[mask].sort_values("observation_time_utc").reset_index(drop=True)
        write_parquet(out, self.data_dir / "interim" / f"dwd_10min_temperature_{station_id}_{start}_{end}.parquet")
        write_parquet(out, self.data_dir / "interim" / f"dwd_10min_temperature_{station_id}.parquet")
        return out

    def discover_station_archives(self, station_id: str, start: date, end: date) -> list[str]:
        urls = []
        for base in (DWD_10MIN_HIST_BASE, DWD_10MIN_RECENT_BASE):
            try:
                listing = requests.get(base, timeout=30).text
            except requests.RequestException:
                continue
            for href in re.findall(r'href="([^"]+\.zip)"', listing):
                if f"_{station_id}_" not in href:
                    continue
                period = re.search(r"_(\d{8})_(\d{8})_", href)
                if period:
                    file_start = pd.to_datetime(period.group(1), format="%Y%m%d").date()
                    file_end = pd.to_datetime(period.group(2), format="%Y%m%d").date()
                    if file_end < start or file_start > end:
                        continue
                elif base == DWD_10MIN_RECENT_BASE and end < date.today() - timedelta(days=550):
                    continue
                urls.append(urljoin(base, href))
        return sorted(set(urls))

    def parse_10min_air_temperature_zip(
        self,
        zip_path: str | Path,
        station_id: str = "01262",
        source_id: str = "dwd.10min.air_temperature.01262",
    ) -> pd.DataFrame:
        zip_path = Path(zip_path)
        with ZipFile(zip_path) as zf:
            product_names = [n for n in zf.namelist() if n.lower().startswith("produkt") and n.lower().endswith(".txt")]
            if not product_names:
                raise ValueError(f"no DWD product txt found in {zip_path}")
            with zf.open(product_names[0]) as fh:
                df = pd.read_csv(fh, sep=";", na_values=[-999, "-999"])
        time_col = "MESS_DATUM"
        temp_col = "TT_10"
        if time_col not in df.columns or temp_col not in df.columns:
            raise ValueError(f"expected {time_col}/{temp_col} in DWD file, got {list(df.columns)}")
        out = pd.DataFrame(
            {
                "station_id": df["STATIONS_ID"].astype(str).str.zfill(5),
                "observation_time_utc": pd.to_datetime(df[time_col].astype(str), format="%Y%m%d%H%M", utc=True),
                "temperature_c": pd.to_numeric(df[temp_col], errors="coerce"),
                "quality_flag": df.get("QN", "unknown"),
                "source_id": source_id,
                "source_version": "DWD_CDC_10min_air_temperature",
            }
        )
        out = out[out["station_id"] == station_id].dropna(subset=["temperature_c"]).reset_index(drop=True)
        write_parquet(out, self.data_dir / "interim" / f"dwd_10min_temperature_{station_id}.parquet")
        return out

    def download_file(self, url: str, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)
        return path
