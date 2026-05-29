import argparse
from pathlib import Path
import yaml

from weather_tmax_bot.data.dwd_station_metadata import (
    fetch_station_metadata,
    known_eddm_station,
    read_station_metadata,
    select_nearest_station,
    select_station_for_airport,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.airport != "EDDM":
        raise SystemExit("MVP currently supports EDDM")
    try:
        metadata_path = Path("data/raw/dwd_10min_station_metadata.txt")
        if not args.no_download:
            metadata_path = fetch_station_metadata(metadata_path)
        stations = read_station_metadata(metadata_path)
        s = select_station_for_airport(stations, 48.3538, 11.7861)
        alternatives = select_nearest_station(stations, 48.3538, 11.7861).head(5)
    except Exception as exc:
        s = known_eddm_station()
        alternatives = None
        print(f"Falling back to pinned station metadata: {exc}")

    date_range = ""
    if s.date_from and s.date_to:
        date_range = f"\nData period in current station metadata: `{s.date_from}` to `{s.date_to}`.\n"
    Path("docs/station_selection.md").write_text(
        f"# Station selection for EDDM\n\nSelected DWD station `{s.station_id}` `{s.name}` "
        f"at {s.latitude}, {s.longitude}, elevation {s.elevation_m} m. "
        f"Distance to EDDM reference coordinates is about {s.distance_km} km.\n"
        f"{date_range}\n"
        f"Rationale: {s.rationale}\n\nSource metadata URL: {s.source_url}\n",
        encoding="utf-8",
    )
    if alternatives is not None:
        with Path("docs/station_selection.md").open("a", encoding="utf-8") as fh:
            fh.write("\n## Nearby alternatives\n\n")
            for _, row in alternatives.iterrows():
                fh.write(
                    f"- `{row['station_id']}` `{str(row['name']).replace(chr(252), 'ue')}`: "
                    f"{row['distance_km']:.2f} km, {row['date_from']} to {row['date_to']}\n"
                )

    airports_path = Path("config/airports.yaml")
    raw = yaml.safe_load(airports_path.read_text(encoding="utf-8"))
    raw["airports"]["EDDM"]["truth_station"] = {
        "provider": "DWD",
        "station_id": s.station_id,
        "station_name": s.name,
        "latitude": s.latitude,
        "longitude": s.longitude,
        "elevation_m": s.elevation_m,
        "distance_km": s.distance_km,
        "date_from": None if s.date_from is None else s.date_from.isoformat(),
        "date_to": None if s.date_to is None else s.date_to.isoformat(),
        "source_id": f"dwd.10min.air_temperature.{s.station_id}",
    }
    airports_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    print(f"Selected {s.station_id} {s.name}, distance {s.distance_km} km")


if __name__ == "__main__":
    main()
