import argparse
from datetime import date

from weather_tmax_bot.data.dwd_observations import DWDAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--zip-path")
    parser.add_argument("--station-id", default="01262")
    args = parser.parse_args()
    adapter = DWDAdapter()
    if args.zip_path:
        df = adapter.parse_10min_air_temperature_zip(args.zip_path, station_id=args.station_id)
        print(f"Parsed {len(df)} DWD rows")
    else:
        if not args.start or not args.end:
            raise SystemExit("Provide --start YYYY-MM-DD and --end YYYY-MM-DD, or pass --zip-path")
        df = adapter.fetch_observations(
            airport=args.airport,
            start=date.fromisoformat(args.start),
            end=date.fromisoformat(args.end),
            station_id=args.station_id,
        )
        print(f"Downloaded/parsed {len(df)} DWD rows")


if __name__ == "__main__":
    main()
