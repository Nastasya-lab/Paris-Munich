import argparse
from datetime import datetime, timezone

from weather_tmax_bot.data.iem import IEMAdapter
from weather_tmax_bot.data.storage import write_parquet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end", default="2025-12-31T23:59:00Z")
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = datetime.fromisoformat(args.end.replace("Z", "+00:00")).astimezone(timezone.utc)
    adapter = IEMAdapter()
    metar = adapter.fetch_metar(args.airport, start, end)
    taf = adapter.fetch_taf(args.airport, start, end)
    write_parquet(metar, f"data/interim/metar_iem_{args.airport}.parquet")
    write_parquet(taf, f"data/interim/taf_iem_{args.airport}.parquet")
    print(f"Wrote {len(metar)} METAR rows and {len(taf)} TAF rows")


if __name__ == "__main__":
    main()
