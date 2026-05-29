import argparse
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.build_target import build_daily_tmax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--input", default="data/interim/dwd_10min_temperature_01262.parquet")
    args = parser.parse_args()
    if not Path(args.input).exists():
        raise SystemExit(f"{args.input} not found; run script 02 with --zip-path first")
    obs = pd.read_parquet(args.input)
    target = build_daily_tmax(obs, airport_icao=args.airport)
    write_parquet(target, "data/processed/daily_target.parquet")
    print(f"Wrote {len(target)} daily targets")


if __name__ == "__main__":
    main()
