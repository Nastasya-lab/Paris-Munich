import argparse
from pathlib import Path

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.metar_feature_table import build_metar_feature_table_from_files
from weather_tmax_bot.features.nwp_feature_table import build_nwp_feature_table_from_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metar", action="store_true")
    parser.add_argument("--nwp", action="store_true")
    args = parser.parse_args()
    run_all = not args.metar and not args.nwp
    metar_path = Path("data/interim/metar_iem_EDDM.parquet")
    if (run_all or args.metar) and metar_path.exists():
        metar_features = build_metar_feature_table_from_files(metar_path=metar_path)
        write_parquet(metar_features, "data/processed/metar_features_iem_EDDM.parquet")
        print(f"Wrote {len(metar_features)} METAR feature rows")
    elif run_all or args.metar:
        print("No METAR parquet found; skipping METAR feature table")
    nwp_path = Path("data/forecasts/open_meteo_archive.parquet")
    if (run_all or args.nwp) and nwp_path.exists():
        nwp_features = build_nwp_feature_table_from_files(nwp_path=nwp_path)
        write_parquet(nwp_features, "data/processed/nwp_features_open_meteo.parquet")
        print(f"Wrote {len(nwp_features)} NWP feature rows")
    elif run_all or args.nwp:
        print("No Open-Meteo NWP archive found; skipping NWP feature table")


if __name__ == "__main__":
    main()
