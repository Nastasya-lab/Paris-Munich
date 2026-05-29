import argparse
import json
from datetime import date

from weather_tmax_bot.operations.refresh import refresh_open_meteo_nwp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    parser.add_argument("--target-date", default=date.today().isoformat())
    parser.add_argument("--provider", default="open-meteo")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.target_date)
    if args.airport != "EDDM":
        raise SystemExit("MVP archiver currently knows EDDM coordinates only")
    if args.provider == "open-meteo":
        summary = refresh_open_meteo_nwp(args.airport, target_date)
        if summary["rows_fetched"] == 0:
            raise SystemExit("Open-Meteo returned no forecast rows for requested target date")
        print(json.dumps(summary, indent=2))
        return
    raise SystemExit(f"Unsupported provider: {args.provider}")


if __name__ == "__main__":
    main()
