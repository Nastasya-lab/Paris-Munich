import argparse
import json

from weather_tmax_bot.operations.refresh import refresh_awc_live


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", default="EDDM")
    args = parser.parse_args()
    summary = refresh_awc_live(args.airport)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
