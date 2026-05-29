from pathlib import Path


def main():
    Path("data/forecasts").mkdir(parents=True, exist_ok=True)
    print("Prepared data/forecasts. Honest forecast-as-issued NWP must be supplied or accumulated by script 11.")


if __name__ == "__main__":
    main()
