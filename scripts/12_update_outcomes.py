from weather_tmax_bot.evaluation.outcomes import build_forecast_outcome_status, update_forecast_outcomes


def main():
    out = update_forecast_outcomes()
    status = build_forecast_outcome_status()
    print(f"Wrote/updated {len(out)} forecast monitoring rows")
    print(f"Wrote/updated {len(status)} forecast outcome status rows")


if __name__ == "__main__":
    main()
