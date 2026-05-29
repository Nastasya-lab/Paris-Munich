from weather_tmax_bot.evaluation.monitoring import write_monitoring_report


def main():
    path = write_monitoring_report()
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
