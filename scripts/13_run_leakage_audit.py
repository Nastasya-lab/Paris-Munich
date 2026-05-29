from weather_tmax_bot.evaluation.leakage_audit import audit_training_dataset_file


def main():
    report, passed = audit_training_dataset_file()
    print(report.to_string(index=False))
    if not passed:
        raise SystemExit("Leakage audit failed")


if __name__ == "__main__":
    main()
