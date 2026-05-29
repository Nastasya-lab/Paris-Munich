from datetime import datetime, timezone

from weather_tmax_bot.bot.formatter import format_prediction
from weather_tmax_bot.models.predict import predict_with_climatology


def main():
    issue = datetime.now(timezone.utc)
    dist = predict_with_climatology(issue.date())
    print(
        format_prediction(
            "EDDM",
            issue.date(),
            issue,
            dist,
            model_version="climatology_mvp",
            forecast_acceptance={"accepted": True},
            warnings=["climatology MVP mode"],
        )
    )


if __name__ == "__main__":
    main()
