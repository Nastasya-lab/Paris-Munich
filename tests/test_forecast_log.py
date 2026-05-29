from datetime import date, datetime, timezone

from weather_tmax_bot.bot.forecast_log import log_forecast
from weather_tmax_bot.models.distribution import TmaxDistribution


def test_forecast_log_writes_jsonl(tmp_path):
    path = tmp_path / "forecast_log.jsonl"
    forecast_id = log_forecast(
        "EDDM",
        datetime(2026, 7, 15, 6, tzinfo=timezone.utc),
        date(2026, 7, 15),
        TmaxDistribution([24, 25], [0.4, 0.6]),
        model_version="test_model",
        path=path,
    )
    assert forecast_id
    assert path.read_text(encoding="utf-8").count("\n") == 1
