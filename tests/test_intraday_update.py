from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.intraday_update import apply_intraday_update


def test_late_post_storm_drop_suppresses_upper_tail(tmp_path):
    training_path = tmp_path / "training.parquet"
    timing_path = tmp_path / "daily_target.parquet"
    _write_training(training_path, issue_hour=15, month=5, future_increase=0.2, observed_max=29, last_temp=18)
    _write_timing(timing_path, month=5, tmax_hour=14)

    base = TmaxDistribution([29, 30, 31, 32], [0.04, 0.58, 0.28, 0.10])
    result = apply_intraday_update(
        base,
        {
            "month": 5,
            "observed_max_so_far_from_metar": 29.0,
            "last_metar_temp_c": 18.0,
            "temp_trend_3h": -10.0,
            "has_precip_recent": True,
            "has_thunder_recent": True,
        },
        date(2026, 5, 31),
        datetime(2026, 5, 31, 15, 0, tzinfo=timezone.utc),
        training_dataset_path=training_path,
        daily_target_path=timing_path,
    )

    assert result.details["active"] is True
    assert result.details["peak_passed_probability"] >= 0.9
    assert result.distribution.threshold_ge(31) < 0.06


def test_morning_rain_does_not_force_peak_passed(tmp_path):
    training_path = tmp_path / "training.parquet"
    timing_path = tmp_path / "daily_target.parquet"
    _write_training(training_path, issue_hour=6, month=5, future_increase=7.0, observed_max=18, last_temp=15)
    _write_timing(timing_path, month=5, tmax_hour=15)

    base = TmaxDistribution([18, 22, 25, 26], [0.05, 0.2, 0.5, 0.25])
    result = apply_intraday_update(
        base,
        {
            "month": 5,
            "observed_max_so_far_from_metar": 18.0,
            "last_metar_temp_c": 15.0,
            "temp_trend_3h": -3.0,
            "has_precip_recent": True,
            "has_thunder_recent": False,
            "model_temp_at_11_local": 21.0,
            "model_temp_at_14_local": 25.0,
            "model_temp_at_17_local": 24.0,
        },
        date(2026, 5, 31),
        datetime(2026, 5, 31, 6, 0, tzinfo=timezone.utc),
        training_dataset_path=training_path,
        daily_target_path=timing_path,
    )

    assert result.details["active"] is True
    assert result.details["peak_passed_probability"] < 0.5
    assert result.details["intraday_blend_weight"] <= 0.45
    assert result.distribution.expected_tmax_c >= 23.0


def _write_training(path, *, issue_hour: int, month: int, future_increase: float, observed_max: float, last_temp: float) -> None:
    rows = []
    for idx in range(80):
        rows.append(
            {
                "tmax_c": observed_max + future_increase + (idx % 3) * 0.1,
                "observed_max_so_far_from_metar": observed_max,
                "last_metar_temp_c": last_temp,
                "issue_hour_utc": issue_hour,
                "month": month,
                "temp_trend_3h": last_temp - observed_max,
                "has_precip_recent": True,
                "has_thunder_recent": issue_hour >= 12,
            }
        )
    pd.DataFrame(rows).to_parquet(path)


def _write_timing(path, *, month: int, tmax_hour: int) -> None:
    rows = []
    for day in range(1, 29):
        rows.append(
            {
                "target_date_local": f"2026-{month:02d}-{day:02d}",
                "tmax_time_local": f"2026-{month:02d}-{day:02d}T{tmax_hour:02d}:00:00+02:00",
            }
        )
    pd.DataFrame(rows).to_parquet(path)
