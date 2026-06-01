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


def test_backtest_can_supply_train_only_frames(tmp_path):
    training = _training_frame(issue_hour=15, month=5, future_increase=0.2, observed_max=29, last_temp=18)
    timing = _timing_frame(month=5, tmax_hour=14)
    base = TmaxDistribution([29, 30, 31], [0.05, 0.65, 0.30])

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
        training_dataset_path=tmp_path / "missing_training.parquet",
        daily_target_path=tmp_path / "missing_timing.parquet",
        training_frame=training,
        daily_target_frame=timing,
    )

    assert result.details["active"] is True
    assert result.details["training_rows"] == 80
    assert result.distribution.threshold_ge(31) < 0.05


def test_timing_prior_handles_mixed_dst_offsets(tmp_path):
    training = _training_frame(issue_hour=15, month=10, future_increase=0.2, observed_max=18, last_temp=12)
    timing = _timing_frame(month=10, tmax_hour=14)
    timing.loc[0, "tmax_time_local"] = "2025-10-01T14:00:00+02:00"
    timing.loc[1, "tmax_time_local"] = "2025-10-27T14:00:00+01:00"
    base = TmaxDistribution([18, 19, 20], [0.2, 0.6, 0.2])

    result = apply_intraday_update(
        base,
        {
            "month": 10,
            "observed_max_so_far_from_metar": 18.0,
            "last_metar_temp_c": 12.0,
            "temp_trend_3h": -5.0,
            "has_precip_recent": True,
        },
        date(2025, 10, 27),
        datetime(2025, 10, 27, 15, 0, tzinfo=timezone.utc),
        training_frame=training,
        daily_target_frame=timing,
        training_dataset_path=tmp_path / "missing_training.parquet",
        daily_target_path=tmp_path / "missing_timing.parquet",
    )

    assert result.details["active"] is True
    assert result.details["timing_peak_passed_prior"] == 1.0


def test_seasonal_shadow_uses_warm_profile_without_changing_production(tmp_path):
    training = _training_frame(issue_hour=12, month=8, future_increase=2.0, observed_max=25, last_temp=24)
    timing = _timing_frame(month=8, tmax_hour=15)
    base = TmaxDistribution([25, 26, 27, 28], [0.10, 0.30, 0.40, 0.20])
    feature_row = {
        "month": 8,
        "observed_max_so_far_from_metar": 25.0,
        "last_metar_temp_c": 24.0,
        "temp_trend_3h": 1.0,
    }

    production = apply_intraday_update(
        base,
        feature_row,
        date(2025, 8, 15),
        datetime(2025, 8, 15, 12, 0, tzinfo=timezone.utc),
        training_frame=training,
        daily_target_frame=timing,
    )
    shadow = apply_intraday_update(
        base,
        feature_row,
        date(2025, 8, 15),
        datetime(2025, 8, 15, 12, 0, tzinfo=timezone.utc),
        training_frame=training,
        daily_target_frame=timing,
        blend_weight_profile="seasonal_shadow",
    )

    assert production.details["blend_weight_profile"] == "production_dynamic_v1"
    assert production.details["shadow_mode"] is False
    assert shadow.details["blend_weight_profile"] == "seasonal_intraday_challenger_v1"
    assert shadow.details["shadow_mode"] is True
    assert shadow.details["seasonal_profile"] == "warm"
    assert shadow.details["seasonal_weight_group"] == "utc_12"
    assert shadow.details["intraday_blend_weight"] == 0.25
    assert shadow.details["intraday_blend_weight"] != production.details["intraday_blend_weight"]


def test_seasonal_shadow_activates_late_drop_override(tmp_path):
    training = _training_frame(issue_hour=15, month=8, future_increase=0.2, observed_max=29, last_temp=18)
    timing = _timing_frame(month=8, tmax_hour=14)
    base = TmaxDistribution([29, 30, 31], [0.10, 0.60, 0.30])

    shadow = apply_intraday_update(
        base,
        {
            "month": 8,
            "observed_max_so_far_from_metar": 29.0,
            "last_metar_temp_c": 18.0,
            "temp_trend_3h": -10.0,
            "has_precip_recent": True,
        },
        date(2025, 8, 15),
        datetime(2025, 8, 15, 15, 0, tzinfo=timezone.utc),
        training_frame=training,
        daily_target_frame=timing,
        blend_weight_profile="seasonal_shadow",
    )

    assert shadow.details["late_drop_override_active"] is True
    assert shadow.details["seasonal_base_weight"] == 0.70
    assert shadow.details["intraday_blend_weight"] == 0.95


def _write_training(path, *, issue_hour: int, month: int, future_increase: float, observed_max: float, last_temp: float) -> None:
    _training_frame(
        issue_hour=issue_hour,
        month=month,
        future_increase=future_increase,
        observed_max=observed_max,
        last_temp=last_temp,
    ).to_parquet(path)


def _training_frame(*, issue_hour: int, month: int, future_increase: float, observed_max: float, last_temp: float) -> pd.DataFrame:
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
    return pd.DataFrame(rows)


def _write_timing(path, *, month: int, tmax_hour: int) -> None:
    _timing_frame(month=month, tmax_hour=tmax_hour).to_parquet(path)


def _timing_frame(*, month: int, tmax_hour: int) -> pd.DataFrame:
    rows = []
    for day in range(1, 29):
        rows.append(
            {
                "target_date_local": f"2026-{month:02d}-{day:02d}",
                "tmax_time_local": f"2026-{month:02d}-{day:02d}T{tmax_hour:02d}:00:00+02:00",
            }
        )
    return pd.DataFrame(rows)
