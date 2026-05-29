from weather_tmax_bot.models.extrapolation import detect_feature_extrapolation


class DummyModel:
    feature_ranges = {
        "last_metar_temp_c": {"min": -10.0, "max": 35.0},
        "issue_hour_utc": {"min": 0.0, "max": 18.0},
    }


def test_detect_feature_extrapolation_warns_outside_training_range():
    result = detect_feature_extrapolation({"last_metar_temp_c": 45.0, "issue_hour_utc": 6}, DummyModel())

    assert result["extrapolated"]
    assert result["severity"] == "minor"
    assert result["violations"][0]["feature"] == "last_metar_temp_c"
    assert result["warnings"]


def test_detect_feature_extrapolation_allows_inside_range():
    result = detect_feature_extrapolation({"last_metar_temp_c": 20.0, "issue_hour_utc": 6}, DummyModel())

    assert not result["extrapolated"]
    assert result["severity"] == "none"
    assert result["violations"] == []


def test_detect_feature_extrapolation_ignores_schedule_control_fields():
    class ScheduleModel:
        feature_ranges = {
            "issue_minute_utc": {"min": 0.0, "max": 0.0},
            "issue_schedule_offset_minutes": {"min": 0.0, "max": 0.0},
        }

    result = detect_feature_extrapolation(
        {"issue_minute_utc": 10, "issue_schedule_offset_minutes": 10},
        ScheduleModel(),
    )

    assert not result["extrapolated"]


def test_detect_feature_extrapolation_marks_many_or_hard_features_severe():
    result = detect_feature_extrapolation({"issue_hour_utc": 24.0}, DummyModel())

    assert result["extrapolated"]
    assert result["severity"] == "severe"
