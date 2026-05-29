from weather_tmax_bot.operations.quality import assess_forecast_quality


def test_forecast_quality_marks_stale_as_degraded():
    quality = assess_forecast_quality(
        {
            "freshness": {
                "metar": {"state": "stale"},
                "taf": {"state": "fresh"},
                "nwp": {"state": "fresh"},
            }
        },
        [],
    )

    assert quality["status"] == "degraded"
    assert "metar is stale" in quality["reasons"]
    assert "--auto-refresh" in quality["recommendation"]


def test_forecast_quality_marks_future_timestamp_as_invalid():
    quality = assess_forecast_quality({"freshness": {"metar": {"state": "future_timestamp"}}}, [])

    assert quality["status"] == "invalid"


def test_forecast_quality_marks_clean_forecast_ok():
    quality = assess_forecast_quality({"freshness": {"metar": {"state": "fresh"}}}, [])

    assert quality["status"] == "ok"
    assert quality["reasons"] == []
    assert quality["cautions"] == []


def test_forecast_quality_distinguishes_known_compatible_source_difference():
    quality = assess_forecast_quality(
        {
            "freshness": {"metar": {"state": "fresh"}},
            "source_compatibility": {"metar": {"status": "known_runtime_compatible"}},
        },
        [],
    )

    assert quality["status"] == "ok"
    assert "known compatible runtime source differs from training source" in quality["cautions"]
    assert quality["reasons"] == []


def test_forecast_quality_marks_unknown_source_difference():
    quality = assess_forecast_quality(
        {
            "freshness": {"metar": {"state": "fresh"}},
            "source_compatibility": {"metar": {"status": "unknown_runtime_source"}},
        },
        [],
    )

    assert quality["status"] == "degraded"
    assert "unknown runtime source differs from training source" in quality["reasons"]


def test_forecast_quality_can_be_ok_with_only_known_compatible_source_caution():
    quality = assess_forecast_quality(
        {
            "freshness": {"metar": {"state": "fresh"}, "taf": {"state": "fresh"}, "nwp": {"state": "fresh"}},
            "source_compatibility": {
                "metar": {"status": "known_runtime_compatible"},
                "taf": {"status": "known_runtime_compatible"},
            },
        },
        [],
    )

    assert quality["status"] == "ok"
    assert len(quality["cautions"]) == 1


def test_forecast_quality_preliminary_calibration_is_caution_only():
    quality = assess_forecast_quality(
        {"freshness": {"metar": {"state": "fresh"}}},
        ["Quantile MVP model used; calibration layer is still preliminary."],
    )

    assert quality["status"] == "ok"
    assert "calibration is preliminary" in quality["cautions"]


def test_forecast_quality_treats_minor_extrapolation_as_caution():
    quality = assess_forecast_quality(
        {
            "freshness": {"metar": {"state": "fresh"}},
            "extrapolation": {"extrapolated": True, "severity": "minor"},
        },
        [],
    )

    assert quality["status"] == "ok"
    assert "minor live feature extrapolation" in quality["cautions"]


def test_forecast_quality_treats_severe_extrapolation_as_degraded():
    quality = assess_forecast_quality(
        {
            "freshness": {"metar": {"state": "fresh"}},
            "extrapolation": {"extrapolated": True, "severity": "severe"},
        },
        [],
    )

    assert quality["status"] == "degraded"
    assert "live features outside training range" in quality["reasons"]
