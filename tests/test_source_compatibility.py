from weather_tmax_bot.temporal.source_compatibility import assess_source_compatibility


def test_awc_live_sources_are_known_compatible_with_iem_training_sources():
    result = assess_source_compatibility(
        {
            "latest_metar_source_id": "awc.metar.live.EDDM",
            "latest_taf_source_id": "awc.taf.live.EDDM",
        }
    )

    assert result["sources"]["metar"]["status"] == "known_runtime_compatible"
    assert result["sources"]["taf"]["status"] == "known_runtime_compatible"
    assert result["warnings"]


def test_unknown_runtime_source_is_not_compatible():
    result = assess_source_compatibility({"latest_metar_source_id": "unknown.metar"})

    assert result["sources"]["metar"]["status"] == "unknown_runtime_source"
