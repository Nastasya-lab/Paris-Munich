from weather_tmax_bot.bot.lineage import build_data_lineage


def test_build_data_lineage_from_feature_snapshot():
    lineage = build_data_lineage(
        {
            "latest_metar_time_utc": "2026-01-01T05:50:00Z",
            "latest_metar_source_id": "awc.metar.live.EDDM",
            "max_metar_knowledge_time_utc": "2026-01-01T05:55:00Z",
            "latest_taf_source_id": "awc.taf.live.EDDM",
            "latest_nwp_model_name": "open_meteo.icon_d2",
            "latest_nwp_source_id": "open_meteo.live.icon_d2",
            "max_nwp_knowledge_time_utc": "2026-01-01T04:00:00Z",
        }
    )
    assert lineage["metar_latest_time_utc"] == "2026-01-01T05:50:00Z"
    assert lineage["metar_source_id"] == "awc.metar.live.EDDM"
    assert lineage["taf_source_id"] == "awc.taf.live.EDDM"
    assert lineage["nwp_runs"][0]["model_name"] == "open_meteo.icon_d2"
    assert lineage["max_feature_knowledge_time_utc"] == "2026-01-01T05:55:00Z"
