import pandas as pd

from weather_tmax_bot.features.nwp_feature_table import build_nwp_feature_table


def test_nwp_feature_table_respects_availability_time():
    nwp = pd.DataFrame(
        {
            "target_date_local": ["2026-05-29"],
            "model_availability_time_utc": ["2026-05-29T14:17:00Z"],
            "knowledge_time_utc": ["2026-05-29T14:17:00Z"],
            "model_name": ["open_meteo.icon_d2"],
            "source_id": ["open_meteo.live.icon_d2"],
            "model_tmax_c": [27.3],
        }
    )
    features = build_nwp_feature_table(nwp, ["2026-05-29"], issue_hours_utc=[12, 15])
    before = features[features["issue_hour_utc"] == 12].iloc[0]
    after = features[features["issue_hour_utc"] == 15].iloc[0]
    assert before["nwp_missing"] == True
    assert after["nwp_missing"] == False
    assert after["model_tmax_c"] == 27.3
