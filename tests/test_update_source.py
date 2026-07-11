from __future__ import annotations

import json

from weather_tmax_bot.operations import update_source


def _payload(*, temperature: float, tmax: float) -> dict:
    return {
        "metar_signal": {
            "latest_metar_time_utc": "2026-07-11T10:00:00+00:00",
            "latest_metar_temp_c": temperature,
        },
        "data_lineage": {
            "latest_nwp_source_id": "open_meteo.live.icon_d2",
            "model_tmax_c": tmax,
            "max_nwp_knowledge_time_utc": "2026-07-11T10:01:00+00:00",
        },
    }


def test_record_update_source_persists_and_compares_snapshots(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(update_source, "STATE_PATH", tmp_path / "forecast_input_state.json")

    first = update_source.record_update_source(
        airport="LFPB", trigger="new_metar", payload=_payload(temperature=25.0, tmax=31.0)
    )
    second = update_source.record_update_source(
        airport="LFPB", trigger="new_metar", payload=_payload(temperature=26.0, tmax=31.0)
    )

    assert first["state_persisted"] is True
    assert first["metar_changed"] is False
    assert second["metar_changed"] is True
    assert second["nwp_changed"] is False
    state = json.loads(update_source.STATE_PATH.read_text(encoding="utf-8"))
    assert state["airports"]["LFPB"]["metar"]["temperature_c"] == 26.0


def test_record_update_source_recovers_from_invalid_state_file(tmp_path, monkeypatch) -> None:
    state_path = tmp_path / "forecast_input_state.json"
    state_path.write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(update_source, "STATE_PATH", state_path)

    result = update_source.record_update_source(
        airport="EDDM", trigger="scheduled_forecast", payload=_payload(temperature=20.0, tmax=25.0)
    )

    assert result["state_persisted"] is True
    assert result["previous"] is None
