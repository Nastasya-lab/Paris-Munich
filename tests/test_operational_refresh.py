from datetime import date, datetime, timezone

import pandas as pd

from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.operations import refresh as refresh_module


def test_nwp_archive_deduplicates_raw_hash(tmp_path):
    path = tmp_path / "nwp.parquet"
    rows = pd.DataFrame(
        {
            "raw_record_hash": ["h1", "h1"],
            "ingest_time_utc": [
                datetime(2026, 1, 1, 0, tzinfo=timezone.utc),
                datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            ],
            "model_tmax_c": [20.0, 21.0],
        }
    )

    NWPArchive(path).append_extract(rows)

    out = pd.read_parquet(path)
    assert len(out) == 1
    assert out.iloc[0]["model_tmax_c"] == 21.0


def test_refresh_operational_data_can_skip_network_sources(tmp_path):
    summary = refresh_module.refresh_operational_data(
        airport="EDDM",
        target_date_local=date(2026, 7, 15),
        refresh_awc=False,
        refresh_nwp=False,
        root=tmp_path,
    )

    assert summary["airport"] == "EDDM"
    assert summary["sources"] == {}
    assert "freshness_gate" in summary


def test_refresh_open_meteo_nwp_uses_fetcher(monkeypatch, tmp_path):
    def fake_fetch(**kwargs):
        return pd.DataFrame(
            {
                "raw_record_hash": ["nwp1"],
                "ingest_time_utc": [datetime(2026, 7, 14, 12, tzinfo=timezone.utc)],
                "knowledge_time_utc": [datetime(2026, 7, 14, 12, tzinfo=timezone.utc)],
                "model_tmax_c": [24.0],
            }
        )

    monkeypatch.setattr(refresh_module, "fetch_open_meteo_live_extract", fake_fetch)

    summary = refresh_module.refresh_open_meteo_nwp("EDDM", date(2026, 7, 15), root=tmp_path)

    assert summary["rows_fetched"] == 1
    assert summary["archive_rows"] == 1


def test_refresh_frame_normalizes_mixed_awc_visibility_types():
    rows = pd.DataFrame(
        {
            "visibility": [6.0, "6+"],
            "wind_direction_deg": [270, "VRB"],
            "temperature_c": ["18", 19],
        }
    )

    normalized = refresh_module._normalize_refresh_frame(rows)

    assert str(normalized["visibility"].dtype) == "string"
    assert pd.isna(normalized.iloc[1]["wind_direction_deg"])
    assert normalized.iloc[0]["temperature_c"] == 18


def test_awc_append_dedup_preserves_first_ingest_time(tmp_path):
    path = tmp_path / "awc_metar.parquet"
    first_ingest = datetime(2026, 6, 1, 12, 24, tzinfo=timezone.utc)
    later_ingest = datetime(2026, 6, 1, 12, 25, tzinfo=timezone.utc)
    refresh_module._append_dedup(
        pd.DataFrame({"raw_record_hash": ["h1"], "ingest_time_utc": [first_ingest], "knowledge_time_utc": [first_ingest]}),
        path,
    )
    refresh_module._append_dedup(
        pd.DataFrame({"raw_record_hash": ["h1"], "ingest_time_utc": [later_ingest], "knowledge_time_utc": [later_ingest]}),
        path,
    )

    out = pd.read_parquet(path)

    assert len(out) == 1
    assert pd.Timestamp(out.iloc[0]["ingest_time_utc"]) == pd.Timestamp(first_ingest)
    assert pd.Timestamp(out.iloc[0]["knowledge_time_utc"]) == pd.Timestamp(first_ingest)
