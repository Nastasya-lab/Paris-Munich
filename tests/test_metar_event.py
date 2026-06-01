import json
from datetime import date, datetime, timezone

import pandas as pd

from weather_tmax_bot.operations import metar_event


def test_metar_event_skips_when_no_new_metar(tmp_path, monkeypatch):
    _write_metar(tmp_path, "2026-06-01T07:50:00Z")

    def fake_refresh(airport, root):
        _write_metar(tmp_path, "2026-06-01T07:50:00Z")
        return {"metar_rows_fetched": 1}

    monkeypatch.setattr(metar_event, "refresh_awc_live", fake_refresh)

    result = metar_event.run_metar_event_cycle(
        airport="EDDM",
        target_date_local=date(2026, 6, 1),
        issue_time_utc=datetime(2026, 6, 1, 8, tzinfo=timezone.utc),
        root=tmp_path,
        notify=False,
    )

    assert result["status"] == "no_new_metar"
    assert result["forecast_logged"] is False
    assert result["notification_sent"] is False


def test_metar_event_logs_new_metar_and_compares_probabilities(tmp_path, monkeypatch):
    _write_metar(tmp_path, "2026-06-01T07:50:00Z")
    log_path = tmp_path / "forecast_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "airport": "EDDM",
                "target_date_local": "2026-06-01",
                "forecast_id": "previous",
                "issue_time_utc": "2026-06-01T07:40:00+00:00",
                "expected_tmax_c": 22.0,
                "median_tmax_c": 22.0,
                "most_likely_integer_c": 22,
                "probability_distribution": {"22": 0.8, "25": 0.2},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_refresh(airport, root):
        _write_metar(tmp_path, "2026-06-01T08:20:00Z")
        return {"metar_rows_fetched": 1}

    def fake_run_prediction(**kwargs):
        return {
            "forecast_id": "current",
            "metadata": {"model_version": "m1"},
            "distribution": _DummyDistribution(),
            "feature_snapshot": {"forecast_components": _components()},
            "data_lineage": {},
            "warnings": [],
            "forecast_quality": {"status": "ok", "reasons": []},
            "forecast_acceptance": {"accepted": True},
        }

    sent = {}

    def fake_notify(text):
        sent["text"] = text
        return {"sent": True}

    monkeypatch.setattr(metar_event, "refresh_awc_live", fake_refresh)
    monkeypatch.setattr(metar_event, "run_prediction", fake_run_prediction)
    monkeypatch.setattr(metar_event, "notify_if_configured", fake_notify)

    result = metar_event.run_metar_event_cycle(
        airport="EDDM",
        target_date_local=date(2026, 6, 1),
        issue_time_utc=datetime(2026, 6, 1, 8, 25, tzinfo=timezone.utc),
        root=tmp_path,
        forecast_log_path=log_path,
    )

    assert result["status"] == "new_metar_forecast"
    assert result["comparison_to_previous"]["deltas"]["expected_tmax_delta_c"] == 1.0
    assert result["comparison_to_previous"]["deltas"]["ge_25_delta"] == 0.2
    assert result["notification_needed"] is True
    assert result["notification_sent"] is True
    assert "expected_tmax_changed" in result["notification_reasons"]
    assert "METAR" in sent["text"]


def test_should_notify_metar_event_includes_probability_and_shadow_changes():
    payload = {
        "forecast_components": {
            "intraday_update": {"drop_from_observed_max_c": 0.0, "peak_passed_probability": 0.1},
            "shadow_mode": {"comparison_to_champion": {"expected_tmax_delta_c": 1.2}},
        }
    }
    comparison = {
        "has_previous": True,
        "deltas": {
            "expected_tmax_delta_c": 0.1,
            "most_likely_integer_changed": False,
            "ge_20_delta": 0.0,
            "ge_25_delta": 0.11,
            "ge_30_delta": 0.0,
        },
    }

    should_notify, reasons = metar_event.should_notify_metar_event(payload, comparison)

    assert should_notify is True
    assert "ge_25_probability_changed" in reasons
    assert "shadow_differs_from_champion" in reasons


def test_should_notify_metar_event_sends_routine_new_report_when_changes_are_small():
    payload = {
        "forecast_components": {
            "intraday_update": {"drop_from_observed_max_c": 0.0, "peak_passed_probability": 0.1},
            "shadow_mode": {"comparison_to_champion": {"expected_tmax_delta_c": 0.2}},
        }
    }
    comparison = {
        "has_previous": True,
        "deltas": {
            "expected_tmax_delta_c": 0.1,
            "most_likely_integer_changed": False,
            "ge_20_delta": 0.01,
            "ge_25_delta": 0.01,
            "ge_30_delta": 0.01,
        },
    }

    should_notify, reasons = metar_event.should_notify_metar_event(payload, comparison)

    assert should_notify is True
    assert reasons == ["routine_new_metar_update"]


def _write_metar(root, timestamp: str) -> None:
    path = root / "data" / "forecasts" / "awc_metar_live_EDDM.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"observation_time_utc": [timestamp], "raw_record_hash": [timestamp]}).to_parquet(path, index=False)


def _components() -> dict:
    return {
        "intraday_update": {
            "last_metar_temp_c": 19.0,
            "observed_max_so_far_c": 24.0,
            "drop_from_observed_max_c": 5.0,
            "peak_passed_probability": 0.9,
            "intraday_blend_weight": 0.8,
        },
        "shadow_mode": {
            "intraday_update": {"intraday_blend_weight": 0.7},
            "final_model": {
                "expected_tmax_c": 22.7,
                "threshold_probabilities": {"ge_30": 0.1},
            },
            "comparison_to_champion": {"expected_tmax_delta_c": -0.3},
        },
    }


class _DummyDistribution:
    @property
    def expected_tmax_c(self):
        return 23.0

    @property
    def median_tmax_c(self):
        return 23.0

    @property
    def most_likely_integer_c(self):
        return 23

    def threshold_ge(self, threshold):
        return {20: 1.0, 25: 0.4, 30: 0.05}.get(threshold, 0.0)

    def threshold_le(self, threshold):
        return 0.0

    def interval(self, mass):
        return (22.0, 24.0)

    def to_payload(self):
        return {
            "expected_tmax_c": 23.0,
            "median_tmax_c": 23.0,
            "most_likely_integer_c": 23,
            "intervals": {"50": [22.0, 24.0], "80": [21.0, 25.0], "90": [20.0, 26.0]},
            "probabilities_by_integer_c": {"23": 0.6, "25": 0.35, "30": 0.05},
            "threshold_probabilities": {"ge_20": 1.0, "ge_25": 0.4, "ge_30": 0.05, "le_0": 0.0},
        }
