from weather_tmax_bot.data.awc import AWCAdapter
from weather_tmax_bot.data.awc import _parse_awc_time
import pandas as pd


class _Response:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_awc_metar_adapter_parses_json(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return _Response(
            [
                {
                    "icaoId": "EDDM",
                    "obsTime": "2026-05-28T12:20:00Z",
                    "rawOb": "EDDM 281220Z 27008KT CAVOK 18/09 Q1018",
                    "temp": 18,
                    "dewp": 9,
                    "wdir": 270,
                    "wspd": 8,
                }
            ]
        )

    monkeypatch.setattr("weather_tmax_bot.data.awc.requests.get", fake_get)
    df = AWCAdapter().fetch_latest_metar("EDDM")
    assert len(df) == 1
    assert df.iloc[0]["source_id"] == "awc.metar.live.EDDM"
    assert df.iloc[0]["temperature_c"] == 18


def test_awc_metar_adapter_can_request_recent_hours(monkeypatch):
    captured = {}

    def fake_get(*_args, **kwargs):
        captured["params"] = kwargs["params"]
        return _Response([])

    monkeypatch.setattr("weather_tmax_bot.data.awc.requests.get", fake_get)
    AWCAdapter().fetch_latest_metar("EDDM", hours=30)

    assert captured["params"]["hours"] == 30


def test_awc_metar_adapter_treats_variable_wind_as_missing_direction(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return _Response(
            [
                {
                    "icaoId": "EDDM",
                    "obsTime": "2026-05-31T13:50:00Z",
                    "rawOb": "METAR EDDM 311350Z AUTO VRB02KT SHRA 17/15 Q1015",
                    "temp": 17,
                    "dewp": 15,
                    "wdir": "VRB",
                    "wspd": 2,
                }
            ]
        )

    monkeypatch.setattr("weather_tmax_bot.data.awc.requests.get", fake_get)
    df = AWCAdapter().fetch_latest_metar("EDDM")

    assert pd.isna(df.iloc[0]["wind_direction_deg"])
    assert df.iloc[0]["wind_speed_kt"] == 2.0


def test_awc_taf_adapter_parses_json(monkeypatch):
    def fake_get(*_args, **_kwargs):
        return _Response(
            [
                {
                    "icaoId": "EDDM",
                    "issueTime": "2026-05-28T05:00:00Z",
                    "validTimeFrom": "2026-05-28T06:00:00Z",
                    "validTimeTo": "2026-05-29T12:00:00Z",
                    "rawTAF": "TAF EDDM 280500Z 2806/2912 27008KT CAVOK TX24/2814Z",
                }
            ]
        )

    monkeypatch.setattr("weather_tmax_bot.data.awc.requests.get", fake_get)
    df = AWCAdapter().fetch_latest_taf("EDDM")
    assert len(df) == 1
    assert df.iloc[0]["source_id"] == "awc.taf.live.EDDM"
    assert df.iloc[0]["taf_tx_c"] == 24


def test_parse_awc_epoch_seconds():
    assert _parse_awc_time(1779999000).year == 2026
