from __future__ import annotations

import json
import importlib.util
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import pandas as pd

from weather_tmax_bot.polymarket_paper.client import (
    PolymarketPublicClient,
    validate_settlement_text,
)
from weather_tmax_bot.polymarket_paper.config import PaperTradingConfig
from weather_tmax_bot.polymarket_paper.engine import PaperTradingEngine
from weather_tmax_bot.polymarket_paper.forecast import (
    is_in_trading_window,
    load_forecast_signal,
)
from weather_tmax_bot.polymarket_paper.mapping import (
    parse_temperature_bucket,
    probability_for_bucket,
)
from weather_tmax_bot.polymarket_paper.models import (
    MarketSnapshot,
    OrderLevel,
    PaperPosition,
    TemperatureMarket,
)
from weather_tmax_bot.polymarket_paper.quotes import quote_buy, quote_sell
from weather_tmax_bot.polymarket_paper.reporting import format_trade_events
from weather_tmax_bot.polymarket_paper.state import PaperState, PaperStateStore


def _load_paper_job_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "83_lfpb_polymarket_paper_job.py"
    spec = importlib.util.spec_from_file_location("lfpb_polymarket_paper_job", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temperature_bucket_mapping_supports_tails_and_negative_values():
    assert parse_temperature_bucket("Will Paris be -3 C on June 23?") == (-3, "exact")
    assert parse_temperature_bucket("Will Paris be 27°C or higher?") == (27, "or_higher")
    assert parse_temperature_bucket("Will Paris be 19 C or below?") == (19, "or_lower")

    probabilities = {18: 0.1, 19: 0.2, 20: 0.3, 21: 0.4}
    assert probability_for_bucket(probabilities, 20, "exact") == pytest.approx(0.3)
    assert probability_for_bucket(probabilities, 20, "or_higher") == pytest.approx(0.7)
    assert probability_for_bucket(probabilities, 19, "or_lower") == pytest.approx(0.3)


def test_orderbook_quotes_use_real_depth_without_fallback_liquidity():
    asks = [OrderLevel(0.30, 10), OrderLevel(0.40, 10)]
    buy = quote_buy(asks, 5.0)
    assert buy.fill_ratio == pytest.approx(1.0)
    assert buy.average_price == pytest.approx(5.0 / 15.0)
    assert buy.levels_used == 2

    thin = quote_buy([OrderLevel(0.30, 1)], 5.0)
    assert thin.fill_ratio == pytest.approx(0.06)

    sell = quote_sell([OrderLevel(0.50, 3), OrderLevel(0.40, 10)], 5)
    assert sell.fill_ratio == pytest.approx(1.0)
    assert sell.average_price == pytest.approx(0.46)


def test_forecast_loader_uses_only_requested_shadow_variant(tmp_path):
    path = tmp_path / "forecast.json"
    path.write_text(
        json.dumps(
            {
                "forecast_id": "f1",
                "airport": "LFPB",
                "target_date_local": "2026-06-23",
                "issue_time_utc": "2026-06-23T10:30:00+00:00",
                "forecast": {
                    "probabilities_by_integer_c": {"25": 0.8, "26": 0.2}
                },
                "forecast_variants": {
                    "shadow_unimodal_pmf": {
                        "distribution": {
                            "probabilities_by_integer_c": {"25": 0.25, "26": 0.75}
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    signal = load_forecast_signal(path, "shadow_unimodal_pmf")
    assert signal.shadow_probabilities == {25: 0.25, 26: 0.75}
    assert signal.production_probabilities == {25: 0.8, 26: 0.2}
    assert is_in_trading_window(signal, start_hour=10, end_hour=17)

    production_signal = load_forecast_signal(path, "production_champion")
    assert production_signal.shadow_probabilities == {25: 0.8, 26: 0.2}
    assert production_signal.production_probabilities == {25: 0.8, 26: 0.2}

    with pytest.raises(ValueError, match="has no probability"):
        load_forecast_signal(path, "production_champion_missing")


def test_settlement_validation_is_explicit():
    verified, notes = validate_settlement_text(
        "Highest temperature in Paris. Resolution source: Weather Underground."
    )
    assert verified is True
    assert notes == []

    verified, notes = validate_settlement_text("Highest value today")
    assert verified is False
    assert notes


def test_client_selects_exact_date_from_year_in_slug(tmp_path):
    client = PolymarketPublicClient(_config(tmp_path))
    events = [
        {
            "title": "Highest temperature in Paris on June 24?",
            "slug": "highest-temperature-in-paris-on-june-24-2026",
        },
        {
            "title": "Highest temperature in Paris on June 23?",
            "slug": "highest-temperature-in-paris-on-june-23-2026",
        },
    ]
    selected = client._select_event(events, date(2026, 6, 23))
    assert selected["slug"].endswith("june-23-2026")


def test_client_selects_the_requested_city_market(tmp_path):
    client = PolymarketPublicClient(_config(tmp_path))
    events = [
        {
            "title": "Highest temperature in Munich on June 23?",
            "slug": "highest-temperature-in-munich-on-june-23-2026",
        },
        {
            "title": "Highest temperature in Paris on June 23?",
            "slug": "highest-temperature-in-paris-on-june-23-2026",
        },
    ]
    selected = client._select_event(events, date(2026, 6, 23), city_name="Munich")
    assert "munich" in selected["slug"]


def test_config_uses_isolated_airport_prefixes(monkeypatch, tmp_path):
    monkeypatch.setenv("EDDM_POLYMARKET_STATE_PATH", str(tmp_path / "eddm-state.json"))
    monkeypatch.setenv("EDDM_POLYMARKET_DECISION_LOG_PATH", str(tmp_path / "eddm-decisions.jsonl"))
    config = PaperTradingConfig.from_env("EDDM")

    assert config.state_path.name == "eddm-state.json"
    assert config.decision_log_path.name == "eddm-decisions.jsonl"
    assert config.signal_variant == "production_champion"


def test_new_airports_have_isolated_production_trading_specs(monkeypatch):
    paper_job = _load_paper_job_module()

    assert paper_job.AIRPORT_SPECS["LEMD"] == {
        "city_name": "Madrid",
        "timezone": "Europe/Madrid",
        "env_prefix": "LEMD",
        "report_path": "data/reports/latest_lemd_multinwp_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_LEMD.parquet",
        "station_markers": ("lemd", "madrid", "barajas"),
    }
    assert paper_job.AIRPORT_SPECS["LIMC"] == {
        "city_name": "Milan",
        "timezone": "Europe/Rome",
        "env_prefix": "LIMC",
        "report_path": "data/reports/latest_limc_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_LIMC.parquet",
        "station_markers": ("limc", "milan", "malpensa"),
    }
    assert paper_job.AIRPORT_SPECS["RCSS"] == {
        "city_name": "Taipei",
        "timezone": "Asia/Taipei",
        "env_prefix": "RCSS",
        "report_path": "data/reports/latest_rcss_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_RCSS.parquet",
        "station_markers": ("rcss", "taipei", "songshan"),
    }

    for airport in ("LEMD", "LIMC", "RCSS"):
        config = PaperTradingConfig.from_env(airport)
        assert config.signal_variant == "production_champion"
        assert config.allow_yes_positions is False
        assert config.signal_confirmations_required == 2
        assert config.max_positions == 5
        assert config.state_path.name == f"{airport.lower()}_paper_state.json"
        assert config.decision_log_path.name == f"{airport.lower()}_paper_decisions.jsonl"


@pytest.mark.parametrize(
    ("airport", "chat_id"),
    [
        ("LEMD", "-1004409683948"),
        ("LIMC", "-1004371899833"),
        ("RCSS", "-1004469237763"),
    ],
)
def test_new_airport_trading_telegram_routing(monkeypatch, airport, chat_id):
    paper_job = _load_paper_job_module()
    monkeypatch.delenv(f"TELEGRAM_BOT_TOKEN_{airport}", raising=False)
    monkeypatch.delenv(f"TELEGRAM_CHAT_ID_{airport}", raising=False)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "main-token")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN_LFPB", "paris-token")

    paper_job._activate_airport_telegram(airport)

    assert paper_job.os.environ["TELEGRAM_BOT_TOKEN"] == "main-token"
    assert paper_job.os.environ["TELEGRAM_CHAT_ID"] == chat_id


def test_forecast_loader_supports_flat_operational_airport_report(tmp_path):
    path = tmp_path / "eddm.json"
    path.write_text(
        json.dumps(
            {
                "forecast_id": None,
                "airport": "EDDM",
                "target_date_local": "2026-06-23",
                "issue_time_utc": "2026-06-23T10:30:00+00:00",
                "probabilities_by_integer_c": {"25": 0.25, "26": 0.75},
            }
        ),
        encoding="utf-8",
    )

    signal = load_forecast_signal(path, "production_champion", expected_airport="EDDM")
    assert signal.shadow_probabilities == {25: 0.25, 26: 0.75}
    assert signal.forecast_id == "EDDM:2026-06-23T10:30:00+00:00"


def test_engine_opens_best_shadow_trade_and_reserves_cash(tmp_path):
    config = _config(tmp_path)
    signal = _signal({25: 0.2, 26: 0.7, 27: 0.1}, {25: 0.4, 26: 0.4, 27: 0.2})
    snapshot = _snapshot(
        [
            _market(
                market_id="m26",
                temperature=26,
                yes_asks=[OrderLevel(0.40, 100)],
                yes_bids=[OrderLevel(0.39, 100)],
                no_asks=[OrderLevel(0.62, 100)],
                no_bids=[OrderLevel(0.59, 100)],
            )
        ]
    )
    state = PaperState(1, 1000.0, 1000.0, 0.0)
    result = PaperTradingEngine(config).process(signal, snapshot, state)

    assert [event["action"] for event in result["events"]] == ["BUY"]
    assert result["events"][0]["side"] == "YES"
    assert result["events"][0]["model_probability"] == pytest.approx(0.7)
    assert result["events"][0]["production_probability"] == pytest.approx(0.4)
    assert len(state.positions) == 1
    assert state.positions[0].size_usd == pytest.approx(10.0)
    assert state.cash_balance_usd == pytest.approx(990.0)

    messages = format_trade_events(result)
    assert "shadow-unimodal" in messages[0]
    assert "Вероятность production" in messages[0]


def test_engine_sells_when_shadow_edge_disappears(tmp_path):
    config = _config(tmp_path)
    signal = _signal({26: 0.45, 27: 0.55}, {26: 0.6, 27: 0.4})
    position = PaperPosition(
        position_id="p1",
        market_id="m26",
        question="Will the highest temperature in Paris be 26 C on June 23?",
        token_id="yes-m26",
        side="YES",
        temperature_c=26,
        tail="exact",
        target_date_local="2026-06-23",
        entry_price=0.30,
        entry_model_probability=0.70,
        entry_production_probability=0.50,
        entry_raw_edge=0.40,
        entry_effective_edge=0.34,
        size_usd=10.0,
        shares=10.0 / 0.30,
        opened_at_utc="2026-06-23T10:00:00+00:00",
        forecast_id="old",
        market_slug="paris-26c",
    )
    state = PaperState(1, 1000.0, 990.0, 0.0, positions=[position])
    snapshot = _snapshot(
        [
            _market(
                market_id="m26",
                temperature=26,
                yes_asks=[OrderLevel(0.46, 100)],
                yes_bids=[OrderLevel(0.44, 100)],
                no_asks=[OrderLevel(0.57, 100)],
                no_bids=[OrderLevel(0.54, 100)],
            )
        ]
    )
    result = PaperTradingEngine(config).process(signal, snapshot, state)

    assert result["events"][0]["action"] == "SELL"
    assert result["events"][0]["realized_pnl_usd"] == pytest.approx(4.666666, rel=1e-5)
    assert not state.positions
    assert state.cash_balance_usd == pytest.approx(1004.666666, rel=1e-5)


def test_engine_opens_only_confirmed_no_candidate(tmp_path):
    config = replace(_config(tmp_path), allow_yes_positions=False, signal_confirmations_required=2)
    signal = _signal({25: 0.1, 26: 0.1, 27: 0.8}, {25: 0.2, 26: 0.2, 27: 0.6})
    snapshot = _snapshot(
        [
            _market(
                market_id="m26",
                temperature=26,
                yes_asks=[OrderLevel(0.20, 100)],
                yes_bids=[OrderLevel(0.18, 100)],
                no_asks=[OrderLevel(0.60, 100)],
                no_bids=[OrderLevel(0.58, 100)],
            )
        ]
    )
    state = PaperState(1, 1000.0, 1000.0, 0.0)
    engine = PaperTradingEngine(config)

    first = engine.process(signal, snapshot, state)
    assert first["events"] == []
    assert first["candidate_snapshots"][0]["side"] == "NO"
    assert first["holds"][0]["reason"] == "entry_confirmation_pending"
    assert state.pending_entries

    duplicate = engine.process(signal, snapshot, state)
    assert duplicate["events"] == []
    assert duplicate["holds"][0]["confirmation_count"] == 1

    confirmed = replace(signal, forecast_id="f-confirmed")
    result = engine.process(confirmed, snapshot, state)
    assert [event["action"] for event in result["events"]] == ["BUY"]
    assert result["events"][0]["side"] == "NO"


def test_engine_sells_only_after_confirmed_exit_signal(tmp_path):
    config = replace(_config(tmp_path), signal_confirmations_required=2)
    signal = _signal({26: 0.45, 27: 0.55}, {26: 0.6, 27: 0.4})
    position = PaperPosition(
        position_id="p-no",
        market_id="m26",
        question="Will the highest temperature in Paris be 26 C on June 23?",
        token_id="no-m26",
        side="NO",
        temperature_c=26,
        tail="exact",
        target_date_local="2026-06-23",
        entry_price=0.30,
        entry_model_probability=0.70,
        entry_production_probability=0.50,
        entry_raw_edge=0.40,
        entry_effective_edge=0.34,
        size_usd=10.0,
        shares=10.0 / 0.30,
        opened_at_utc="2026-06-23T10:00:00+00:00",
        forecast_id="old",
        market_slug="paris-26c",
    )
    state = PaperState(1, 1000.0, 990.0, 0.0, positions=[position])
    snapshot = _snapshot(
        [
            _market(
                market_id="m26",
                temperature=26,
                yes_asks=[OrderLevel(0.46, 100)],
                yes_bids=[OrderLevel(0.44, 100)],
                no_asks=[OrderLevel(0.57, 100)],
                no_bids=[OrderLevel(0.54, 100)],
            )
        ]
    )
    engine = PaperTradingEngine(config)

    first = engine.process(signal, snapshot, state)
    assert first["events"] == []
    assert first["holds"][0]["reason"] == "exit_confirmation_pending"
    assert len(state.positions) == 1

    result = engine.process(replace(signal, forecast_id="f-exit-confirmed"), snapshot, state)
    assert result["events"][0]["action"] == "SELL"
    assert not state.positions


def test_engine_settles_position_from_official_token_price(tmp_path):
    config = _config(tmp_path)
    signal = _signal({26: 0.5, 27: 0.5}, {26: 0.5, 27: 0.5})
    position = PaperPosition(
        position_id="p-settle",
        market_id="old-market",
        question="Will the highest temperature in Paris be 26 C on June 22?",
        token_id="old-yes-token",
        side="YES",
        temperature_c=26,
        tail="exact",
        target_date_local="2026-06-22",
        entry_price=0.40,
        entry_model_probability=0.60,
        entry_production_probability=0.55,
        entry_raw_edge=0.20,
        entry_effective_edge=0.14,
        size_usd=10.0,
        shares=25.0,
        opened_at_utc="2026-06-22T10:00:00+00:00",
        forecast_id="old",
        market_slug="paris-june-22-26c",
    )
    state = PaperState(1, 1000.0, 990.0, 0.0, positions=[position])
    result = PaperTradingEngine(config).process(
        signal,
        _snapshot([]),
        state,
        resolved_token_prices={"old-yes-token": 1.0},
    )

    assert result["events"][0]["action"] == "SETTLE"
    assert result["events"][0]["realized_pnl_usd"] == pytest.approx(15.0)
    assert state.cash_balance_usd == pytest.approx(1015.0)
    assert not state.positions


def test_paper_job_builds_local_metar_settlement_fallback(tmp_path, monkeypatch):
    paper_job = _load_paper_job_module()
    metar_path = tmp_path / "awc_metar_live_LFPB.parquet"
    pd.DataFrame(
        [
            {
                "observation_time_utc": "2026-06-24T10:00:00+00:00",
                "temperature_c": 37.0,
            },
            {
                "observation_time_utc": "2026-06-24T13:00:00+00:00",
                "temperature_c": 39.0,
            },
        ]
    ).to_parquet(metar_path, index=False)
    monkeypatch.setattr(paper_job, "DEFAULT_METAR_PATH", metar_path)
    position = PaperPosition(
        position_id="p-local",
        market_id="old-market",
        question="Will the highest temperature in Paris be 39 C on June 24?",
        token_id="old-yes-token",
        side="YES",
        temperature_c=39,
        tail="exact",
        target_date_local="2026-06-24",
        entry_price=0.40,
        entry_model_probability=0.60,
        entry_production_probability=0.55,
        entry_raw_edge=0.20,
        entry_effective_edge=0.14,
        size_usd=10.0,
        shares=25.0,
        opened_at_utc="2026-06-24T10:00:00+00:00",
        forecast_id="old",
        market_slug="missing-from-gamma",
    )

    payouts, notes = paper_job._local_metar_resolved_token_prices(
        [position],
        current_date_local=date(2026, 6, 25),
        metar_path=metar_path,
    )

    assert payouts == {"old-yes-token": 1.0}
    assert notes[0]["actual_metar_tmax_integer_c"] == 39
    assert notes[0]["reason"] == "local_lfpb_metar_truth_fallback"


def test_state_round_trip_is_separate_and_persistent(tmp_path):
    store = PaperStateStore(tmp_path / "polymarket" / "state.json", 1000.0)
    state = store.load()
    state.cash_balance_usd = 975.0
    store.save(state)
    loaded = store.load()
    assert loaded.cash_balance_usd == 975.0
    assert loaded.start_balance_usd == 1000.0


def test_config_enables_paper_trading_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("LFPB_POLYMARKET_PAPER_ENABLED", raising=False)
    monkeypatch.setenv("LFPB_POLYMARKET_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("LFPB_POLYMARKET_DECISION_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    assert PaperTradingConfig.from_env().enabled is True
    assert PaperTradingConfig.from_env().signal_variant == "production_champion"
    assert PaperTradingConfig.from_env().allow_yes_positions is False
    assert PaperTradingConfig.from_env().signal_confirmations_required == 2

    monkeypatch.setenv("LFPB_POLYMARKET_PAPER_ENABLED", "0")
    assert PaperTradingConfig.from_env().enabled is False


def test_paper_job_skips_stale_forecast_before_network(tmp_path):
    paper_job = _load_paper_job_module()
    forecast_path = tmp_path / "stale_forecast.json"
    forecast_path.write_text(
        json.dumps(
            {
                "forecast_id": "stale-1",
                "airport": "LFPB",
                "target_date_local": "2026-06-09",
                "issue_time_utc": "2026-06-09T10:30:00+00:00",
                "forecast": {"probabilities_by_integer_c": {"25": 1.0}},
                "forecast_variants": {
                    "shadow_unimodal_pmf": {
                        "distribution": {
                            "probabilities_by_integer_c": {"25": 1.0}
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    config = _config(tmp_path)

    result = paper_job.run_paper_cycle(forecast_path, config=config, force_window=True)

    assert result["status"] == "stale_forecast"
    assert result["forecast_id"] == "stale-1"
    assert config.decision_log_path.exists()


def _config(tmp_path: Path) -> PaperTradingConfig:
    return PaperTradingConfig(
        enabled=True,
        signal_variant="shadow_unimodal_pmf",
        state_path=tmp_path / "state.json",
        decision_log_path=tmp_path / "decisions.jsonl",
        start_balance_usd=1000.0,
        calibration_buffer=0.05,
        cost_buffer=0.01,
        min_effective_edge=0.08,
        close_effective_edge=0.02,
        max_position_fraction=0.01,
        max_daily_exposure_fraction=0.02,
        max_positions=2,
        min_contract_price=0.02,
        max_contract_price=0.95,
        min_fill_ratio=0.98,
        allow_yes_positions=True,
        signal_confirmations_required=1,
        local_hour_start=10,
        local_hour_end=17,
        require_verified_settlement=False,
        gamma_api_url="https://example.test",
        clob_api_url="https://example.test",
        request_timeout_seconds=1.0,
    )


def _signal(shadow: dict[int, float], production: dict[int, float]):
    from weather_tmax_bot.polymarket_paper.forecast import ForecastSignal

    return ForecastSignal(
        forecast_id="f-new",
        issue_time_utc=datetime(2026, 6, 23, 10, 30, tzinfo=UTC),
        target_date_local=date(2026, 6, 23),
        variant="shadow_unimodal_pmf",
        shadow_probabilities=shadow,
        production_probabilities=production,
    )


def _snapshot(markets: list[TemperatureMarket]) -> MarketSnapshot:
    return MarketSnapshot(
        event_title="Highest temperature in Paris on June 23, 2026?",
        event_slug="paris-june-23",
        target_date_local="2026-06-23",
        settlement_text="Paris temperature Weather Underground",
        settlement_verified=True,
        settlement_notes=[],
        markets=markets,
    )


def _market(
    *,
    market_id: str,
    temperature: int,
    yes_asks: list[OrderLevel],
    yes_bids: list[OrderLevel],
    no_asks: list[OrderLevel],
    no_bids: list[OrderLevel],
) -> TemperatureMarket:
    return TemperatureMarket(
        market_id=market_id,
        question=f"Will the highest temperature in Paris be {temperature} C on June 23?",
        slug=market_id,
        temperature_c=temperature,
        tail="exact",
        yes_token_id=f"yes-{market_id}",
        no_token_id=f"no-{market_id}",
        yes_asks=yes_asks,
        yes_bids=yes_bids,
        no_asks=no_asks,
        no_bids=no_bids,
    )
