from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from weather_tmax_bot.notifications.telegram import notify_if_configured
from weather_tmax_bot.polymarket_paper.client import PolymarketPublicClient
from weather_tmax_bot.polymarket_paper.config import PaperTradingConfig
from weather_tmax_bot.polymarket_paper.engine import PaperTradingEngine
from weather_tmax_bot.polymarket_paper.forecast import (
    PARIS_TIMEZONE,
    is_in_trading_window,
    load_forecast_signal,
)
from weather_tmax_bot.polymarket_paper.reporting import format_trade_events
from weather_tmax_bot.polymarket_paper.state import (
    PaperStateStore,
    append_decision_log,
)


DEFAULT_FORECAST_PATH = Path(
    "data/reports/latest_lfpb_icon_d2_metar_tmax_prediction.json"
)
DEFAULT_METAR_PATH = Path("data/forecasts/awc_metar_live_LFPB.parquet")

AIRPORT_SPECS = {
    "LFPB": {
        "city_name": "Paris",
        "timezone": "Europe/Paris",
        "env_prefix": "LFPB",
        "report_path": "data/reports/latest_lfpb_icon_d2_metar_tmax_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_LFPB.parquet",
        "station_markers": ("lfpb", "le bourget"),
    },
    "EDDM": {
        "city_name": "Munich",
        "timezone": "Europe/Berlin",
        "env_prefix": "EDDM",
        "report_path": "data/reports/latest_operational_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_EDDM.parquet",
        "station_markers": ("eddm", "munich", "muenchen", "münchen"),
    },
    "EHAM": {
        "city_name": "Amsterdam",
        "timezone": "Europe/Amsterdam",
        "env_prefix": "EHAM",
        "report_path": "data/reports/latest_eham_icon_d2_metar_tmax_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_EHAM.parquet",
        "station_markers": ("eham", "amsterdam", "schiphol"),
    },
    "LEMD": {
        "city_name": "Madrid",
        "timezone": "Europe/Madrid",
        "env_prefix": "LEMD",
        "report_path": "data/reports/latest_lemd_multinwp_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_LEMD.parquet",
        "station_markers": ("lemd", "madrid", "barajas"),
    },
    "LIMC": {
        "city_name": "Milan",
        "timezone": "Europe/Rome",
        "env_prefix": "LIMC",
        "report_path": "data/reports/latest_limc_prediction.json",
        "metar_path": "data/forecasts/awc_metar_live_LIMC.parquet",
        "station_markers": ("limc", "milan", "malpensa"),
    },
}


def main() -> None:
    args = _parse_args()
    spec = AIRPORT_SPECS[args.airport]
    config = PaperTradingConfig.from_env(args.env_prefix or spec["env_prefix"])
    _activate_airport_telegram(args.airport)
    if not config.enabled and not args.force:
        _print_result({"status": "disabled", "enabled": False})
        return
    try:
        result = run_paper_cycle(
            Path(args.forecast_path or spec["report_path"]),
            config=config,
            force_window=args.force_window,
            airport=args.airport,
            city_name=spec["city_name"],
            timezone=ZoneInfo(spec["timezone"]),
            station_markers=spec["station_markers"],
            metar_path=Path(args.metar_path or spec["metar_path"]),
        )
        if args.notify:
            result["telegram"] = [
                notify_if_configured(message)
                for message in format_trade_events(result)
            ]
        _print_result(result)
    except Exception as exc:
        error = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            append_decision_log(config.decision_log_path, error)
        except Exception:
            pass
        _print_result(error)
        if args.fail_on_error:
            raise


def run_paper_cycle(
    forecast_path: Path,
    *,
    config: PaperTradingConfig,
    client: PolymarketPublicClient | None = None,
    force_window: bool = False,
    airport: str = "LFPB",
    city_name: str = "Paris",
    timezone: ZoneInfo = PARIS_TIMEZONE,
    station_markers: tuple[str, ...] = ("lfpb", "le bourget"),
    metar_path: Path = DEFAULT_METAR_PATH,
) -> dict:
    metadata = _read_forecast_metadata(forecast_path)
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(timezone)
    target_date_local = metadata.get("target_date_local")
    if target_date_local is not None and target_date_local < now_local.date():
        result = {
            "status": "stale_forecast",
            "forecast_id": metadata.get("forecast_id"),
            "signal_variant": config.signal_variant,
            "target_date_local": target_date_local.isoformat(),
            "current_date_local": now_local.date().isoformat(),
            "created_at_utc": now_utc.isoformat(),
        }
        append_decision_log(config.decision_log_path, result)
        return result
    try:
        signal = load_forecast_signal(
            forecast_path,
            config.signal_variant,
            expected_airport=airport,
        )
    except ValueError as exc:
        if "has no probability distribution" not in str(exc):
            raise
        result = {
            "status": "missing_signal_variant",
            "forecast_id": metadata.get("forecast_id"),
            "signal_variant": config.signal_variant,
            "target_date_local": target_date_local.isoformat() if target_date_local else None,
            "error": str(exc),
            "created_at_utc": now_utc.isoformat(),
        }
        append_decision_log(config.decision_log_path, result)
        return result
    if not force_window and not is_in_trading_window(
        signal,
        start_hour=config.local_hour_start,
        end_hour=config.local_hour_end,
        timezone=timezone,
    ):
        result = {
            "status": "outside_trading_window",
            "forecast_id": signal.forecast_id,
            "signal_variant": signal.variant,
            "target_date_local": signal.target_date_local.isoformat(),
            "issue_time_utc": signal.issue_time_utc.isoformat(),
            "created_at_utc": now_utc.isoformat(),
        }
        append_decision_log(config.decision_log_path, result)
        return result

    store = PaperStateStore(config.state_path, config.start_balance_usd)
    state = store.load()
    market_client = client or PolymarketPublicClient(config)
    resolved_token_prices = market_client.fetch_resolved_token_prices(
        {position.market_slug for position in state.positions}
    )
    fallback_prices, fallback_notes = _local_metar_resolved_token_prices(
        state.positions,
        current_date_local=now_local.date(),
        metar_path=metar_path,
        timezone=timezone,
        airport=airport,
    )
    resolved_token_reasons = {}
    for token_id, payout in fallback_prices.items():
        if token_id not in resolved_token_prices:
            resolved_token_prices[token_id] = payout
            resolved_token_reasons[token_id] = f"local_{airport.lower()}_metar_truth_fallback"
    snapshot = market_client.fetch_temperature_market(
        signal.target_date_local,
        city_name=city_name,
        station_markers=station_markers,
    )
    result = PaperTradingEngine(config).process(
        signal,
        snapshot,
        state,
        resolved_token_prices=resolved_token_prices,
        resolved_token_reasons=resolved_token_reasons,
    )
    result.update(
        {
            "airport": airport,
            "city_name": city_name,
            "event_title": snapshot.event_title,
            "event_slug": snapshot.event_slug,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    if fallback_notes:
        result["local_metar_settlement_fallback"] = fallback_notes
    store.save(state)
    append_decision_log(config.decision_log_path, result)
    return result


def _local_metar_resolved_token_prices(
    positions,
    *,
    current_date_local: date,
    metar_path: Path,
    timezone: ZoneInfo = PARIS_TIMEZONE,
    airport: str = "LFPB",
) -> tuple[dict[str, float], list[dict]]:
    if not positions or not metar_path.exists():
        return {}, []
    frame = pd.read_parquet(metar_path)
    if frame.empty or "observation_time_utc" not in frame.columns or "temperature_c" not in frame.columns:
        return {}, []
    observations = frame.copy()
    observations["observation_time_utc"] = pd.to_datetime(
        observations["observation_time_utc"],
        utc=True,
        errors="coerce",
    )
    observations["temperature_c"] = pd.to_numeric(observations["temperature_c"], errors="coerce")
    observations = observations.dropna(subset=["observation_time_utc", "temperature_c"])
    if observations.empty:
        return {}, []
    observations["target_date_local"] = observations["observation_time_utc"].dt.tz_convert(timezone).dt.date

    payouts: dict[str, float] = {}
    notes: list[dict] = []
    for position in positions:
        try:
            target_date = date.fromisoformat(str(position.target_date_local))
        except ValueError:
            continue
        if target_date >= current_date_local:
            continue
        day = observations[observations["target_date_local"] == target_date]
        if day.empty:
            continue
        actual_bin = int(round(float(day["temperature_c"].max())))
        yes_wins = _bucket_wins(actual_bin, int(position.temperature_c), str(position.tail))
        token_wins = yes_wins if position.side == "YES" else not yes_wins
        payout = 1.0 if token_wins else 0.0
        payouts[str(position.token_id)] = payout
        notes.append(
            {
                "position_id": position.position_id,
                "target_date_local": target_date.isoformat(),
                "actual_metar_tmax_integer_c": actual_bin,
                "side": position.side,
                "temperature_c": position.temperature_c,
                "tail": position.tail,
                "payout": payout,
                "reason": f"local_{airport.lower()}_metar_truth_fallback",
            }
        )
    return payouts, notes


def _bucket_wins(actual_bin: int, temperature_c: int, tail: str) -> bool:
    if tail == "or_higher":
        return actual_bin >= temperature_c
    if tail == "or_lower":
        return actual_bin <= temperature_c
    return actual_bin == temperature_c


def _read_forecast_metadata(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_date = None
    raw_target = payload.get("target_date_local")
    if raw_target:
        try:
            target_date = date.fromisoformat(str(raw_target))
        except ValueError:
            target_date = None
    return {
        "forecast_id": payload.get("forecast_id"),
        "target_date_local": target_date,
    }


def _activate_airport_telegram(airport: str) -> None:
    token = os.getenv(f"TELEGRAM_BOT_TOKEN_{airport}")
    chat_id = os.getenv(f"TELEGRAM_CHAT_ID_{airport}")
    if airport == "EHAM":
        token = token or os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
        chat_id = chat_id or "-1004216691526"
    elif airport == "LEMD":
        token = token or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
        chat_id = chat_id or "-1004409683948"
    elif airport == "LIMC":
        token = token or os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
        chat_id = chat_id or "-1004371899833"
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    if chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = chat_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an isolated airport Polymarket paper trader."
    )
    parser.add_argument("--airport", choices=sorted(AIRPORT_SPECS), default="LFPB")
    parser.add_argument("--forecast-path", default=None)
    parser.add_argument("--metar-path", default=None)
    parser.add_argument("--env-prefix", default=None)
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when the selected airport paper-trading module is disabled.",
    )
    parser.add_argument(
        "--force-window",
        action="store_true",
        help="Ignore the configured airport-local trading window.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code after logging an error.",
    )
    return parser.parse_args()


def _print_result(result: dict) -> None:
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
