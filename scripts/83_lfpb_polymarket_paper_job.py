from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path

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


def main() -> None:
    args = _parse_args()
    config = PaperTradingConfig.from_env()
    _activate_lfpb_telegram()
    if not config.enabled and not args.force:
        _print_result({"status": "disabled", "enabled": False})
        return
    try:
        result = run_paper_cycle(
            Path(args.forecast_path),
            config=config,
            force_window=args.force_window,
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
) -> dict:
    metadata = _read_forecast_metadata(forecast_path)
    now_utc = datetime.now(UTC)
    now_local = now_utc.astimezone(PARIS_TIMEZONE)
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
        signal = load_forecast_signal(forecast_path, config.signal_variant)
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
    snapshot = market_client.fetch_paris_market(signal.target_date_local)
    result = PaperTradingEngine(config).process(
        signal,
        snapshot,
        state,
        resolved_token_prices=resolved_token_prices,
    )
    result.update(
        {
            "event_title": snapshot.event_title,
            "event_slug": snapshot.event_slug,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    store.save(state)
    append_decision_log(config.decision_log_path, result)
    return result


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


def _activate_lfpb_telegram() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN_LFPB")
    chat_id = os.getenv("TELEGRAM_CHAT_ID_LFPB")
    if token:
        os.environ["TELEGRAM_BOT_TOKEN"] = token
    if chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = chat_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the isolated LFPB shadow-unimodal Polymarket paper trader."
    )
    parser.add_argument("--forecast-path", default=str(DEFAULT_FORECAST_PATH))
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even when LFPB_POLYMARKET_PAPER_ENABLED is disabled.",
    )
    parser.add_argument(
        "--force-window",
        action="store_true",
        help="Ignore the configured Paris-local trading window.",
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
