from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from weather_tmax_bot.polymarket_paper.config import PaperTradingConfig
from weather_tmax_bot.polymarket_paper.forecast import ForecastSignal
from weather_tmax_bot.polymarket_paper.mapping import probability_for_bucket
from weather_tmax_bot.polymarket_paper.models import (
    MarketSnapshot,
    PaperPosition,
    TemperatureMarket,
    TradeCandidate,
    TradeEvent,
)
from weather_tmax_bot.polymarket_paper.quotes import quote_buy, quote_sell
from weather_tmax_bot.polymarket_paper.state import PaperState


class PaperTradingEngine:
    def __init__(self, config: PaperTradingConfig) -> None:
        self.config = config

    def process(
        self,
        signal: ForecastSignal,
        snapshot: MarketSnapshot,
        state: PaperState,
        resolved_token_prices: dict[str, float] | None = None,
        resolved_token_reasons: dict[str, str] | None = None,
    ) -> dict:
        now = datetime.now(UTC).isoformat()
        market_by_id = {market.market_id: market for market in snapshot.markets}
        events: list[TradeEvent] = []
        holds: list[dict] = []
        self._settle_positions(
            signal,
            state,
            resolved_token_prices or {},
            resolved_token_reasons or {},
            now,
            events,
        )
        self._review_positions(signal, state, market_by_id, now, events, holds)
        candidates = self._build_candidates(signal, snapshot, state)
        self._open_candidates(signal, state, candidates, now, events)
        state.last_forecast_id = signal.forecast_id
        return {
            "status": "processed",
            "forecast_id": signal.forecast_id,
            "signal_variant": signal.variant,
            "target_date_local": signal.target_date_local.isoformat(),
            "settlement_verified": snapshot.settlement_verified,
            "settlement_notes": snapshot.settlement_notes,
            "events": [event.to_dict() for event in events],
            "holds": holds,
            "candidate_count": len(candidates),
            "open_positions": [position.to_dict() for position in state.positions],
            "cash_balance_usd": round(state.cash_balance_usd, 2),
            "realized_pnl_usd": round(state.realized_pnl_usd, 2),
        }

    def _settle_positions(
        self,
        signal: ForecastSignal,
        state: PaperState,
        resolved_token_prices: dict[str, float],
        resolved_token_reasons: dict[str, str],
        now: str,
        events: list[TradeEvent],
    ) -> None:
        remaining = []
        for position in state.positions:
            payout = resolved_token_prices.get(position.token_id)
            if payout is None:
                remaining.append(position)
                continue
            proceeds = position.shares * payout
            pnl = proceeds - position.size_usd
            state.cash_balance_usd += proceeds
            state.realized_pnl_usd += pnl
            event = TradeEvent(
                event_id=uuid4().hex,
                action="SETTLE",
                position_id=position.position_id,
                occurred_at_utc=now,
                market_id=position.market_id,
                question=position.question,
                side=position.side,
                price=payout,
                shares=position.shares,
                notional_usd=proceeds,
                model_probability=float(
                    position.last_model_probability
                    if position.last_model_probability is not None
                    else position.entry_model_probability
                ),
                production_probability=position.entry_production_probability,
                raw_edge=0.0,
                effective_edge=0.0,
                realized_pnl_usd=pnl,
                reason=resolved_token_reasons.get(position.token_id, "official_polymarket_resolution"),
                forecast_id=signal.forecast_id,
            )
            state.events.append(event)
            events.append(event)
        state.positions = remaining

    def _review_positions(
        self,
        signal: ForecastSignal,
        state: PaperState,
        market_by_id: dict[str, TemperatureMarket],
        now: str,
        events: list[TradeEvent],
        holds: list[dict],
    ) -> None:
        remaining: list[PaperPosition] = []
        for position in state.positions:
            market = market_by_id.get(position.market_id)
            if market is None:
                remaining.append(position)
                holds.append(
                    {
                        "position_id": position.position_id,
                        "action": "HOLD",
                        "reason": "market_not_in_active_snapshot",
                    }
                )
                continue
            fair_yes = probability_for_bucket(
                signal.shadow_probabilities,
                market.temperature_c,
                market.tail,
            )
            production_yes = probability_for_bucket(
                signal.production_probabilities,
                market.temperature_c,
                market.tail,
            )
            fair = fair_yes if position.side == "YES" else 1.0 - fair_yes
            production = production_yes if position.side == "YES" else 1.0 - production_yes
            bid_levels = market.yes_bids if position.side == "YES" else market.no_bids
            quote = quote_sell(bid_levels, position.shares)
            if quote.average_price is None or quote.fill_ratio < self.config.min_fill_ratio:
                remaining.append(position)
                holds.append(
                    {
                        "position_id": position.position_id,
                        "action": "HOLD",
                        "reason": "insufficient_exit_liquidity",
                    }
                )
                continue
            raw_edge = fair - quote.average_price
            effective_edge = raw_edge - self.config.calibration_buffer - self.config.cost_buffer
            position.last_price = quote.average_price
            position.last_model_probability = fair
            position.last_unrealized_pnl_usd = quote.notional_usd - position.size_usd
            position.updated_at_utc = now
            if effective_edge > self.config.close_effective_edge:
                remaining.append(position)
                holds.append(
                    {
                        "position_id": position.position_id,
                        "action": "HOLD",
                        "effective_edge": effective_edge,
                        "price": quote.average_price,
                    }
                )
                continue
            proceeds = quote.notional_usd
            pnl = proceeds - position.size_usd
            state.cash_balance_usd += proceeds
            state.realized_pnl_usd += pnl
            event = TradeEvent(
                event_id=uuid4().hex,
                action="SELL",
                position_id=position.position_id,
                occurred_at_utc=now,
                market_id=position.market_id,
                question=position.question,
                side=position.side,
                price=quote.average_price,
                shares=quote.shares,
                notional_usd=proceeds,
                model_probability=fair,
                production_probability=production,
                raw_edge=raw_edge,
                effective_edge=effective_edge,
                realized_pnl_usd=pnl,
                reason="effective_edge_at_or_below_close_threshold",
                forecast_id=signal.forecast_id,
            )
            state.events.append(event)
            events.append(event)
        state.positions = remaining

    def _build_candidates(
        self,
        signal: ForecastSignal,
        snapshot: MarketSnapshot,
        state: PaperState,
    ) -> list[TradeCandidate]:
        if self.config.require_verified_settlement and not snapshot.settlement_verified:
            return []
        open_tokens = {position.token_id for position in state.positions}
        daily_exposure = sum(
            position.size_usd
            for position in state.positions
            if position.target_date_local == signal.target_date_local.isoformat()
        )
        equity = max(state.equity_basis_usd, 1.0)
        position_budget = min(
            equity * self.config.max_position_fraction,
            equity * self.config.max_daily_exposure_fraction - daily_exposure,
            state.cash_balance_usd,
        )
        if position_budget <= 0:
            return []
        candidates: list[TradeCandidate] = []
        for market in snapshot.markets:
            fair_yes = probability_for_bucket(
                signal.shadow_probabilities,
                market.temperature_c,
                market.tail,
            )
            production_yes = probability_for_bucket(
                signal.production_probabilities,
                market.temperature_c,
                market.tail,
            )
            for side, token_id, levels, fair, production in (
                ("YES", market.yes_token_id, market.yes_asks, fair_yes, production_yes),
                ("NO", market.no_token_id, market.no_asks, 1.0 - fair_yes, 1.0 - production_yes),
            ):
                if token_id in open_tokens:
                    continue
                quote = quote_buy(levels, position_budget)
                if quote.average_price is None or quote.fill_ratio < self.config.min_fill_ratio:
                    continue
                if not (
                    self.config.min_contract_price
                    <= quote.average_price
                    <= self.config.max_contract_price
                ):
                    continue
                raw_edge = fair - quote.average_price
                effective_edge = (
                    raw_edge - self.config.calibration_buffer - self.config.cost_buffer
                )
                if effective_edge < self.config.min_effective_edge:
                    continue
                candidates.append(
                    TradeCandidate(
                        market_id=market.market_id,
                        market_slug=market.slug,
                        question=market.question,
                        token_id=token_id,
                        side=side,
                        temperature_c=market.temperature_c,
                        tail=market.tail,
                        model_probability=fair,
                        production_probability=production,
                        quote=quote,
                        raw_edge=raw_edge,
                        effective_edge=effective_edge,
                    )
                )
        candidates.sort(
            key=lambda candidate: (
                -candidate.effective_edge,
                -candidate.quote.fill_ratio,
                candidate.quote.average_price or 1.0,
            )
        )
        return candidates

    def _open_candidates(
        self,
        signal: ForecastSignal,
        state: PaperState,
        candidates: list[TradeCandidate],
        now: str,
        events: list[TradeEvent],
    ) -> None:
        used_markets = {position.market_id for position in state.positions}
        for candidate in candidates:
            if len(state.positions) >= self.config.max_positions:
                break
            if candidate.market_id in used_markets:
                continue
            daily_exposure = sum(
                position.size_usd
                for position in state.positions
                if position.target_date_local == signal.target_date_local.isoformat()
            )
            max_daily = state.equity_basis_usd * self.config.max_daily_exposure_fraction
            remaining_daily = max(0.0, max_daily - daily_exposure)
            size = min(candidate.quote.notional_usd, remaining_daily, state.cash_balance_usd)
            if size <= 0:
                break
            scale = size / candidate.quote.notional_usd
            shares = candidate.quote.shares * scale
            position = PaperPosition(
                position_id=uuid4().hex,
                market_id=candidate.market_id,
                question=candidate.question,
                token_id=candidate.token_id,
                side=candidate.side,
                temperature_c=candidate.temperature_c,
                tail=candidate.tail,
                target_date_local=signal.target_date_local.isoformat(),
                entry_price=float(candidate.quote.average_price),
                entry_model_probability=candidate.model_probability,
                entry_production_probability=candidate.production_probability,
                entry_raw_edge=candidate.raw_edge,
                entry_effective_edge=candidate.effective_edge,
                size_usd=size,
                shares=shares,
                opened_at_utc=now,
                forecast_id=signal.forecast_id,
                market_slug=candidate.market_slug,
                last_price=float(candidate.quote.average_price),
                last_model_probability=candidate.model_probability,
                updated_at_utc=now,
            )
            state.positions.append(position)
            state.cash_balance_usd -= size
            event = TradeEvent(
                event_id=uuid4().hex,
                action="BUY",
                position_id=position.position_id,
                occurred_at_utc=now,
                market_id=position.market_id,
                question=position.question,
                side=position.side,
                price=position.entry_price,
                shares=position.shares,
                notional_usd=position.size_usd,
                model_probability=position.entry_model_probability,
                production_probability=position.entry_production_probability,
                raw_edge=position.entry_raw_edge,
                effective_edge=position.entry_effective_edge,
                realized_pnl_usd=None,
                reason="shadow_unimodal_effective_edge_above_entry_threshold",
                forecast_id=signal.forecast_id,
            )
            state.events.append(event)
            events.append(event)
            used_markets.add(candidate.market_id)
