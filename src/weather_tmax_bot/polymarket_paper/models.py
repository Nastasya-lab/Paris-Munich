from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class OrderLevel:
    price: float
    shares: float


@dataclass(slots=True)
class TemperatureMarket:
    market_id: str
    question: str
    slug: str
    temperature_c: int
    tail: str
    yes_token_id: str
    no_token_id: str
    yes_asks: list[OrderLevel] = field(default_factory=list)
    yes_bids: list[OrderLevel] = field(default_factory=list)
    no_asks: list[OrderLevel] = field(default_factory=list)
    no_bids: list[OrderLevel] = field(default_factory=list)


@dataclass(slots=True)
class MarketSnapshot:
    event_title: str
    event_slug: str
    target_date_local: str
    settlement_text: str
    settlement_verified: bool
    settlement_notes: list[str]
    markets: list[TemperatureMarket]


@dataclass(slots=True)
class ExecutionQuote:
    average_price: float | None
    shares: float
    notional_usd: float
    fill_ratio: float
    levels_used: int


@dataclass(slots=True)
class TradeCandidate:
    market_id: str
    market_slug: str
    question: str
    token_id: str
    side: str
    temperature_c: int
    tail: str
    model_probability: float
    production_probability: float | None
    quote: ExecutionQuote
    raw_edge: float
    effective_edge: float


@dataclass(slots=True)
class PaperPosition:
    position_id: str
    market_id: str
    question: str
    token_id: str
    side: str
    temperature_c: int
    tail: str
    target_date_local: str
    entry_price: float
    entry_model_probability: float
    entry_production_probability: float | None
    entry_raw_edge: float
    entry_effective_edge: float
    size_usd: float
    shares: float
    opened_at_utc: str
    forecast_id: str | None
    market_slug: str = ""
    last_price: float | None = None
    last_model_probability: float | None = None
    last_unrealized_pnl_usd: float = 0.0
    updated_at_utc: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TradeEvent:
    event_id: str
    action: str
    position_id: str
    occurred_at_utc: str
    market_id: str
    question: str
    side: str
    price: float
    shares: float
    notional_usd: float
    model_probability: float
    production_probability: float | None
    raw_edge: float
    effective_edge: float
    realized_pnl_usd: float | None
    reason: str
    forecast_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
