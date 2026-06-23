from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests

from weather_tmax_bot.polymarket_paper.config import PaperTradingConfig
from weather_tmax_bot.polymarket_paper.mapping import (
    normalize_order_levels,
    parse_json_array,
    parse_temperature_bucket,
)
from weather_tmax_bot.polymarket_paper.models import MarketSnapshot, TemperatureMarket


class PolymarketPublicClient:
    def __init__(
        self,
        config: PaperTradingConfig,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()

    def fetch_paris_market(self, target_date_local: date) -> MarketSnapshot:
        response = self.session.get(
            f"{self.config.gamma_api_url}/events",
            params={
                "limit": 200,
                "offset": 0,
                "tag_slug": "weather",
                "active": "true",
                "closed": "false",
            },
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()
        events = response.json()
        event = self._select_event(events, target_date_local)
        markets = [
            market
            for raw_market in event.get("markets") or []
            if (market := self._parse_market(raw_market)) is not None
        ]
        if not markets:
            raise RuntimeError("Paris weather event has no parseable temperature markets")
        self._hydrate_books(markets)
        settlement_text = self._settlement_text(event)
        verified, notes = validate_settlement_text(settlement_text)
        return MarketSnapshot(
            event_title=str(event.get("title") or ""),
            event_slug=str(event.get("slug") or ""),
            target_date_local=target_date_local.isoformat(),
            settlement_text=settlement_text,
            settlement_verified=verified,
            settlement_notes=notes,
            markets=markets,
        )

    def fetch_resolved_token_prices(self, market_slugs: set[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for slug in sorted(slug for slug in market_slugs if slug):
            try:
                response = self.session.get(
                    f"{self.config.gamma_api_url}/markets",
                    params={"slug": slug},
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException:
                continue
            if not isinstance(payload, list) or not payload:
                continue
            market = payload[0]
            resolved = bool(market.get("closed")) or (
                str(market.get("umaResolutionStatus") or "").lower() == "resolved"
            )
            if not resolved:
                continue
            token_ids = [str(value) for value in parse_json_array(market.get("clobTokenIds"))]
            outcome_prices = parse_json_array(market.get("outcomePrices"))
            for token_id, raw_price in zip(token_ids, outcome_prices, strict=False):
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if price <= 0.001:
                    prices[token_id] = 0.0
                elif price >= 0.999:
                    prices[token_id] = 1.0
        return prices

    def _select_event(self, events: list[dict[str, Any]], target_date_local: date) -> dict[str, Any]:
        candidates = []
        for event in events:
            title = str(event.get("title") or "")
            normalized = title.lower()
            if "highest temperature" not in normalized or "paris" not in normalized:
                continue
            event_date = _extract_event_date(event)
            if event_date is not None and event_date != target_date_local:
                continue
            candidates.append(event)
        if not candidates:
            raise RuntimeError(
                f"No active Paris highest-temperature event found for {target_date_local.isoformat()}"
            )
        exact_dates = [
            event
            for event in candidates
            if _extract_event_date(event) == target_date_local
        ]
        return (exact_dates or candidates)[0]

    def _parse_market(self, raw: dict[str, Any]) -> TemperatureMarket | None:
        question = str(raw.get("question") or "")
        parsed_bucket = parse_temperature_bucket(question)
        token_ids = [str(value) for value in parse_json_array(raw.get("clobTokenIds"))]
        if parsed_bucket is None or len(token_ids) < 2:
            return None
        temperature_c, tail = parsed_bucket
        return TemperatureMarket(
            market_id=str(raw.get("conditionId") or raw.get("id") or question),
            question=question,
            slug=str(raw.get("slug") or ""),
            temperature_c=temperature_c,
            tail=tail,
            yes_token_id=token_ids[0],
            no_token_id=token_ids[1],
        )

    def _hydrate_books(self, markets: list[TemperatureMarket]) -> None:
        books: dict[str, dict[str, Any]] = {}
        for token_id in {
            token_id
            for market in markets
            for token_id in (market.yes_token_id, market.no_token_id)
        }:
            try:
                response = self.session.get(
                    f"{self.config.clob_api_url}/book",
                    params={"token_id": token_id},
                    timeout=self.config.request_timeout_seconds,
                )
                response.raise_for_status()
                books[token_id] = response.json()
            except requests.RequestException:
                books[token_id] = {}
        for market in markets:
            yes_book = books.get(market.yes_token_id, {})
            no_book = books.get(market.no_token_id, {})
            market.yes_asks = normalize_order_levels(yes_book.get("asks"), lowest_first=True)
            market.yes_bids = normalize_order_levels(yes_book.get("bids"), lowest_first=False)
            market.no_asks = normalize_order_levels(no_book.get("asks"), lowest_first=True)
            market.no_bids = normalize_order_levels(no_book.get("bids"), lowest_first=False)

    def _settlement_text(self, event: dict[str, Any]) -> str:
        parts = [
            event.get("title"),
            event.get("description"),
            event.get("resolutionSource"),
            event.get("rules"),
        ]
        for market in event.get("markets") or []:
            parts.extend(
                [
                    market.get("description"),
                    market.get("resolutionSource"),
                    market.get("rules"),
                ]
            )
        return "\n".join(str(part) for part in parts if part)


def validate_settlement_text(text: str) -> tuple[bool, list[str]]:
    normalized = text.lower()
    notes = []
    if "paris" not in normalized:
        notes.append("settlement text does not identify Paris")
    source_verified = any(
        marker in normalized
        for marker in ("lfpb", "le bourget", "weather underground", "wunderground")
    )
    if not source_verified:
        notes.append("station or weather source was not recognized")
    temperature_verified = "temperature" in normalized
    if not temperature_verified:
        notes.append("temperature settlement rule was not recognized")
    return not notes, notes


def _extract_event_date(event: dict[str, Any]) -> date | None:
    import re

    slug = str(event.get("slug") or "")
    slug_match = re.search(r"-(\d{4})$", slug)
    title = str(event.get("title") or "")
    title_match = re.search(r"\bon ([A-Za-z]+ \d{1,2})(?:, (\d{4}))?\??$", title)
    if title_match:
        year = title_match.group(2)
        if year is None and slug_match:
            year = slug_match.group(1)
        if year is not None:
            try:
                return datetime.strptime(
                    f"{title_match.group(1)}, {year}",
                    "%B %d, %Y",
                ).date()
            except ValueError:
                pass
    end_date = str(event.get("endDate") or "")
    if end_date:
        try:
            return datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None
