from __future__ import annotations

from weather_tmax_bot.polymarket_paper.models import ExecutionQuote, OrderLevel


def quote_buy(levels: list[OrderLevel], target_usd: float) -> ExecutionQuote:
    if target_usd <= 0:
        return ExecutionQuote(None, 0.0, 0.0, 0.0, 0)
    remaining_usd = target_usd
    filled_usd = 0.0
    filled_shares = 0.0
    levels_used = 0
    for level in levels:
        available_usd = level.price * level.shares
        take_usd = min(remaining_usd, available_usd)
        if take_usd <= 0:
            continue
        filled_usd += take_usd
        filled_shares += take_usd / level.price
        remaining_usd -= take_usd
        levels_used += 1
        if remaining_usd <= 1e-9:
            break
    average = filled_usd / filled_shares if filled_shares > 0 else None
    return ExecutionQuote(
        average_price=average,
        shares=filled_shares,
        notional_usd=filled_usd,
        fill_ratio=filled_usd / target_usd,
        levels_used=levels_used,
    )


def quote_sell(levels: list[OrderLevel], target_shares: float) -> ExecutionQuote:
    if target_shares <= 0:
        return ExecutionQuote(None, 0.0, 0.0, 0.0, 0)
    remaining_shares = target_shares
    proceeds = 0.0
    filled_shares = 0.0
    levels_used = 0
    for level in levels:
        take_shares = min(remaining_shares, level.shares)
        if take_shares <= 0:
            continue
        filled_shares += take_shares
        proceeds += take_shares * level.price
        remaining_shares -= take_shares
        levels_used += 1
        if remaining_shares <= 1e-9:
            break
    average = proceeds / filled_shares if filled_shares > 0 else None
    return ExecutionQuote(
        average_price=average,
        shares=filled_shares,
        notional_usd=proceeds,
        fill_ratio=filled_shares / target_shares,
        levels_used=levels_used,
    )

