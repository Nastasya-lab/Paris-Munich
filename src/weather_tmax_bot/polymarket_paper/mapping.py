from __future__ import annotations

import re
from typing import Any

from weather_tmax_bot.polymarket_paper.models import OrderLevel


TEMPERATURE_RE = re.compile(r"(?<!\d)(-?\d+)\s*(?:°|º)?\s*C\b", re.IGNORECASE)


def parse_temperature_bucket(question: str) -> tuple[int, str] | None:
    match = TEMPERATURE_RE.search(question)
    if not match:
        return None
    temperature_c = int(match.group(1))
    normalized = question.lower()
    if "or higher" in normalized or "or above" in normalized:
        tail = "or_higher"
    elif "or lower" in normalized or "or below" in normalized:
        tail = "or_lower"
    else:
        tail = "exact"
    return temperature_c, tail


def probability_for_bucket(
    probabilities: dict[int, float],
    temperature_c: int,
    tail: str,
) -> float:
    if tail == "or_higher":
        return sum(value for bin_c, value in probabilities.items() if bin_c >= temperature_c)
    if tail == "or_lower":
        return sum(value for bin_c, value in probabilities.items() if bin_c <= temperature_c)
    return probabilities.get(temperature_c, 0.0)


def parse_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        import json

        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    return []


def normalize_order_levels(
    values: Any,
    *,
    lowest_first: bool,
) -> list[OrderLevel]:
    levels: list[OrderLevel] = []
    for item in values or []:
        try:
            if isinstance(item, dict):
                price = float(item["price"])
                shares = float(item["size"])
            else:
                price = float(item[0])
                shares = float(item[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if 0.0 < price < 1.0 and shares > 0:
            levels.append(OrderLevel(price=price, shares=shares))
    levels.sort(key=lambda level: level.price, reverse=not lowest_first)
    return levels

