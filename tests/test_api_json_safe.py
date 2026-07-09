from __future__ import annotations

import json
import math

from weather_tmax_bot.bot.api import _json_safe


def test_json_safe_replaces_non_finite_numbers() -> None:
    payload = {
        "ok": 1.5,
        "nan": float("nan"),
        "nested": [{"inf": float("inf"), "neg_inf": -float("inf")}],
    }

    cleaned = _json_safe(payload)

    assert cleaned == {"ok": 1.5, "nan": None, "nested": [{"inf": None, "neg_inf": None}]}
    json.dumps(cleaned, allow_nan=False)


def test_json_safe_preserves_booleans() -> None:
    assert _json_safe({"value": True, "number": 1}) == {"value": True, "number": 1}
    assert math.isfinite(_json_safe({"number": 1})["number"])
