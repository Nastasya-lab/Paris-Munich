from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PARIS_TIMEZONE = ZoneInfo("Europe/Paris")


@dataclass(frozen=True, slots=True)
class ForecastSignal:
    forecast_id: str | None
    issue_time_utc: datetime
    target_date_local: date
    variant: str
    shadow_probabilities: dict[int, float]
    production_probabilities: dict[int, float]


def load_forecast_signal(
    path: Path,
    variant: str,
    *,
    expected_airport: str = "LFPB",
) -> ForecastSignal:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("airport") != expected_airport:
        raise ValueError(
            f"Polymarket paper trader expected {expected_airport} forecast, got {payload.get('airport')}"
        )
    variants = payload.get("forecast_variants") or {}
    variant_payload = variants.get(variant) or {}
    distribution = variant_payload.get("distribution") or {}
    if variant == "production_champion" and not distribution:
        distribution = payload.get("forecast") or payload
    shadow = _normalize_probabilities(distribution.get("probabilities_by_integer_c") or {})
    if not shadow:
        raise ValueError(f"Forecast variant {variant!r} has no probability distribution")
    production = _normalize_probabilities(
        ((payload.get("forecast") or payload).get("probabilities_by_integer_c") or {})
    )
    issue_time = datetime.fromisoformat(str(payload["issue_time_utc"]).replace("Z", "+00:00"))
    target_date = date.fromisoformat(str(payload["target_date_local"]))
    return ForecastSignal(
        forecast_id=payload.get("forecast_id") or f"{expected_airport}:{issue_time.isoformat()}",
        issue_time_utc=issue_time,
        target_date_local=target_date,
        variant=variant,
        shadow_probabilities=shadow,
        production_probabilities=production,
    )


def is_in_trading_window(
    signal: ForecastSignal,
    *,
    start_hour: int,
    end_hour: int,
    timezone: ZoneInfo = PARIS_TIMEZONE,
) -> bool:
    local = signal.issue_time_utc.astimezone(timezone)
    return signal.target_date_local == local.date() and start_hour <= local.hour < end_hour


def _normalize_probabilities(values: dict) -> dict[int, float]:
    probabilities: dict[int, float] = {}
    for raw_bin, raw_probability in values.items():
        try:
            bin_c = int(raw_bin)
            probability = max(0.0, float(raw_probability))
        except (TypeError, ValueError):
            continue
        probabilities[bin_c] = probabilities.get(bin_c, 0.0) + probability
    total = sum(probabilities.values())
    if total <= 0:
        return {}
    return {bin_c: probability / total for bin_c, probability in probabilities.items()}
