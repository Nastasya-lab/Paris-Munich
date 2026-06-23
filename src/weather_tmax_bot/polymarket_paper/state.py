from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weather_tmax_bot.polymarket_paper.models import PaperPosition, TradeEvent


@dataclass(slots=True)
class PaperState:
    version: int
    start_balance_usd: float
    cash_balance_usd: float
    realized_pnl_usd: float
    positions: list[PaperPosition] = field(default_factory=list)
    events: list[TradeEvent] = field(default_factory=list)
    last_forecast_id: str | None = None
    updated_at_utc: str | None = None

    @property
    def equity_basis_usd(self) -> float:
        return self.cash_balance_usd + sum(position.size_usd for position in self.positions)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "positions": [position.to_dict() for position in self.positions],
            "events": [event.to_dict() for event in self.events],
        }


class PaperStateStore:
    def __init__(self, path: Path, start_balance_usd: float) -> None:
        self.path = path
        self.start_balance_usd = start_balance_usd

    def load(self) -> PaperState:
        if not self.path.exists():
            return PaperState(
                version=1,
                start_balance_usd=self.start_balance_usd,
                cash_balance_usd=self.start_balance_usd,
                realized_pnl_usd=0.0,
            )
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return PaperState(
            version=int(payload.get("version", 1)),
            start_balance_usd=float(payload.get("start_balance_usd", self.start_balance_usd)),
            cash_balance_usd=float(payload.get("cash_balance_usd", self.start_balance_usd)),
            realized_pnl_usd=float(payload.get("realized_pnl_usd", 0.0)),
            positions=[PaperPosition(**item) for item in payload.get("positions", [])],
            events=[TradeEvent(**item) for item in payload.get("events", [])],
            last_forecast_id=payload.get("last_forecast_id"),
            updated_at_utc=payload.get("updated_at_utc"),
        )

    def save(self, state: PaperState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.updated_at_utc = datetime.now(UTC).isoformat()
        content = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)


def append_decision_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

