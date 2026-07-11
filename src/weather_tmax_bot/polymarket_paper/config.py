from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class PaperTradingConfig:
    enabled: bool
    signal_variant: str
    state_path: Path
    decision_log_path: Path
    start_balance_usd: float
    calibration_buffer: float
    cost_buffer: float
    min_effective_edge: float
    close_effective_edge: float
    max_position_fraction: float
    max_daily_exposure_fraction: float
    max_positions: int
    min_contract_price: float
    max_contract_price: float
    min_fill_ratio: float
    allow_yes_positions: bool
    signal_confirmations_required: int
    local_hour_start: int
    local_hour_end: int
    require_verified_settlement: bool
    gamma_api_url: str
    clob_api_url: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls, prefix: str = "LFPB") -> "PaperTradingConfig":
        normalized_prefix = prefix.strip().upper()

        def value(name: str, default: str) -> str:
            return os.getenv(f"{normalized_prefix}_POLYMARKET_{name}", default)

        def enabled(name: str, default: bool) -> bool:
            return _env_bool(f"{normalized_prefix}_POLYMARKET_{name}", default)

        return cls(
            enabled=enabled("PAPER_ENABLED", True),
            signal_variant=value("SIGNAL_VARIANT", "production_champion"),
            state_path=Path(
                value("STATE_PATH", f"data/polymarket/{normalized_prefix.lower()}_paper_state.json")
            ),
            decision_log_path=Path(
                value(
                    "DECISION_LOG_PATH",
                    f"data/polymarket/{normalized_prefix.lower()}_paper_decisions.jsonl",
                )
            ),
            start_balance_usd=float(value("START_BALANCE_USD", "1000")),
            calibration_buffer=float(value("CALIBRATION_BUFFER", "0.05")),
            cost_buffer=float(value("COST_BUFFER", "0.01")),
            min_effective_edge=float(value("MIN_EFFECTIVE_EDGE", "0.08")),
            close_effective_edge=float(value("CLOSE_EFFECTIVE_EDGE", "0.02")),
            max_position_fraction=float(value("MAX_POSITION_PCT", "0.01")),
            max_daily_exposure_fraction=float(value("MAX_DAILY_EXPOSURE_PCT", "0.02")),
            max_positions=int(value("MAX_POSITIONS", "5")),
            min_contract_price=float(value("MIN_CONTRACT_PRICE", "0.02")),
            max_contract_price=float(value("MAX_CONTRACT_PRICE", "0.95")),
            min_fill_ratio=float(value("MIN_FILL_RATIO", "0.98")),
            allow_yes_positions=enabled("ALLOW_YES", False),
            signal_confirmations_required=max(
                1,
                int(value("SIGNAL_CONFIRMATIONS_REQUIRED", "2")),
            ),
            local_hour_start=int(value("LOCAL_HOUR_START", "10")),
            local_hour_end=int(value("LOCAL_HOUR_END", "17")),
            require_verified_settlement=enabled("REQUIRE_VERIFIED_SETTLEMENT", False),
            gamma_api_url=value("GAMMA_URL", "https://gamma-api.polymarket.com"),
            clob_api_url=value("CLOB_URL", "https://clob.polymarket.com"),
            request_timeout_seconds=float(value("REQUEST_TIMEOUT_SECONDS", "15")),
        )
