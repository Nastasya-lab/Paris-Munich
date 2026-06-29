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
    local_hour_start: int
    local_hour_end: int
    require_verified_settlement: bool
    gamma_api_url: str
    clob_api_url: str
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "PaperTradingConfig":
        return cls(
            enabled=_env_bool("LFPB_POLYMARKET_PAPER_ENABLED", True),
            signal_variant=os.getenv("LFPB_POLYMARKET_SIGNAL_VARIANT", "shadow_unimodal_pmf"),
            state_path=Path(
                os.getenv(
                    "LFPB_POLYMARKET_STATE_PATH",
                    "data/polymarket/lfpb_paper_state.json",
                )
            ),
            decision_log_path=Path(
                os.getenv(
                    "LFPB_POLYMARKET_DECISION_LOG_PATH",
                    "data/polymarket/lfpb_paper_decisions.jsonl",
                )
            ),
            start_balance_usd=float(os.getenv("LFPB_POLYMARKET_START_BALANCE_USD", "1000")),
            calibration_buffer=float(os.getenv("LFPB_POLYMARKET_CALIBRATION_BUFFER", "0.05")),
            cost_buffer=float(os.getenv("LFPB_POLYMARKET_COST_BUFFER", "0.01")),
            min_effective_edge=float(os.getenv("LFPB_POLYMARKET_MIN_EFFECTIVE_EDGE", "0.08")),
            close_effective_edge=float(os.getenv("LFPB_POLYMARKET_CLOSE_EFFECTIVE_EDGE", "0.02")),
            max_position_fraction=float(os.getenv("LFPB_POLYMARKET_MAX_POSITION_PCT", "0.01")),
            max_daily_exposure_fraction=float(
                os.getenv("LFPB_POLYMARKET_MAX_DAILY_EXPOSURE_PCT", "0.02")
            ),
            max_positions=int(os.getenv("LFPB_POLYMARKET_MAX_POSITIONS", "2")),
            min_contract_price=float(os.getenv("LFPB_POLYMARKET_MIN_CONTRACT_PRICE", "0.02")),
            max_contract_price=float(os.getenv("LFPB_POLYMARKET_MAX_CONTRACT_PRICE", "0.95")),
            min_fill_ratio=float(os.getenv("LFPB_POLYMARKET_MIN_FILL_RATIO", "0.98")),
            local_hour_start=int(os.getenv("LFPB_POLYMARKET_LOCAL_HOUR_START", "10")),
            local_hour_end=int(os.getenv("LFPB_POLYMARKET_LOCAL_HOUR_END", "17")),
            require_verified_settlement=_env_bool(
                "LFPB_POLYMARKET_REQUIRE_VERIFIED_SETTLEMENT",
                False,
            ),
            gamma_api_url=os.getenv(
                "LFPB_POLYMARKET_GAMMA_URL",
                "https://gamma-api.polymarket.com",
            ),
            clob_api_url=os.getenv(
                "LFPB_POLYMARKET_CLOB_URL",
                "https://clob.polymarket.com",
            ),
            request_timeout_seconds=float(
                os.getenv("LFPB_POLYMARKET_REQUEST_TIMEOUT_SECONDS", "15")
            ),
        )
