from __future__ import annotations

from dataclasses import dataclass

from weather_tmax_bot.utils.validation import SourceIdentityError


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    source_name: str
    source_version: str = "unknown"
    source_url_or_reference: str | None = None


def require_same_source(actual: str, expected: str, context: str = "") -> None:
    if actual != expected:
        suffix = f" for {context}" if context else ""
        raise SourceIdentityError(f"source mismatch{suffix}: expected {expected}, got {actual}")
