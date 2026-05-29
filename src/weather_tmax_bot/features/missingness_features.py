from __future__ import annotations


def missingness_flags(features: dict) -> dict:
    return {f"{key}_missing_flag": value is None for key, value in features.items()}
