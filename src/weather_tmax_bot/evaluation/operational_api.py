from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def operational_monitoring_payload(root: str | Path = ".") -> dict:
    root = Path(root)
    return {
        "by_model": _read_table(root / "data/reports/operational_by_model.parquet"),
        "source_mismatch": _read_table(root / "data/reports/operational_source_mismatch.parquet"),
        "availability": _read_table(root / "data/reports/operational_availability.parquet"),
        "acceptance": _read_table(root / "data/reports/operational_acceptance.parquet"),
        "forecast_inventory": _read_table(root / "data/reports/operational_forecast_inventory.parquet"),
        "pending_forecasts": _read_table(root / "data/reports/operational_pending_forecasts.parquet"),
        "shadow_promotion_gate": _read_json(root / "data/reports/shadow_promotion_gate.json"),
    }


def _read_table(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_parquet(path)
    if df.empty:
        return []
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
