from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path


def operational_prediction_payload(
    *,
    airport: str,
    target_date_local: date,
    issue_time_utc: datetime,
    result: dict,
) -> dict:
    payload = result["distribution"].to_payload()
    payload.update(
        {
            "airport": airport,
            "target_date_local": target_date_local.isoformat(),
            "issue_time_utc": issue_time_utc.isoformat(),
            "model_version": result["metadata"]["model_version"],
            "forecast_id": result["forecast_id"],
            "warnings": result["warnings"],
            "data_lineage": result["data_lineage"],
            "data_freshness": result["feature_snapshot"].get("freshness", {}),
            "extrapolation": result["feature_snapshot"].get("extrapolation", {}),
            "source_compatibility": result["feature_snapshot"].get("source_compatibility", {}),
            "forecast_components": result["feature_snapshot"].get("forecast_components", {}),
            "forecast_variants": result["feature_snapshot"].get("forecast_variants", {}),
            "latest_metar_record": result["feature_snapshot"].get("latest_metar_record"),
            "forecast_quality": result["forecast_quality"],
            "forecast_acceptance": result["forecast_acceptance"],
            "refresh_summary": result.get("refresh_summary"),
        }
    )
    return payload


def write_operational_prediction_report(payload: dict, path: str | Path = "data/reports/latest_operational_prediction.json") -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output
