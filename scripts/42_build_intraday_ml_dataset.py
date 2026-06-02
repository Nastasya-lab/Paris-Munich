from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.models.intraday_ml import prepare_intraday_ml_dataset


def main() -> None:
    source_path = Path("data/processed/training_dataset.parquet")
    output_path = Path("data/processed/intraday_ml_dataset.parquet")
    report_path = Path("data/reports/intraday_ml_dataset_audit.json")
    doc_path = Path("docs/intraday_ml_dataset_audit.md")
    dataset = prepare_intraday_ml_dataset(pd.read_parquet(source_path))
    write_parquet(dataset, output_path)
    audit = _audit(dataset)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    doc_path.write_text(_doc(audit), encoding="utf-8")
    print(json.dumps(audit, indent=2, default=str))


def _audit(dataset: pd.DataFrame) -> dict:
    dates = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    nwp_available = (~dataset.get("nwp_missing", True).fillna(True).astype(bool)) & dataset.get("model_tmax_c", pd.Series(index=dataset.index)).notna()
    taf_available = ~dataset.get("taf_missing", True).fillna(True).astype(bool)
    by_hour = (
        dataset.assign(nwp_available=nwp_available, taf_available=taf_available)
        .groupby("issue_hour_utc", dropna=False)
        .agg(
            rows=("issue_hour_utc", "size"),
            distinct_days=("target_date_local", "nunique"),
            mean_remaining_upside_c=("remaining_upside_c", "mean"),
            peak_already_passed_rate=("peak_already_passed", "mean"),
            upside_ge_1c_rate=("upside_ge_1c", "mean"),
            upside_ge_2c_rate=("upside_ge_2c", "mean"),
            upside_ge_3c_rate=("upside_ge_3c", "mean"),
            nwp_available_rate=("nwp_available", "mean"),
            taf_available_rate=("taf_available", "mean"),
        )
        .reset_index()
    )
    return {
        "design": "One row per target local day and issue time. Features remain restricted to as-of knowledge; labels are appended only after the local day closes.",
        "source": "data/processed/training_dataset.parquet",
        "output": "data/processed/intraday_ml_dataset.parquet",
        "rows": len(dataset),
        "distinct_target_days": int(dataset["target_date_local"].nunique()),
        "period": [str(min(dates)), str(max(dates))],
        "nwp_available_rows": int(nwp_available.sum()),
        "nwp_available_rate": float(nwp_available.mean()),
        "taf_available_rows": int(taf_available.sum()),
        "taf_available_rate": float(taf_available.mean()),
        "leakage_check_passed_rate": float(dataset.get("leakage_check_passed", False).fillna(False).astype(bool).mean()),
        "label_rates": {
            "peak_already_passed": float(dataset["peak_already_passed"].mean()),
            "upside_ge_1c": float(dataset["upside_ge_1c"].mean()),
            "upside_ge_2c": float(dataset["upside_ge_2c"].mean()),
            "upside_ge_3c": float(dataset["upside_ge_3c"].mean()),
        },
        "by_issue_hour": json.loads(by_hour.to_json(orient="records")),
        "limitations": [
            "Historical IEM TAF archive is currently empty, so TAF features gracefully degrade to missing flags during training.",
            "Forecast-as-issued ICON-D2 overlap starts in late May 2025; the core model is trained to remain usable when NWP is missing.",
            "Historical rows use scheduled 00/03/06/09/12/15/18 UTC issues. Railway METAR-event timing remains a forward-shadow concern.",
        ],
    }


def _doc(audit: dict) -> str:
    lines = [
        "# Intraday ML dataset audit",
        "",
        audit["design"],
        "",
        f"- rows: `{audit['rows']}`",
        f"- distinct target days: `{audit['distinct_target_days']}`",
        f"- period: `{audit['period'][0]}` to `{audit['period'][1]}`",
        f"- NWP available rows: `{audit['nwp_available_rows']}` (`{audit['nwp_available_rate']:.1%}`)",
        f"- TAF available rows: `{audit['taf_available_rows']}` (`{audit['taf_available_rate']:.1%}`)",
        f"- leakage checks passed: `{audit['leakage_check_passed_rate']:.1%}`",
        "",
        "## Labels",
        "",
    ]
    lines.extend(f"- `{key}`: `{value:.1%}`" for key, value in audit["label_rates"].items())
    lines.extend(["", "## By issue hour", "", _table(audit["by_issue_hour"]), "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in audit["limitations"])
    lines.append("")
    return "\n".join(lines)


def _table(rows: list[dict]) -> str:
    if not rows:
        return "No rows."
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[key]) for key in keys) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
