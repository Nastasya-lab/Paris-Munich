from __future__ import annotations

from pathlib import Path

import pandas as pd


def audit_training_dataset(dataset: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    issue = pd.to_datetime(dataset["issue_time_utc"], utc=True)
    checks: dict[str, int] = {}
    for col in ("max_metar_knowledge_time_utc", "max_nwp_knowledge_time_utc", "latest_metar_time_utc"):
        if col in dataset.columns:
            values = pd.to_datetime(dataset[col], utc=True, errors="coerce")
            checks[col] = int(((values.notna()) & (values > issue)).sum())
    target_cols = sorted(set(dataset.columns).intersection({"target", "target_tmax_c"}))
    checks["forbidden_target_feature_columns"] = len(target_cols)
    checks["rows"] = len(dataset)
    passed = all(value == 0 for key, value in checks.items() if key != "rows")
    report = pd.DataFrame([{"check": key, "violations": value} for key, value in checks.items()])
    return report, passed


def audit_training_dataset_file(
    dataset_path: str | Path = "data/processed/training_dataset.parquet",
    report_path: str | Path | None = "data/reports/leakage_audit.parquet",
) -> tuple[pd.DataFrame, bool]:
    dataset = pd.read_parquet(dataset_path)
    report, passed = audit_training_dataset(dataset)
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report.to_parquet(path, index=False)
    return report, passed
