from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_operational_monitoring_tables(
    monitoring_path: str | Path = "data/reports/forecast_monitoring.parquet",
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    outcome_status_path: str | Path = "data/reports/forecast_outcome_status.parquet",
    output_dir: str | Path = "data/reports",
) -> dict[str, pd.DataFrame]:
    path = Path(monitoring_path)
    inventory = forecast_log_inventory(forecast_log_path)
    pending = pending_forecast_summary(outcome_status_path)
    if not path.exists():
        tables = {
            "by_model": pd.DataFrame(),
            "source_mismatch": pd.DataFrame(),
            "availability": pd.DataFrame(),
            "acceptance": pd.DataFrame(),
            "forecast_inventory": inventory,
            "pending_forecasts": pending,
        }
        _write_tables(tables, output_dir)
        return tables
    monitoring = pd.read_parquet(path)
    if monitoring.empty:
        tables = {
            "by_model": pd.DataFrame(),
            "source_mismatch": pd.DataFrame(),
            "availability": pd.DataFrame(),
            "acceptance": pd.DataFrame(),
            "forecast_inventory": inventory,
            "pending_forecasts": pending,
        }
        _write_tables(tables, output_dir)
        return tables
    tables = {
        "by_model": summarize_by_model(monitoring),
        "source_mismatch": summarize_source_mismatch(monitoring),
        "availability": summarize_availability(monitoring),
        "acceptance": summarize_acceptance(monitoring),
        "forecast_inventory": inventory,
        "pending_forecasts": pending,
    }
    _write_tables(tables, output_dir)
    return tables


def forecast_log_inventory(forecast_log_path: str | Path = "data/logs/forecast_log.jsonl") -> pd.DataFrame:
    path = Path(forecast_log_path)
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metadata = record.get("raw_input_metadata", {}) or {}
        quality = metadata.get("forecast_quality", {}) or {}
        acceptance = metadata.get("forecast_acceptance", {}) or {}
        rows.append(
            {
                "forecast_id": record.get("forecast_id"),
                "model_version": record.get("model_version"),
                "airport": record.get("airport"),
                "target_date_local": record.get("target_date_local"),
                "issue_time_utc": record.get("issue_time_utc"),
                "latest_metar_source_id": metadata.get("latest_metar_source_id"),
                "latest_taf_source_id": metadata.get("latest_taf_source_id"),
                "latest_nwp_source_id": metadata.get("latest_nwp_source_id"),
                "metar_missing": bool(metadata.get("metar_missing", False)),
                "taf_missing": bool(metadata.get("taf_missing", False)),
                "nwp_missing": bool(metadata.get("nwp_missing", False)),
                "forecast_quality_status": quality.get("status", "unknown"),
                "forecast_accepted": _acceptance_label(acceptance.get("accepted")),
                "forecast_acceptance_blocking_reasons": _join_list(acceptance.get("blocking_reasons")),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["model_version", "airport"], dropna=False)
        .agg(
            logged_forecasts=("forecast_id", "count"),
            first_issue_time_utc=("issue_time_utc", "min"),
            latest_issue_time_utc=("issue_time_utc", "max"),
            metar_missing_rate=("metar_missing", "mean"),
            taf_missing_rate=("taf_missing", "mean"),
            nwp_missing_rate=("nwp_missing", "mean"),
            accepted_rate=("forecast_accepted", lambda x: float((x == "accepted").mean())),
            rejected_rate=("forecast_accepted", lambda x: float((x == "rejected").mean())),
            unknown_acceptance_rate=("forecast_accepted", lambda x: float((x == "unknown").mean())),
            metar_sources=("latest_metar_source_id", _unique_join),
            taf_sources=("latest_taf_source_id", _unique_join),
            nwp_sources=("latest_nwp_source_id", _unique_join),
            quality_statuses=("forecast_quality_status", _unique_join),
            acceptance_blocking_reasons=("forecast_acceptance_blocking_reasons", _unique_join),
        )
        .reset_index()
    )
    return grouped


def pending_forecast_summary(outcome_status_path: str | Path = "data/reports/forecast_outcome_status.parquet") -> pd.DataFrame:
    path = Path(outcome_status_path)
    if not path.exists():
        return pd.DataFrame()
    status = pd.read_parquet(path)
    if status.empty:
        return pd.DataFrame()
    if "forecast_accepted" not in status.columns:
        status["forecast_accepted"] = "unknown"
    else:
        status["forecast_accepted"] = status["forecast_accepted"].map(_acceptance_label)
    if "forecast_quality_status" not in status.columns:
        status["forecast_quality_status"] = "unknown"
    else:
        status["forecast_quality_status"] = status["forecast_quality_status"].fillna("unknown")
    if "forecast_acceptance_blocking_reasons" not in status.columns:
        status["forecast_acceptance_blocking_reasons"] = None
    return (
        status.groupby(["outcome_status", "model_version", "forecast_accepted", "forecast_quality_status"], dropna=False)
        .agg(
            forecasts=("forecast_id", "count"),
            first_target_date_local=("target_date_local", "min"),
            latest_target_date_local=("target_date_local", "max"),
            acceptance_blocking_reasons=("forecast_acceptance_blocking_reasons", _unique_join),
        )
        .reset_index()
    )


def _write_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(out / f"operational_{name}.parquet", index=False)


def summarize_by_model(monitoring: pd.DataFrame) -> pd.DataFrame:
    return (
        monitoring.groupby("model_version", dropna=False)
        .agg(
            forecasts=("forecast_id", "count"),
            mae_expected=("error_expected_c", lambda x: float(x.abs().mean())),
            bias_expected=("error_expected_c", "mean"),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
            brier_ge_20=("brier_ge_20", "mean"),
            brier_ge_25=("brier_ge_25", "mean"),
            brier_ge_30=("brier_ge_30", "mean"),
        )
        .reset_index()
    )


def summarize_source_mismatch(monitoring: pd.DataFrame) -> pd.DataFrame:
    df = monitoring.copy()
    for col in ("metar_source_mismatch", "taf_source_mismatch"):
        if col not in df.columns:
            df[col] = False
    df["any_source_mismatch"] = df["metar_source_mismatch"].fillna(False) | df["taf_source_mismatch"].fillna(False)
    return (
        df.groupby(["model_version", "any_source_mismatch"], dropna=False)
        .agg(
            forecasts=("forecast_id", "count"),
            mae_expected=("error_expected_c", lambda x: float(x.abs().mean())),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
    )


def summarize_availability(monitoring: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, col in (("metar", "metar_missing"), ("taf", "taf_missing"), ("nwp", "nwp_missing")):
        if col not in monitoring.columns:
            missing_rate = None
            rows_count = len(monitoring)
        else:
            missing = monitoring[col].fillna(False).astype(bool)
            missing_rate = float(missing.mean()) if len(missing) else None
            rows_count = len(missing)
        rows.append({"source": source, "forecasts": rows_count, "missing_rate": missing_rate})
    return pd.DataFrame(rows)


def summarize_acceptance(monitoring: pd.DataFrame) -> pd.DataFrame:
    df = monitoring.copy()
    if "forecast_accepted" not in df.columns:
        df["forecast_accepted"] = "unknown"
    else:
        df["forecast_accepted"] = df["forecast_accepted"].map(_acceptance_label)
    return (
        df.groupby(["model_version", "forecast_accepted"], dropna=False)
        .agg(
            forecasts=("forecast_id", "count"),
            mae_expected=("error_expected_c", lambda x: float(x.abs().mean())),
            bias_expected=("error_expected_c", "mean"),
            mean_nll=("nll", "mean"),
            mean_crps=("crps", "mean"),
        )
        .reset_index()
    )


def _acceptance_label(value) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"accepted", "true", "1"}:
            return "accepted"
        if normalized in {"rejected", "false", "0"}:
            return "rejected"
        return "unknown"
    return "accepted" if bool(value) else "rejected"


def _join_list(value) -> str | None:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    if value in (None, ""):
        return None
    return str(value)


def _unique_join(values: pd.Series) -> str:
    cleaned = sorted({str(value) for value in values.dropna() if str(value) not in ("", "None", "nan", "NaT")})
    return ", ".join(cleaned)
