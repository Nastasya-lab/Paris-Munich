from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import bias, mae, rmse


def main() -> None:
    nwp = pd.read_parquet("data/processed/nwp_features_open_meteo.parquet")
    target = pd.read_parquet("data/processed/daily_target.parquet")
    target = target[target["quality_flags"] == "ok"].copy()
    df = nwp[nwp["nwp_missing"] == False].copy()  # noqa: E712 - pandas boolean filtering.
    df = df.merge(
        target[["target_date_local", "tmax_c"]],
        on="target_date_local",
        how="inner",
    )
    df["model_tmax_c"] = pd.to_numeric(df["model_tmax_c"], errors="coerce")
    df["tmax_c"] = pd.to_numeric(df["tmax_c"], errors="coerce")
    df = df[df["model_tmax_c"].notna() & df["tmax_c"].notna()].copy()
    df["error_c"] = df["model_tmax_c"] - df["tmax_c"]
    df["month"] = pd.to_datetime(df["target_date_local"]).dt.month

    summary = pd.DataFrame(
        [
            _summary_row(df, ["all"]),
            *[_summary_row(group, ["issue_hour_utc", int(hour)]) for hour, group in df.groupby("issue_hour_utc")],
            *[_summary_row(group, ["month", int(month)]) for month, group in df.groupby("month")],
        ]
    )
    summary["value"] = summary["value"].astype(str)
    write_parquet(df, "data/reports/nwp_backfill_scored_rows.parquet")
    write_parquet(summary, "data/reports/nwp_backfill_summary.parquet")
    _write_doc(df, summary)
    print(f"Wrote NWP backfill analysis for {len(df)} scored rows")


def _summary_row(df: pd.DataFrame, label: list) -> dict:
    return {
        "group": label[0],
        "value": label[1] if len(label) > 1 else "all",
        "rows": len(df),
        "first_target_date_local": df["target_date_local"].min() if not df.empty else None,
        "latest_target_date_local": df["target_date_local"].max() if not df.empty else None,
        "mae_model_tmax": mae(df["tmax_c"], df["model_tmax_c"]) if not df.empty else None,
        "rmse_model_tmax": rmse(df["tmax_c"], df["model_tmax_c"]) if not df.empty else None,
        "bias_model_tmax": bias(df["tmax_c"], df["model_tmax_c"]) if not df.empty else None,
    }


def _write_doc(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    overall = summary[summary["group"] == "all"].iloc[0].to_dict() if not summary.empty else {}
    issue = summary[summary["group"] == "issue_hour_utc"].copy()
    month = summary[summary["group"] == "month"].copy()
    lines = [
        "# NWP backfill analysis",
        "",
        "This report scores Open-Meteo Single Runs ICON-D2 `model_tmax_c` against DWD daily Tmax truth.",
        "It is a source-quality analysis, not yet a promoted production model comparison.",
        "",
        "## Coverage",
        "",
        f"- scored rows: `{len(df)}`",
        f"- first target date: `{overall.get('first_target_date_local')}`",
        f"- latest target date: `{overall.get('latest_target_date_local')}`",
        f"- source IDs: `{', '.join(sorted(df['latest_nwp_source_id'].dropna().astype(str).unique().tolist())) if not df.empty else 'none'}`",
        "",
        "## Overall raw NWP error",
        "",
        f"- MAE: `{overall.get('mae_model_tmax')}`",
        f"- RMSE: `{overall.get('rmse_model_tmax')}`",
        f"- bias: `{overall.get('bias_model_tmax')}`",
        "",
        "## By issue hour UTC",
        "",
        _markdown_table(issue),
        "",
        "## By month",
        "",
        _markdown_table(month),
        "",
        "## Interpretation",
        "",
        "NWP rows are now available for NWP-aware experiments, but the active production model is not automatically promoted.",
        "Promotion should wait for a direct probabilistic comparison and at least one clean validation slice where NWP is present in both train and test.",
    ]
    Path("docs/nwp_backfill_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = ["value", "rows", "mae_model_tmax", "rmse_model_tmax", "bias_model_tmax"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df[cols].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
