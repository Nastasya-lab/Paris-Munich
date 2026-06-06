from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


POLICIES = (
    "safe_blend_promoted",
    "seasonal_promoted",
    "ml_promoted",
    "conservative_phase",
    "dynamic_phase",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/reports/model_analysis_all_completed_days.csv")
    parser.add_argument("--output-json", default="data/reports/operational_ablation_report.json")
    parser.add_argument("--output-md", default="docs/operational_ablation_report.md")
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist; run completed-day model analysis first")
    rows = pd.read_csv(source)
    augmented = build_policy_rows(rows)
    report = build_report(augmented)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(format_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_json": args.output_json, "output_md": args.output_md}, indent=2))


def build_policy_rows(rows: pd.DataFrame) -> pd.DataFrame:
    policy_rows = []
    for (_, _), group in rows.groupby(["day", "forecast_id"]):
        for policy in POLICIES:
            selected = _pick_policy_row(group, policy)
            if selected is None:
                continue
            payload = selected.to_dict()
            payload["variant"] = f"policy_{policy}"
            payload["source_variant"] = selected["variant"]
            policy_rows.append(payload)
    if not policy_rows:
        return rows.copy()
    return pd.concat([rows, pd.DataFrame(policy_rows)], ignore_index=True)


def build_report(rows: pd.DataFrame) -> dict:
    sections = {
        "all": rows,
        "day_window_10_20": rows[(rows["local_hour"] >= 10.0) & (rows["local_hour"] < 20.0)],
        "main_heating_12_20": rows[(rows["local_hour"] >= 12.0) & (rows["local_hour"] < 20.0)],
        "late_16_20": rows[(rows["local_hour"] >= 16.0) & (rows["local_hour"] < 20.0)],
    }
    summary = {name: _summarize(frame) for name, frame in sections.items()}
    return {
        "status": "ok",
        "input_rows": int(len(rows)),
        "policies_evaluated": list(POLICIES),
        "summary": summary,
        "recommendations": _recommendations(summary),
    }


def _pick_policy_row(group: pd.DataFrame, policy: str) -> pd.Series | None:
    hour = float(group["local_hour"].iloc[0])

    def row(name: str) -> pd.Series | None:
        matches = group[group["variant"] == name]
        return None if matches.empty else matches.iloc[0]

    def first(*names: str) -> pd.Series | None:
        for name in names:
            selected = row(name)
            if selected is not None:
                return selected
        return None

    if policy == "safe_blend_promoted":
        return first("shadow_safe_blend", "production_champion")
    if policy == "seasonal_promoted":
        return first("shadow_seasonal_intraday", "production_champion")
    if policy == "ml_promoted":
        return first("shadow_intraday_ml", "production_champion")
    if policy == "conservative_phase":
        if hour < 12.0:
            return row("production_champion")
        if hour < 16.0:
            return first("shadow_safe_blend", "shadow_seasonal_intraday", "production_champion")
        return first("shadow_seasonal_intraday", "shadow_safe_blend", "production_champion")
    if policy == "dynamic_phase":
        if hour < 12.0:
            return row("production_champion")
        if hour < 16.0:
            return first("shadow_safe_blend", "shadow_seasonal_intraday", "production_champion")
        if hour < 20.0:
            return first("shadow_intraday_ml", "shadow_seasonal_intraday", "shadow_safe_blend", "production_champion")
        return first("shadow_seasonal_intraday", "shadow_safe_blend", "shadow_intraday_ml", "production_champion")
    raise ValueError(f"unknown policy: {policy}")


def _summarize(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    summary = (
        frame.groupby("variant")
        .agg(
            rows=("mae", "size"),
            days=("day", "nunique"),
            forecasts=("forecast_id", "nunique"),
            mae=("mae", "mean"),
            median_mae=("mae", "median"),
            bias=("bias", "mean"),
            p_actual=("p_actual", "mean"),
            bin_ok_rate=("bin_ok", "mean"),
        )
        .reset_index()
        .sort_values(["mae", "median_mae"])
    )
    return [
        {
            "variant": str(row.variant),
            "rows": int(row.rows),
            "days": int(row.days),
            "forecasts": int(row.forecasts),
            "mae": float(row.mae),
            "median_mae": float(row.median_mae),
            "bias": float(row.bias),
            "p_actual": float(row.p_actual),
            "bin_ok_rate": float(row.bin_ok_rate),
        }
        for row in summary.itertuples(index=False)
    ]


def _recommendations(summary: dict) -> list[str]:
    recommendations = []
    all_rows = summary.get("all", [])
    window = summary.get("day_window_10_20", [])
    late = summary.get("late_16_20", [])
    best_all = all_rows[0]["variant"] if all_rows else None
    best_window = window[0]["variant"] if window else None
    best_late = late[0]["variant"] if late else None
    if best_all == "policy_dynamic_phase":
        recommendations.append("Keep production_champion, but log phase_arbitrated_shadow_v1 as the leading promotion candidate.")
    if best_window in {"shadow_safe_blend", "policy_dynamic_phase", "policy_conservative_phase"}:
        recommendations.append("Increase safe-blend attention in the 12-16 local window; it is the most useful daytime improvement.")
    if best_late in {"shadow_intraday_ml", "policy_dynamic_phase"}:
        recommendations.append("Keep ML shadow as a late-day signal, but do not promote it globally because morning behavior is unstable.")
    recommendations.append("Discard base_prior as a standalone operational candidate; keep it only as the NWP prior component.")
    return recommendations


def format_markdown(report: dict) -> str:
    lines = [
        "# Operational Ablation Report",
        "",
        "This report compares logged model variants and phase-selection policies on completed operational days.",
        "",
        "## Recommendations",
        "",
        *[f"- {item}" for item in report.get("recommendations", [])],
    ]
    for section, rows in report.get("summary", {}).items():
        lines.extend(["", f"## {section}", "", "| variant | rows | days | MAE | median MAE | bias | P(actual bin) | bin ok |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['variant']} | {row['rows']} | {row['days']} | {row['mae']:.3f} | "
                f"{row['median_mae']:.3f} | {row['bias']:+.3f} | {row['p_actual']:.3f} | {row['bin_ok_rate']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
