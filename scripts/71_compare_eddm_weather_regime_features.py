from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.models.intraday_ml import DEFAULT_INTRADAY_ML_FEATURES
from weather_tmax_bot.models.weather_regime import REGIME_LABELS, add_weather_regime_columns


AIRPORT = "EDDM"
REGIME_FEATURE_COLUMNS = [
    *[f"weather_regime_{label}_score" for label in REGIME_LABELS],
    *[f"weather_regime_is_{label}" for label in REGIME_LABELS],
]

_ENHANCED_COMPARE = None


def main() -> None:
    dataset = _load_or_build_dataset()
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    usable = dataset[dataset["target_date_local"] <= pd.to_datetime("2025-12-30").date()].copy()
    base_features = list(DEFAULT_INTRADAY_ML_FEATURES)
    candidate_features = base_features + REGIME_FEATURE_COLUMNS
    enhanced_compare = _load_enhanced_compare_module()
    scored, folds = enhanced_compare._rolling_backtest_with_feature_sets(
        usable,
        {
            "enhanced_intraday_ml": base_features,
            "enhanced_weather_regime_ml": candidate_features,
        },
    )
    summary = enhanced_compare._group_summary(scored, ["model_variant"])
    by_hour = enhanced_compare._group_summary(scored, ["model_variant", "issue_hour_utc"])
    by_regime = enhanced_compare._group_summary(scored, ["model_variant", "weather_regime"])
    regime_counts = _regime_counts(usable)
    base = enhanced_compare._row(summary, "enhanced_intraday_ml")
    candidate = enhanced_compare._row(summary, "enhanced_weather_regime_ml")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": AIRPORT,
        "experiment": "enhanced Munich intraday ML vs weather-regime feature layer",
        "target": "official DWD daily Tmax; regimes use only as-of METAR/NWP features",
        "dataset_rows": len(dataset),
        "usable_rows": len(usable),
        "days": int(usable["target_date_local"].nunique()),
        "period": [str(usable["target_date_local"].min()), str(usable["target_date_local"].max())],
        "folds": folds,
        "base_feature_count": len(base_features),
        "candidate_feature_count": len(candidate_features),
        "regime_labels": REGIME_LABELS,
        "regime_feature_columns": REGIME_FEATURE_COLUMNS,
        "regime_counts": json.loads(regime_counts.to_json(orient="records")),
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, candidate, by_regime, regime_counts),
    }
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored, report_dir / "eddm_weather_regime_feature_comparison_rows.parquet")
    summary.to_csv(report_dir / "eddm_weather_regime_feature_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "eddm_weather_regime_feature_comparison_by_hour.csv", index=False)
    by_regime.to_csv(report_dir / "eddm_weather_regime_feature_comparison_by_regime.csv", index=False)
    regime_counts.to_csv(report_dir / "eddm_weather_regime_feature_counts.csv", index=False)
    (report_dir / "eddm_weather_regime_feature_comparison.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/eddm_weather_regime_feature_comparison.md").write_text(
        _markdown(report, summary, by_hour, by_regime, regime_counts),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _load_or_build_dataset() -> pd.DataFrame:
    output = Path("data/processed/intraday_ml_dataset_enhanced_regime.parquet")
    if output.exists():
        return pd.read_parquet(output)
    source = Path("data/processed/intraday_ml_dataset_enhanced.parquet")
    if not source.exists():
        raise FileNotFoundError("Run scripts/68_compare_eddm_intraday_enhanced_features.py first")
    frame = pd.read_parquet(source)
    out = add_weather_regime_columns(frame)
    write_parquet(out, output)
    return out


def _recommendation(base: dict, candidate: dict, by_regime: pd.DataFrame, regime_counts: pd.DataFrame) -> dict:
    mae_delta = float(candidate["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(base["mean_crps"])
    false_upside_delta = float(candidate["mean_false_upside_probability"]) - float(
        base["mean_false_upside_probability"]
    )
    max_regime_nll_regression = _max_group_regression(
        by_regime,
        candidate="enhanced_weather_regime_ml",
        base="enhanced_intraday_ml",
        metric="mean_nll",
        key="weather_regime",
    )
    enough_regime_rows = bool((regime_counts["rows"] >= 30).sum() >= 3)
    checks = {
        "nll_improves_at_least_0_03": nll_delta <= -0.03,
        "mae_not_worse_by_0_03c": mae_delta <= 0.03,
        "crps_not_worse_by_0_005": crps_delta <= 0.005,
        "false_upside_not_worse_by_3pp": false_upside_delta <= 0.03,
        "max_regime_nll_regression_within_0_25": max_regime_nll_regression <= 0.25,
        "enough_regime_rows": enough_regime_rows,
    }
    decision = "promote_to_main_model" if all(checks.values()) else "do_not_promote_yet"
    return {
        "decision": decision,
        "checks": checks,
        "candidate_minus_base_mae": mae_delta,
        "candidate_minus_base_nll": nll_delta,
        "candidate_minus_base_crps": crps_delta,
        "candidate_minus_base_false_upside_probability": false_upside_delta,
        "max_regime_nll_regression": max_regime_nll_regression,
    }


def _max_group_regression(grouped: pd.DataFrame, *, candidate: str, base: str, metric: str, key: str) -> float:
    candidate_rows = grouped[grouped["model_variant"] == candidate][[key, metric]]
    base_rows = grouped[grouped["model_variant"] == base][[key, metric]]
    merged = candidate_rows.merge(base_rows, on=key, suffixes=("_candidate", "_base"))
    if merged.empty:
        return 0.0
    return float((merged[f"{metric}_candidate"] - merged[f"{metric}_base"]).max())


def _regime_counts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["weather_regime"].value_counts().to_dict()
    return pd.DataFrame(
        {
            "weather_regime": label,
            "rows": int(counts.get(label, 0)),
        }
        for label in REGIME_LABELS
    )


def _load_enhanced_compare_module():
    global _ENHANCED_COMPARE
    if _ENHANCED_COMPARE is not None:
        return _ENHANCED_COMPARE
    path = Path("scripts/68_compare_eddm_intraday_enhanced_features.py")
    spec = importlib.util.spec_from_file_location("eddm_enhanced_compare", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ENHANCED_COMPARE = module
    return module


def _markdown(
    report: dict,
    summary: pd.DataFrame,
    by_hour: pd.DataFrame,
    by_regime: pd.DataFrame,
    regime_counts: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# EDDM weather-regime feature comparison",
            "",
            f"- created: `{report['created_at_utc']}`",
            f"- target: {report['target']}",
            f"- period: `{report['period'][0]}` to `{report['period'][1]}`",
            f"- rows: `{report['usable_rows']}`",
            f"- days: `{report['days']}`",
            f"- recommendation: `{report['recommendation']['decision']}`",
            "",
            "## Summary",
            "",
            _table(summary),
            "",
            "## By UTC Issue Hour",
            "",
            _table(by_hour),
            "",
            "## By Weather Regime",
            "",
            _table(by_regime),
            "",
            "## Regime Counts",
            "",
            _table(regime_counts),
            "",
        ]
    )


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
