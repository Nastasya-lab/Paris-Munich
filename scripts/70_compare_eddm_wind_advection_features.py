from __future__ import annotations

import json
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.features.wind_advection import (
    add_wind_advection_features_to_frame,
    wind_advection_feature_columns,
)
from weather_tmax_bot.models.intraday_ml import DEFAULT_INTRADAY_ML_FEATURES


AIRPORT = "EDDM"
TIMEZONE_NAME = "Europe/Berlin"
STATIONS = [AIRPORT]
_ENHANCED_COMPARE = None


def main() -> None:
    dataset = _load_or_build_dataset()
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"], errors="coerce").dt.date
    usable = dataset[dataset["target_date_local"] <= pd.to_datetime("2025-12-30").date()].copy()
    base_features = list(DEFAULT_INTRADAY_ML_FEATURES)
    candidate_features = base_features + wind_advection_feature_columns(STATIONS)
    enhanced_compare = _load_enhanced_compare_module()
    scored, folds = enhanced_compare._rolling_backtest_with_feature_sets(
        usable,
        {
            "enhanced_intraday_ml": base_features,
            "enhanced_wind_advection_ml": candidate_features,
        },
    )
    summary = enhanced_compare._group_summary(scored, ["model_variant"])
    by_hour = enhanced_compare._group_summary(scored, ["model_variant", "issue_hour_utc"])
    by_regime = enhanced_compare._group_summary(scored, ["model_variant", "advection_regime"])
    base = enhanced_compare._row(summary, "enhanced_intraday_ml")
    candidate = enhanced_compare._row(summary, "enhanced_wind_advection_ml")
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "airport": AIRPORT,
        "experiment": "enhanced Munich intraday ML vs EDDM-only wind/advection features",
        "target": "official DWD daily Tmax; wind/advection uses only as-of EDDM METAR",
        "dataset_rows": len(dataset),
        "usable_rows": len(usable),
        "days": int(usable["target_date_local"].nunique()),
        "period": [str(usable["target_date_local"].min()), str(usable["target_date_local"].max())],
        "folds": folds,
        "base_feature_count": len(base_features),
        "candidate_feature_count": len(candidate_features),
        "wind_advection_stations": STATIONS,
        "wind_advection_feature_columns": wind_advection_feature_columns(STATIONS),
        "availability": _availability_report(usable),
        "summary": json.loads(summary.to_json(orient="records")),
        "recommendation": _recommendation(base, candidate),
    }
    report_dir = Path("data/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(scored, report_dir / "eddm_wind_advection_feature_comparison_rows.parquet")
    summary.to_csv(report_dir / "eddm_wind_advection_feature_comparison_summary.csv", index=False)
    by_hour.to_csv(report_dir / "eddm_wind_advection_feature_comparison_by_hour.csv", index=False)
    by_regime.to_csv(report_dir / "eddm_wind_advection_feature_comparison_by_regime.csv", index=False)
    (report_dir / "eddm_wind_advection_feature_comparison.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    Path("docs/eddm_wind_advection_feature_comparison.md").write_text(
        enhanced_compare._markdown(report, summary, by_hour, by_regime),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, default=str))


def _load_or_build_dataset() -> pd.DataFrame:
    output = Path("data/processed/intraday_ml_dataset_enhanced_wind_advection.parquet")
    if output.exists():
        return pd.read_parquet(output)
    source = Path("data/processed/intraday_ml_dataset_enhanced.parquet")
    if not source.exists():
        raise FileNotFoundError("Run scripts/68_compare_eddm_intraday_enhanced_features.py first")
    metar_path = Path("data/interim/metar_iem_EDDM.parquet")
    if not metar_path.exists():
        raise FileNotFoundError(metar_path)
    frame = pd.read_parquet(source)
    metar = pd.read_parquet(metar_path)
    out = add_wind_advection_features_to_frame(
        frame,
        {AIRPORT: metar},
        timezone_name=TIMEZONE_NAME,
        stations=STATIONS,
    )
    out["leakage_check_passed"] = out["leakage_check_passed"].fillna(False).astype(bool) & out[
        "adv_leakage_check_passed"
    ].fillna(False).astype(bool)
    write_parquet(out, output)
    return out


def _availability_report(frame: pd.DataFrame) -> dict:
    return {
        "any_advection_station_available_rate": float((frame["adv_available_station_count"] > 0).mean()),
        "eddm_available_rate": float(frame.get("adv_eddm_available", pd.Series(False, index=frame.index)).mean()),
        "cold_advection_rate": float(frame.get("adv_any_cold_advection_signal", pd.Series(False, index=frame.index)).mean()),
        "warm_advection_rate": float(frame.get("adv_any_warm_advection_signal", pd.Series(False, index=frame.index)).mean()),
        "frontal_passage_rate": float(frame.get("adv_any_frontal_passage_signal", pd.Series(False, index=frame.index)).mean()),
    }


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


def _recommendation(base: dict, candidate: dict) -> dict:
    mae_delta = float(candidate["mae_expected"]) - float(base["mae_expected"])
    nll_delta = float(candidate["mean_nll"]) - float(base["mean_nll"])
    crps_delta = float(candidate["mean_crps"]) - float(base["mean_crps"])
    false_upside_delta = float(candidate["mean_false_upside_probability"]) - float(
        base["mean_false_upside_probability"]
    )
    checks = {
        "mae_improves_at_least_0_02c": mae_delta <= -0.02,
        "nll_not_materially_worse": nll_delta <= 0.03,
        "crps_not_materially_worse": crps_delta <= 0.005,
        "false_upside_not_worse_by_3pp": false_upside_delta <= 0.03,
    }
    decision = "promote_to_main_model" if all(checks.values()) else "do_not_promote_yet"
    return {
        "decision": decision,
        "checks": checks,
        "candidate_minus_base_mae": mae_delta,
        "candidate_minus_base_nll": nll_delta,
        "candidate_minus_base_crps": crps_delta,
        "candidate_minus_base_false_upside_probability": false_upside_delta,
    }


if __name__ == "__main__":
    main()
