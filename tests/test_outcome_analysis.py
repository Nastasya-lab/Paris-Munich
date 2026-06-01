import pandas as pd

from weather_tmax_bot.evaluation.outcome_analysis import build_outcome_analysis, format_outcome_analysis_markdown


def test_outcome_analysis_pending_without_monitoring(tmp_path):
    analysis = build_outcome_analysis(
        monitoring_path=tmp_path / "missing.parquet",
        output_json_path=tmp_path / "outcome_analysis.json",
        output_markdown_path=tmp_path / "outcome_analysis.md",
    )

    assert analysis["status"] == "pending"
    assert (tmp_path / "outcome_analysis.json").exists()
    assert "No scored forecasts yet" in (tmp_path / "outcome_analysis.md").read_text(encoding="utf-8")


def test_outcome_analysis_ready_with_scored_rows(tmp_path):
    monitoring_path = tmp_path / "forecast_monitoring.parquet"
    variant_path = tmp_path / "forecast_variant_monitoring.parquet"
    pd.DataFrame(
        {
            "forecast_id": ["f1", "f2"],
            "model_version": ["m1", "m1"],
            "target_date_local": ["2026-07-15", "2026-07-16"],
            "actual_tmax_c": [25.0, 20.0],
            "expected_tmax_c": [24.0, 22.0],
            "error_expected_c": [-1.0, 2.0],
            "nll": [2.0, 3.0],
            "crps": [0.5, 1.5],
            "brier_ge_20": [0.1, 0.2],
            "brier_ge_25": [0.2, 0.3],
            "brier_ge_30": [0.3, 0.4],
            "forecast_quality_status": ["ok", "degraded"],
            "forecast_accepted": [True, False],
            "metar_source_mismatch": [False, True],
            "taf_source_mismatch": [False, False],
            "metar_source_compatibility_status": ["same_source", "known_runtime_compatible"],
            "taf_source_compatibility_status": ["same_source", "same_source"],
        }
    ).to_parquet(monitoring_path, index=False)
    pd.DataFrame(
        {
            "forecast_id": ["f1", "f1"],
            "forecast_variant": ["production_champion", "shadow_seasonal_intraday"],
            "model_version": ["m1", "m1"],
            "target_date_local": ["2026-07-15", "2026-07-15"],
            "actual_tmax_c": [25.0, 25.0],
            "expected_tmax_c": [24.0, 25.2],
            "error_expected_c": [-1.0, 0.2],
            "nll": [2.0, 0.5],
            "crps": [0.5, 0.2],
            "brier_ge_20": [0.1, 0.05],
            "brier_ge_25": [0.2, 0.04],
            "brier_ge_30": [0.3, 0.02],
            "probability_actual_integer_bin": [0.2, 0.7],
        }
    ).to_parquet(variant_path, index=False)

    analysis = build_outcome_analysis(
        monitoring_path,
        tmp_path / "analysis.json",
        tmp_path / "analysis.md",
        variant_monitoring_path=variant_path,
    )

    assert analysis["status"] == "ready"
    assert analysis["overall"]["forecasts"] == 2
    assert {row["forecast_variant"] for row in analysis["by_forecast_variant"]} == {
        "production_champion",
        "shadow_seasonal_intraday",
    }
    assert analysis["champion_vs_shadow"]["paired_forecasts"] == 1
    assert analysis["champion_vs_shadow"]["shadow_mae_win_rate"] == 1.0
    assert {row["forecast_quality_status"] for row in analysis["by_quality"]} == {"ok", "degraded"}
    assert {row["forecast_accepted"] for row in analysis["by_acceptance"]} == {"accepted", "rejected"}
    assert {row["metar_source_compatibility_status"] for row in analysis["by_metar_compatibility"]} == {
        "same_source",
        "known_runtime_compatible",
    }


def test_format_outcome_analysis_markdown():
    text = format_outcome_analysis_markdown(
        {
            "status": "ready",
            "rows": 1,
            "overall": {"forecasts": 1, "mean_crps": 0.5},
            "by_model": [],
            "by_quality": [],
            "by_acceptance": [],
            "by_source_mismatch": [],
            "by_forecast_variant": [],
            "champion_vs_shadow": {},
            "worst_by_crps": [],
        }
    )

    assert "Outcome analysis" in text
