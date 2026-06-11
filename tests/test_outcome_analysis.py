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
            "model_disagreement_status": ["evaluated", "evaluated"],
            "model_disagreement_severity": ["none", "high"],
            "metar_source_mismatch": [False, True],
            "taf_source_mismatch": [False, False],
            "nwp_source_mismatch": [False, True],
            "metar_source_compatibility_status": ["exact_match", "known_compatible"],
            "taf_source_compatibility_status": ["exact_match", "exact_match"],
            "nwp_source_compatibility_status": ["exact_match", "known_compatible"],
        }
    ).to_parquet(monitoring_path, index=False)
    pd.DataFrame(
        {
            "forecast_id": ["f1"],
            "issue_time_utc": ["2026-07-15T06:00:00+00:00"],
            "forecast_variant": ["production_champion"],
            "model_version": ["m1"],
            "target_date_local": ["2026-07-15"],
            "actual_tmax_c": [25.0],
            "expected_tmax_c": [24.0],
            "error_expected_c": [-1.0],
            "nll": [2.0],
            "crps": [0.5],
            "brier_ge_20": [0.1],
            "brier_ge_25": [0.2],
            "brier_ge_30": [0.3],
            "probability_actual_integer_bin": [0.2],
            "probability_above_actual_integer_bin": [0.1],
            "coverage_80": [True],
            "variant_version": ["production_dynamic_v1"],
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
    }
    assert analysis["champion_vs_shadow"] == {}
    assert {row["forecast_quality_status"] for row in analysis["by_quality"]} == {"ok", "degraded"}
    assert {row["forecast_accepted"] for row in analysis["by_acceptance"]} == {"accepted", "rejected"}
    assert {row["model_disagreement_severity"] for row in analysis["by_model_disagreement"]} == {"none", "high"}
    assert {row["metar_source_compatibility_status"] for row in analysis["by_metar_compatibility"]} == {
        "exact_match",
        "known_compatible",
    }
    assert {row["nwp_source_compatibility_status"] for row in analysis["by_nwp_compatibility"]} == {
        "exact_match",
        "known_compatible",
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
