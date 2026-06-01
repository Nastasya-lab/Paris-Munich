import json

import pandas as pd

from weather_tmax_bot.evaluation.outcomes import build_forecast_outcome_status, update_forecast_outcomes


def test_update_forecast_outcomes(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    target_path = tmp_path / "daily_target.parquet"
    output_path = tmp_path / "monitoring.parquet"
    variant_output_path = tmp_path / "variant_monitoring.parquet"
    record = {
        "forecast_id": "f1",
        "airport": "EDDM",
        "issue_time_utc": "2026-07-15T06:00:00+00:00",
        "target_date_local": "2026-07-15",
        "model_version": "test",
        "probability_distribution": {"24": 0.4, "25": 0.6},
        "expected_tmax_c": 24.6,
        "median_tmax_c": 25.0,
        "raw_input_metadata": {
            "latest_metar_source_id": "awc.metar.live.EDDM",
            "latest_taf_source_id": "iem.taf.archive.EDDM",
            "nwp_missing": True,
            "forecast_acceptance": {
                "accepted": False,
                "blocking_reasons": ["quality_status_ok"],
                "cautions": ["preliminary calibration"],
            },
            "forecast_variants": {
                "shadow_seasonal_intraday": {
                    "description": "shadow",
                    "distribution": {"probabilities_by_integer_c": {"25": 0.8, "26": 0.2}},
                    "metadata": {
                        "variant_version": "phase_aware_intraday_challenger_v3",
                        "forecast_phase": "midday_update",
                        "scenario_tracking": "near_observed_track",
                        "local_issue_hour": 8.0,
                    },
                }
            },
        },
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-15"],
            "tmax_c": [25.0],
            "quality_flags": ["ok"],
        }
    ).to_parquet(target_path, index=False)
    out = update_forecast_outcomes(log_path, target_path, output_path, variant_output_path=variant_output_path)
    assert len(out) == 1
    assert output_path.exists()
    assert variant_output_path.exists()
    variants = pd.read_parquet(variant_output_path)
    assert set(variants["forecast_variant"]) == {"production_champion", "shadow_seasonal_intraday"}
    assert variants.loc[variants["forecast_variant"] == "shadow_seasonal_intraday", "probability_actual_integer_bin"].iloc[0] == 0.8
    shadow = variants[variants["forecast_variant"] == "shadow_seasonal_intraday"].iloc[0]
    assert shadow["variant_version"] == "phase_aware_intraday_challenger_v3"
    assert shadow["forecast_phase"] == "midday_update"
    assert shadow["scenario_tracking"] == "near_observed_track"
    assert shadow["probability_above_actual_integer_bin"] == 0.2
    assert shadow["coverage_80"] == True
    assert out.iloc[0]["forecast_id"] == "f1"
    assert out.iloc[0]["metar_source_mismatch"] == True
    assert out.iloc[0]["nwp_missing"] == True
    assert out.iloc[0]["forecast_accepted"] == False
    assert out.iloc[0]["forecast_acceptance_blocking_reasons"] == "quality_status_ok"
    assert out.iloc[0]["forecast_acceptance_cautions"] == "preliminary calibration"


def test_build_forecast_outcome_status_marks_pending_and_scored(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    target_path = tmp_path / "daily_target.parquet"
    output_path = tmp_path / "status.parquet"
    records = [
        {
            "forecast_id": "f1",
            "airport": "EDDM",
            "issue_time_utc": "2026-07-15T06:00:00+00:00",
            "target_date_local": "2026-07-15",
            "model_version": "test",
            "raw_input_metadata": {"forecast_acceptance": {"accepted": True}, "forecast_quality": {"status": "ok"}},
        },
        {
            "forecast_id": "f2",
            "airport": "EDDM",
            "issue_time_utc": "2026-07-16T06:00:00+00:00",
            "target_date_local": "2026-07-16",
            "model_version": "test",
            "raw_input_metadata": {
                "forecast_acceptance": {"accepted": False, "blocking_reasons": ["quality_status_ok"]},
                "forecast_quality": {"status": "degraded"},
            },
        },
    ]
    log_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-15"],
            "tmax_c": [25.0],
            "quality_flags": ["ok"],
        }
    ).to_parquet(target_path, index=False)

    status = build_forecast_outcome_status(log_path, target_path, output_path)

    assert set(status["outcome_status"]) == {"scored", "pending_truth"}
    pending = status[status["outcome_status"] == "pending_truth"].iloc[0]
    assert pending["forecast_accepted"] == False
    assert pending["forecast_quality_status"] == "degraded"
    assert pending["forecast_acceptance_blocking_reasons"] == "quality_status_ok"
    assert output_path.exists()
