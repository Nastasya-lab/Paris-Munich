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
            "latest_nwp_source_id": "open_meteo.live.icon_d2",
            "source_compatibility": {
                "metar": {"status": "known_compatible"},
                "taf": {"status": "exact_match"},
                "nwp": {"status": "known_compatible"},
            },
            "model_disagreement": {
                "status": "evaluated",
                "severity": "high",
                "reasons": ["expected_tmax_spread_high"],
                "summary": {
                    "expected_tmax_spread_c": 4.0,
                    "ge_25_probability_spread": 0.3,
                    "ge_30_probability_spread": 0.2,
                },
            },
            "nwp_missing": True,
            "forecast_acceptance": {
                "accepted": False,
                "blocking_reasons": ["quality_status_ok"],
                "cautions": ["preliminary calibration"],
            },
            "forecast_variants": {
                "shadow_phase_arbitrated": {
                    "description": "phase-arbitrated shadow",
                    "distribution": {"probabilities_by_integer_c": {"25": 0.8, "26": 0.2}},
                    "metadata": {
                        "variant_version": "phase_arbitrated_shadow_v1",
                        "selected_variant": "shadow_safe_blend",
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
    assert set(variants["forecast_variant"]) == {"production_champion"}
    assert out.iloc[0]["forecast_id"] == "f1"
    assert out.iloc[0]["metar_source_mismatch"] == True
    assert out.iloc[0]["nwp_source_mismatch"] == True
    assert out.iloc[0]["nwp_source_compatibility_status"] == "known_compatible"
    assert out.iloc[0]["model_disagreement_severity"] == "high"
    assert out.iloc[0]["model_disagreement_expected_spread_c"] == 4.0
    assert out.iloc[0]["nwp_missing"] == True
    assert out.iloc[0]["forecast_accepted"] == False
    assert out.iloc[0]["forecast_acceptance_blocking_reasons"] == "quality_status_ok"
    assert out.iloc[0]["forecast_acceptance_cautions"] == "preliminary calibration"


def test_update_forecast_outcomes_uses_metar_truth_for_metar_target(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    dwd_target_path = tmp_path / "daily_target.parquet"
    metar_target_path = tmp_path / "metar_tmax_target_EDDM.parquet"
    output_path = tmp_path / "monitoring.parquet"
    variant_output_path = tmp_path / "variant_monitoring.parquet"
    record = {
        "forecast_id": "metar-f1",
        "airport": "EDDM",
        "issue_time_utc": "2026-07-15T10:00:00+00:00",
        "target_date_local": "2026-07-15",
        "model_version": "eddm_metar_tmax_icon_d2_spatial_v1",
        "probability_distribution": {"24": 0.2, "25": 0.8},
        "expected_tmax_c": 24.8,
        "median_tmax_c": 25.0,
        "raw_input_metadata": {"target": "METAR_Tmax"},
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-15"],
            "tmax_c": [31.0],
            "quality_flags": ["ok"],
            "source_id": ["dwd.test"],
        }
    ).to_parquet(dwd_target_path, index=False)
    pd.DataFrame(
        {
            "airport_icao": ["EDDM"],
            "target_date_local": ["2026-07-15"],
            "metar_tmax_c": [25.0],
            "quality_flags": ["ok"],
            "source_id": ["metar.test"],
        }
    ).to_parquet(metar_target_path, index=False)

    out = update_forecast_outcomes(
        log_path,
        dwd_target_path,
        output_path,
        variant_output_path=variant_output_path,
        metar_target_paths={"EDDM": metar_target_path},
        metar_archive_dir=tmp_path / "data",
    )
    status = build_forecast_outcome_status(
        log_path,
        dwd_target_path,
        output_path=None,
        metar_target_paths={"EDDM": metar_target_path},
        metar_archive_dir=tmp_path / "data",
    )

    assert out.iloc[0]["actual_tmax_c"] == 25.0
    assert out.iloc[0]["target_kind"] == "METAR_Tmax"
    assert out.iloc[0]["truth_source"] == "metar.test"
    assert status.iloc[0]["actual_tmax_c"] == 25.0
    assert status.iloc[0]["target_kind"] == "METAR_Tmax"
    assert status.iloc[0]["truth_source"] == "metar.test"


def test_update_forecast_outcomes_scores_unimodal_shadow_variant(tmp_path):
    log_path = tmp_path / "forecast_log.jsonl"
    target_path = tmp_path / "daily_target.parquet"
    output_path = tmp_path / "monitoring.parquet"
    variant_output_path = tmp_path / "variant_monitoring.parquet"
    record = {
        "forecast_id": "f1",
        "airport": "LFPB",
        "issue_time_utc": "2026-07-15T12:00:00+00:00",
        "target_date_local": "2026-07-15",
        "model_version": "production_v1",
        "probability_distribution": {"35": 0.1, "36": 0.5, "37": 0.1, "38": 0.3},
        "expected_tmax_c": 36.6,
        "median_tmax_c": 36.0,
        "raw_input_metadata": {
            "forecast_variants": {
                "production_champion": {
                    "description": "production",
                    "distribution": {"probabilities_by_integer_c": {"35": 0.1, "36": 0.5, "37": 0.1, "38": 0.3}},
                    "metadata": {
                        "variant_version": "production_v1",
                        "forecast_phase": "afternoon",
                        "local_issue_hour": 14.0,
                    },
                },
                "shadow_unimodal_pmf": {
                    "description": "unimodal shadow",
                    "distribution": {"probabilities_by_integer_c": {"35": 0.1, "36": 0.35, "37": 0.35, "38": 0.2}},
                    "metadata": {
                        "variant_version": "lfpb_pmf_temperature_unimodal_shadow_v1",
                        "forecast_phase": "afternoon",
                        "local_issue_hour": 14.0,
                    },
                },
            }
        },
    }
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    pd.DataFrame(
        {
            "airport_icao": ["LFPB"],
            "target_date_local": ["2026-07-15"],
            "tmax_c": [37.0],
            "quality_flags": ["ok"],
        }
    ).to_parquet(target_path, index=False)

    update_forecast_outcomes(log_path, target_path, output_path, variant_output_path=variant_output_path)

    variants = pd.read_parquet(variant_output_path)
    assert set(variants["forecast_variant"]) == {"production_champion", "shadow_unimodal_pmf"}
    shadow = variants[variants["forecast_variant"] == "shadow_unimodal_pmf"].iloc[0]
    assert shadow["variant_version"] == "lfpb_pmf_temperature_unimodal_shadow_v1"
    assert shadow["forecast_phase"] == "afternoon"
    assert shadow["probability_actual_integer_bin"] == 0.35


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
