from weather_tmax_bot.models.disagreement import assess_model_disagreement


def test_model_disagreement_detects_high_expected_spread():
    audit = assess_model_disagreement(
        {
            "production_champion": {
                "distribution": {
                    "expected_tmax_c": 23.0,
                    "median_tmax_c": 23.0,
                    "most_likely_integer_c": 23,
                    "threshold_probabilities": {"ge_25": 0.1, "ge_30": 0.0},
                }
            },
            "shadow_seasonal_intraday": {
                "distribution": {
                    "expected_tmax_c": 24.0,
                    "median_tmax_c": 24.0,
                    "most_likely_integer_c": 24,
                    "threshold_probabilities": {"ge_25": 0.2, "ge_30": 0.0},
                }
            },
            "shadow_intraday_ml": {
                "distribution": {
                    "expected_tmax_c": 27.2,
                    "median_tmax_c": 27.0,
                    "most_likely_integer_c": 27,
                    "threshold_probabilities": {"ge_25": 0.8, "ge_30": 0.1},
                }
            },
        }
    )

    assert audit["status"] == "evaluated"
    assert audit["severity"] == "high"
    assert "expected_tmax_spread_high" in audit["reasons"]
    assert audit["summary"]["variant_count"] == 3


def test_model_disagreement_is_none_when_variants_are_close():
    audit = assess_model_disagreement(
        {
            "production_champion": {
                "distribution": {
                    "expected_tmax_c": 23.0,
                    "most_likely_integer_c": 23,
                    "threshold_probabilities": {"ge_25": 0.1, "ge_30": 0.0},
                }
            },
            "shadow_seasonal_intraday": {
                "distribution": {
                    "expected_tmax_c": 23.4,
                    "most_likely_integer_c": 23,
                    "threshold_probabilities": {"ge_25": 0.12, "ge_30": 0.0},
                }
            },
        }
    )

    assert audit["severity"] == "none"
    assert audit["reasons"] == []


def test_model_disagreement_ignores_base_prior():
    audit = assess_model_disagreement(
        {
            "production_champion": {
                "distribution": {
                    "expected_tmax_c": 23.0,
                    "most_likely_integer_c": 23,
                    "threshold_probabilities": {"ge_25": 0.1, "ge_30": 0.0},
                }
            },
            "base_prior": {
                "distribution": {
                    "expected_tmax_c": 30.0,
                    "most_likely_integer_c": 30,
                    "threshold_probabilities": {"ge_25": 1.0, "ge_30": 0.5},
                }
            },
            "shadow_seasonal_intraday": {
                "distribution": {
                    "expected_tmax_c": 23.2,
                    "most_likely_integer_c": 23,
                    "threshold_probabilities": {"ge_25": 0.1, "ge_30": 0.0},
                }
            },
        }
    )

    assert "base_prior" not in audit["variants"]
    assert audit["severity"] == "none"
