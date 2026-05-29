from weather_tmax_bot.evaluation.first_analysis import build_first_analysis, format_first_analysis_markdown, write_first_analysis_report


def test_first_analysis_builds_from_existing_project_artifacts():
    analysis = build_first_analysis()

    assert "readiness" in analysis
    assert "historical_backtest_ready" in analysis["readiness"]
    assert "operational_outcome_stage" in analysis["readiness"]
    assert "next_actions" in analysis


def test_first_analysis_report_writes_files(tmp_path):
    json_path = tmp_path / "first_analysis.json"
    markdown_path = tmp_path / "first_analysis.md"

    write_first_analysis_report(json_path=json_path, markdown_path=markdown_path)

    assert json_path.exists()
    assert markdown_path.exists()
    assert "First analysis" in markdown_path.read_text(encoding="utf-8")


def test_first_analysis_markdown_has_selected_calibration():
    text = format_first_analysis_markdown(
        {
            "active_model": {"model_version": "m1"},
            "registry_health": {"passed": True},
            "data_volume": {"training_rows": 1, "daily_target_rows": 1, "forecast_log_rows": 0, "forecast_monitoring_rows": 0},
            "readiness": {"historical_backtest_ready": True, "operational_outcome_stage": "pending", "operational_outcome_rows": 0},
            "selected_calibration": {"forecast_variant": "calibrated_spread", "mean_nll": 2.0},
            "operational_acceptance": [{"model_version": "m1", "forecast_accepted": "accepted", "forecasts": 1}],
            "next_actions": ["Review backtest."],
        }
    )

    assert "calibrated_spread" in text
    assert "Operational outcomes" in text
    assert "accepted" in text


def test_first_analysis_marks_first_outcome_as_preliminary(monkeypatch):
    import weather_tmax_bot.evaluation.first_analysis as first_analysis_module

    monkeypatch.setattr(
        first_analysis_module,
        "build_monitoring_summary",
        lambda root=".": {
            "forecast_monitoring_rows": 1,
            "forecast_log_rows": 1,
            "forecast_outcome_status_rows": 1,
            "registry_health": {"passed": True},
            "freshness_gate": {"passed": True},
            "leakage_audit": [],
            "operational_acceptance": [{"forecast_accepted": "accepted", "forecasts": 1}],
        },
    )
    monkeypatch.setattr(first_analysis_module, "_read_table", lambda path: [{"selected_for_production": True}] if "calibration" in str(path) else [{"rows": 1}])

    analysis = build_first_analysis()

    assert analysis["readiness"]["operational_outcome_analysis_ready"] is True
    assert analysis["readiness"]["operational_outcome_useful_sample"] is False
    assert analysis["readiness"]["operational_outcome_stage"] == "first_outcome"
    assert "smoke-test evidence" in analysis["next_actions"][0]
