from weather_tmax_bot.evaluation.monitoring import build_monitoring_summary, write_monitoring_report


def test_monitoring_summary_has_expected_keys():
    summary = build_monitoring_summary()
    assert "model_artifacts" in summary
    assert "training_rows" in summary
    assert "active_model" in summary
    assert "registry_health" in summary
    assert "archive_freshness" in summary
    assert "freshness_gate" in summary
    assert "latest_retraining_report" in summary
    assert "outcome_analysis" in summary


def test_write_monitoring_report(tmp_path):
    path = write_monitoring_report(tmp_path / "monitoring_report.md")
    assert path.exists()
    assert "Monitoring report" in path.read_text(encoding="utf-8")
    assert "Active model" in path.read_text(encoding="utf-8")
    assert "Registry health" in path.read_text(encoding="utf-8")
    assert "Archive freshness" in path.read_text(encoding="utf-8")
    assert "Outcome analysis" in path.read_text(encoding="utf-8")
