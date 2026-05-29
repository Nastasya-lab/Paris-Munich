from typer.testing import CliRunner

from weather_tmax_bot.bot.cli import app


def test_cli_predict_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(tmp_path / "forecast_log.jsonl"))
    result = CliRunner().invoke(app, ["predict", "--airport", "EDDM", "--target-date", "2026-07-15"])
    assert result.exit_code == 0
    assert "Probabilities:" in result.stdout
    assert "Forecast status:" in result.stdout
    assert "Acceptance:" in result.stdout


def test_cli_predict_with_refresh_flags_runs_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(tmp_path / "forecast_log.jsonl"))
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--airport",
            "EDDM",
            "--target-date",
            "2026-05-29",
            "--issue-time",
            "2026-05-28T20:30:00Z",
            "--auto-refresh",
            "--no-refresh-awc",
            "--no-refresh-nwp",
            "--no-log",
        ],
    )
    assert result.exit_code == 0
    assert "Refresh summary:" in result.stdout
    assert "Quality:" in result.stdout


def test_cli_predict_require_ok_fails_for_degraded_forecast(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_TMAX_FORECAST_LOG_PATH", str(tmp_path / "forecast_log.jsonl"))
    result = CliRunner().invoke(
        app,
        [
            "predict",
            "--airport",
            "EDDM",
            "--target-date",
            "2026-05-29",
            "--issue-time",
            "2026-05-29T06:10:00Z",
            "--no-log",
            "--require-ok",
        ],
    )
    assert result.exit_code == 1
    assert "Forecast status: degraded" in result.stdout
    assert "Acceptance gate:" in result.stdout
