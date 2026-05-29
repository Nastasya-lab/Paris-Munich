from typer.testing import CliRunner

from weather_tmax_bot.bot.cli import app


def test_cli_analyze_runs():
    result = CliRunner().invoke(app, ["analyze"])
    assert result.exit_code == 0
    assert "First analysis" in result.stdout
    assert "Readiness" in result.stdout
