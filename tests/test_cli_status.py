from typer.testing import CliRunner

from weather_tmax_bot.bot.cli import app


def test_cli_status_runs():
    result = CliRunner().invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Weather Tmax Bot status" in result.stdout
    assert "Registry health:" in result.stdout
    assert "Data freshness:" in result.stdout
    assert "Freshness gate:" in result.stdout
    assert "Pending truth rows:" in result.stdout
