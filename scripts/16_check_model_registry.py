from __future__ import annotations

import json
from pathlib import Path

import typer

from weather_tmax_bot.models.registry_health import registry_health


def main(
    model_dir: str = typer.Option("data/models"),
    fallback_model_path: str = typer.Option("data/models/quantile_mvp.joblib"),
    report_path: str = typer.Option("data/reports/model_registry_health.json"),
):
    health = registry_health(model_dir=model_dir, fallback_model_path=fallback_model_path)
    output = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(health, indent=2), encoding="utf-8")
    print(json.dumps(health, indent=2))
    if not health["passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
