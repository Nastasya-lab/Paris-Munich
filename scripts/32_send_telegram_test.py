from __future__ import annotations

import json

import typer

from weather_tmax_bot.notifications.telegram import notify_if_configured


def main(message: str = typer.Option("Weather Tmax Bot Telegram test: notifications are configured.")):
    result = notify_if_configured(message)
    print(json.dumps(result, indent=2, default=str))
    if not result.get("sent"):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
