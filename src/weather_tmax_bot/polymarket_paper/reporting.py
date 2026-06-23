from __future__ import annotations

from html import escape


def format_trade_events(result: dict) -> list[str]:
    messages = []
    settlement_note = (
        "Settlement проверен"
        if result.get("settlement_verified")
        else "Paper-only: settlement источник не подтвержден"
    )
    for event in result.get("events") or []:
        action = str(event["action"])
        side = str(event["side"])
        temperature = _temperature_label(str(event["question"]))
        lines = [
            f"<b>POLYMARKET PAPER - PARIS - {escape(action)}</b>",
            "Источник сигнала: <b>shadow-unimodal</b>",
            f"{escape(settlement_note)}",
            "",
            f"{escape(action)} {escape(side)}: <b>{escape(temperature)}</b>",
            f"Вероятность shadow: <b>{float(event['model_probability']):.1%}</b>",
        ]
        production = event.get("production_probability")
        if production is not None:
            lines.append(f"Вероятность production: {float(production):.1%}")
        lines.extend(
            [
                f"Цена: <b>{float(event['price']):.1%}</b>",
                f"Raw edge: {float(event['raw_edge']):+.1%}",
                f"Effective edge: <b>{float(event['effective_edge']):+.1%}</b>",
                f"Сумма: ${float(event['notional_usd']):.2f}",
                f"Контракты: {float(event['shares']):.2f}",
            ]
        )
        if event.get("realized_pnl_usd") is not None:
            lines.append(f"Результат: <b>{float(event['realized_pnl_usd']):+.2f}$</b>")
        lines.extend(
            [
                "",
                f"Свободный paper-баланс: ${float(result.get('cash_balance_usd', 0.0)):.2f}",
                f"Реализованный PnL: {float(result.get('realized_pnl_usd', 0.0)):+.2f}$",
            ]
        )
        messages.append("\n".join(lines))
    return messages


def _temperature_label(question: str) -> str:
    from weather_tmax_bot.polymarket_paper.mapping import parse_temperature_bucket

    parsed = parse_temperature_bucket(question)
    if parsed is None:
        return question
    temperature, tail = parsed
    suffix = {"or_lower": " or lower", "or_higher": " or higher"}.get(tail, "")
    return f"{temperature:+d} C{suffix}"
