from __future__ import annotations

import os
from datetime import datetime
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
LOCAL_TIMEZONE = ZoneInfo("Europe/Berlin")

QUALITY_LABELS = {
    "ok": "данные прошли проверки",
    "degraded": "есть ограничения",
    "invalid": "прогноз нельзя использовать",
}
FRESHNESS_LABELS = {
    "fresh": "свежие",
    "stale": "устарели",
    "missing": "нет данных",
    "future_timestamp": "ошибка времени",
}
MESSAGE_LABELS = {
    "calibration is preliminary": "калибровка вероятностей пока предварительная",
    "known compatible runtime source differs from training source": "оперативный источник отличается от обучающего, но признан совместимым",
    "minor live feature extrapolation": "некоторые текущие значения немного выходят за обучающий диапазон",
    "issue time is outside configured training schedule": "ручной запуск выполнен вне обученного временного слота",
}
NEXT_ACTION_LABELS = {
    "review_outcome_analysis_and_continue_monitoring": "продолжать накопление прогнозов и контроль качества",
    "continue_forward_logging_until_dwd_truth_is_available": "дождаться публикации фактических данных DWD",
    "run_pending_truth_cron_with_fetch": "обновить фактические данные DWD",
    "resolve_forward_ops_blocking_reasons": "устранить причины блокировки",
}


def telegram_configured() -> bool:
    return bool(os.getenv("TELEGRAM_BOT_TOKEN") and os.getenv("TELEGRAM_CHAT_ID"))


def send_telegram_message(text: str, *, parse_mode: str | None = None, timeout: int = 15) -> dict[str, Any]:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return {"sent": False, "reason": "telegram_not_configured"}
    payload: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text[:3900],
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    response = requests.post(TELEGRAM_API.format(token=token), json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    return {"sent": bool(body.get("ok")), "response": body}


def notify_if_configured(text: str) -> dict[str, Any]:
    if os.getenv("WEATHER_TMAX_ENABLE_TELEGRAM", "1").strip().lower() in {"0", "false", "no"}:
        return {"sent": False, "reason": "telegram_disabled"}
    return send_telegram_message(text, parse_mode="HTML")


def format_operational_cycle_message(summary: dict) -> str:
    acceptance = summary.get("forecast_acceptance", {})
    quality = summary.get("forecast_quality", {})
    forecast = summary.get("forecast", {})
    refresh = summary.get("refresh_summary") or {}
    accepted = bool(summary.get("accepted"))
    status = "Прогноз готов" if accepted else "Прогноз требует внимания"
    status_note = (
        "Проверки пройдены: прогноз можно использовать."
        if accepted
        else "Это тестовый или ограниченный прогноз. Не используйте его как штатный."
    )
    lines = [
        f"<b>{status}: {escape(str(summary.get('airport', 'EDDM')))}</b>",
        f"Дата: <b>{escape(str(summary.get('target_date_local', 'не указана')))}</b>",
        f"Выпуск: {_format_local_time(summary.get('issue_time_utc'))}",
        status_note,
        "",
        *_format_temperature_summary(forecast),
        "",
        *_format_probability_bins(forecast.get("probabilities_by_integer_c", {})),
        *_format_thresholds(forecast.get("threshold_probabilities", {})),
        *_format_intraday_summary(forecast.get("forecast_components", {})),
        *_format_shadow_summary(forecast.get("forecast_components", {})),
        "",
        "<b>Данные</b>",
        *_format_freshness(refresh),
        f"Общая проверка свежести: {_yes_no((refresh.get('freshness_gate') or {}).get('passed'))}",
        "",
        "<b>Качество прогноза</b>",
        f"Статус: {QUALITY_LABELS.get(quality.get('status'), escape(str(quality.get('status', 'неизвестен'))))}",
        *_format_notices(quality.get("reasons", []), "Причина"),
        *_format_notices(acceptance.get("cautions", []), "Важно"),
        "",
        "<b>Технические сведения</b>",
        f"Модель: <code>{escape(str(summary.get('model_version', 'не указана')))}</code>",
        f"ID прогноза: <code>{escape(str(summary.get('forecast_id', 'не указан')))}</code>",
    ]
    return "\n".join(lines)


def _format_temperature_summary(forecast: dict) -> list[str]:
    if not forecast:
        return ["<b>Температурный прогноз</b>", "Значения пока недоступны."]
    interval = forecast.get("intervals", {}).get("80", [])
    interval_text = "недоступен"
    if len(interval) == 2:
        interval_text = f"{float(interval[0]):.1f}...{float(interval[1]):.1f} °C"
    return [
        "<b>Температурный прогноз</b>",
        f"Ожидаемый максимум: <b>{float(forecast['expected_tmax_c']):.1f} °C</b>",
        f"Медиана: {float(forecast['median_tmax_c']):.1f} °C",
        f"Самая вероятная корзина: <b>{int(forecast['most_likely_integer_c'])} °C</b>",
        f"Интервал 80%: {interval_text}",
    ]


def _format_probability_bins(probabilities: dict) -> list[str]:
    if not probabilities:
        return ["<b>Вероятности по градусам</b>", "Нет данных."]
    rows = sorted((int(bin_c), float(probability)) for bin_c, probability in probabilities.items())
    material = [(bin_c, probability) for bin_c, probability in rows if probability >= 0.01]
    if not material:
        material = sorted(rows, key=lambda row: row[1], reverse=True)[:5]
        material.sort()
    return [
        "<b>Вероятности по градусам</b>",
        "\n".join(f"{bin_c:+d} °C: <b>{probability:.1%}</b>" for bin_c, probability in material),
    ]


def _format_thresholds(thresholds: dict) -> list[str]:
    if not thresholds:
        return []
    return [
        "",
        "<b>Вероятности событий</b>",
        f"Не ниже +20 °C: {float(thresholds.get('ge_20', 0.0)):.1%}",
        f"Не ниже +25 °C: {float(thresholds.get('ge_25', 0.0)):.1%}",
        f"Не ниже +30 °C: {float(thresholds.get('ge_30', 0.0)):.1%}",
        f"Не выше 0 °C: {float(thresholds.get('le_0', 0.0)):.1%}",
    ]


def _format_intraday_summary(components: dict) -> list[str]:
    intraday = (components or {}).get("intraday_update") or {}
    if not intraday:
        return []
    base = (components or {}).get("base_model") or {}
    lines = [
        "",
        "<b>Сигналы модели</b>",
        f"Базовый ICON-D2 prior: {_format_expected(base)}",
    ]
    if not intraday.get("active"):
        reason = escape(str(intraday.get("reason") or "нет причины"))
        lines.append(f"Внутридневное уточнение: неактивно ({reason})")
        return lines
    lines.extend(
        [
            "Внутридневное уточнение: активно",
            f"Вероятность, что пик уже был: <b>{float(intraday.get('peak_passed_probability', 0.0)):.1%}</b>",
            f"Наблюдаемый максимум: {float(intraday.get('observed_max_so_far_c', 0.0)):.1f} °C",
            f"Падение от пика: {float(intraday.get('drop_from_observed_max_c', 0.0)):.1f} °C",
            f"Вес уточнения в итоге: {float(intraday.get('intraday_blend_weight', 0.0)):.1%}",
        ]
    )
    return lines


def _format_expected(component: dict) -> str:
    value = component.get("expected_tmax_c")
    return "недоступен" if value is None else f"{float(value):.1f} °C"


def _format_shadow_summary(components: dict) -> list[str]:
    shadow = (components or {}).get("shadow_mode") or {}
    if not shadow:
        return []
    intraday = shadow.get("intraday_update") or {}
    final_model = shadow.get("final_model") or {}
    comparison = shadow.get("comparison_to_champion") or {}
    lines = [
        "",
        "<b>Теневой сценарий: seasonal intraday</b>",
        "Не влияет на основной прогноз. Нужен для наглядного сравнения.",
    ]
    if not intraday.get("active"):
        reason = escape(str(intraday.get("reason") or "нет причины"))
        lines.append(f"Статус: неактивен ({reason})")
        return lines

    season = {"warm": "теплый", "cool": "холодный"}.get(
        intraday.get("seasonal_profile"),
        escape(str(intraday.get("seasonal_profile", "неизвестный"))),
    )
    interval = final_model.get("intervals", {}).get("80", [])
    interval_text = "недоступен"
    if len(interval) == 2:
        interval_text = f"{float(interval[0]):.1f}...{float(interval[1]):.1f} °C"
    thresholds = final_model.get("threshold_probabilities", {})
    lines.extend(
        [
            f"Профиль: {season}, вес уточнения <b>{float(intraday.get('intraday_blend_weight', 0.0)):.1%}</b>",
            f"Ожидаемый максимум: <b>{float(final_model.get('expected_tmax_c', 0.0)):.1f} °C</b> ({_signed_c(comparison.get('expected_tmax_delta_c'))} к основному)",
            f"Самая вероятная корзина: <b>{int(final_model.get('most_likely_integer_c', 0))} °C</b>",
            f"Интервал 80%: {interval_text}",
            f"Главные корзины: {_format_compact_bins(final_model.get('probabilities_by_integer_c', {}))}",
            f"Не ниже +25 °C: {float(thresholds.get('ge_25', 0.0)):.1%} ({_signed_pct(comparison.get('ge_25_probability_delta'))})",
            f"Не ниже +30 °C: {float(thresholds.get('ge_30', 0.0)):.1%} ({_signed_pct(comparison.get('ge_30_probability_delta'))})",
        ]
    )
    if intraday.get("late_drop_override_active"):
        lines.append("Late-drop override: активен из-за сильного падения температуры после пика")
    return lines


def _format_compact_bins(probabilities: dict, limit: int = 4) -> str:
    if not probabilities:
        return "нет данных"
    rows = sorted(
        ((int(bin_c), float(probability)) for bin_c, probability in probabilities.items()),
        key=lambda row: row[1],
        reverse=True,
    )[:limit]
    rows.sort()
    return ", ".join(f"{bin_c:+d} °C {probability:.1%}" for bin_c, probability in rows)


def _signed_c(value: Any) -> str:
    return "нет данных" if value is None else f"{float(value):+.1f} °C"


def _signed_pct(value: Any) -> str:
    return "нет данных" if value is None else f"{float(value):+.1%} к основному"


def _format_freshness(refresh: dict) -> list[str]:
    statuses = ((refresh.get("freshness_gate") or {}).get("freshness") or {}).get("statuses", {})
    if not statuses:
        return ["Сведения о свежести источников недоступны."]
    source_labels = {"metar": "METAR", "taf": "TAF", "nwp": "Численная модель"}
    lines = []
    for source in ("metar", "taf", "nwp"):
        status = statuses.get(source, {})
        age = status.get("age_hours")
        age_text = "возраст неизвестен" if age is None else f"{float(age):.1f} ч назад"
        state = FRESHNESS_LABELS.get(status.get("state"), escape(str(status.get("state", "неизвестно"))))
        lines.append(f"{source_labels[source]}: {state}, {age_text}")
    return lines


def format_outcome_update_message(result: dict) -> str:
    status = result.get("status", {})
    refresh = result.get("refresh_summary") or {}
    ran_refresh = bool(result.get("ran_refresh"))
    lines = [
        "<b>Обновление фактических результатов</b>",
        "Проверка завершена.",
        "",
        f"Ожидают данных DWD: <b>{status.get('pending_rows', 0)}</b>",
        f"Готовы к оценке: <b>{status.get('ready_rows', 0)}</b>",
        f"Обновление выполнено: {_yes_no(ran_refresh)}",
    ]
    if ran_refresh:
        lines.extend(
            [
                f"Загружено наблюдений: {refresh.get('fetched_rows', 0)}",
                f"Оценено прогнозов: {refresh.get('forecast_monitoring_rows', 0)}",
            ]
        )
    elif status.get("pending_rows"):
        lines.append("Новых завершенных суток для оценки пока нет.")
    return "\n".join(lines)


def format_healthcheck_message(readiness: dict) -> str:
    ready = bool(readiness.get("ready_for_forward_ops"))
    status = "Система работает штатно" if ready else "Системе требуется внимание"
    lines = [
        f"<b>{status}</b>",
        "",
        f"Выпуск прогнозов: {_enabled_disabled(ready)}",
        f"Контроль качества: {_enabled_disabled(bool(readiness.get('ready_for_outcome_monitoring')))}",
        f"Принято штатных прогнозов: {readiness.get('accepted_operational_forecasts', 0)}",
        f"Ожидают фактических данных: {readiness.get('pending_truth_rows', 0)}",
    ]
    reasons = readiness.get("blocking_reasons", [])
    if reasons:
        lines.extend(["", "<b>Причины блокировки</b>", *_format_notices(reasons, "Проверка")])
    lines.extend(
        [
            "",
            f"Следующий шаг: {_translate(readiness.get('next_action'))}",
        ]
    )
    return "\n".join(lines)


def _format_notices(messages: list[str], prefix: str) -> list[str]:
    return [f"{prefix}: {_translate(message)}" for message in messages]


def _translate(message: Any) -> str:
    text = str(message or "не указан")
    return escape(MESSAGE_LABELS.get(text, NEXT_ACTION_LABELS.get(text, text)))


def _format_local_time(value: Any) -> str:
    if not value:
        return "не указано"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        localized = parsed.astimezone(LOCAL_TIMEZONE)
        return f"<b>{localized:%d.%m.%Y %H:%M}</b> по Мюнхену"
    except ValueError:
        return escape(str(value))


def _yes_no(value: Any) -> str:
    return "да" if value else "нет"


def _enabled_disabled(value: bool) -> str:
    return "работает" if value else "не готов"
