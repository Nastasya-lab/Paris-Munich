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
    "unknown runtime source differs from training source": "источник отличается от обучающего, совместимость не подтверждена",
    "forbidden runtime source differs from training source": "источник запрещен для этой роли модели",
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
        *_format_growth_potential_summary(forecast.get("forecast_components", {})),
        "",
        "<b>Данные</b>",
        *_format_freshness(refresh),
        *_format_source_compatibility(forecast.get("source_compatibility", {})),
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
    lines.extend(_format_shadow_interpretation(intraday))
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


def _format_ml_shadow_summary(components: dict) -> list[str]:
    shadow = (components or {}).get("ml_shadow_mode") or {}
    if not shadow:
        return []
    details = shadow.get("details") or {}
    final_model = shadow.get("final_model") or {}
    calibration_status = str(details.get("calibration_status", "unknown"))
    calibration_note = (
        "Предварительная ML-модель. Не влияет на основной прогноз; вероятности проходят отдельную out-of-fold калибровку."
        if calibration_status == "contextual_out_of_fold_survival_calibrated"
        else "Предварительная ML-модель. Не влияет на основной прогноз; последний калибратор проверен, но не включен из-за gate."
    )
    lines = [
        "",
        "<b>ML shadow: remaining upside</b>",
        calibration_note,
    ]
    if not details.get("active"):
        lines.append(f"Статус: неактивна ({escape(str(details.get('reason', 'нет данных')))})")
        return lines
    lines.extend(
        [
            f"Ожидаемый максимум: {float(final_model.get('expected_tmax_c', 0.0)):.1f} °C",
            f"Калибровка: {escape(calibration_status)}",
            f"Вероятность, что пик уже был: {float(details.get('probability_peak_already_passed', 0.0)):.1%}",
            f"Шанс роста еще минимум на +1 °C: {float(details.get('probability_upside_ge_1c', 0.0)):.1%}",
            f"Шанс роста еще минимум на +2 °C: {float(details.get('probability_upside_ge_2c', 0.0)):.1%}",
            f"Шанс роста еще минимум на +3 °C: {float(details.get('probability_upside_ge_3c', 0.0)):.1%}",
            f"Распределение: {_format_compact_bins(final_model.get('probabilities_by_integer_c', {}), limit=6)}",
        ]
    )
    return lines


def _format_growth_potential_summary(components: dict) -> list[str]:
    shadow = (components or {}).get("ml_shadow_mode") or {}
    details = shadow.get("details") or {}
    if not details:
        return []
    if not details.get("active"):
        reason = escape(str(details.get("reason", "нет данных")))
        return ["", "<b>Потенциал роста</b>", f"ML-сигнал недоступен: {reason}"]
    return [
        "",
        "<b>Потенциал роста</b>",
        f"Вероятность, что пик уже был: {float(details.get('probability_peak_already_passed', 0.0)):.1%}",
        f"Шанс роста еще минимум на +1 °C: {float(details.get('probability_upside_ge_1c', 0.0)):.1%}",
        f"Шанс роста еще минимум на +2 °C: {float(details.get('probability_upside_ge_2c', 0.0)):.1%}",
        f"Шанс роста еще минимум на +3 °C: {float(details.get('probability_upside_ge_3c', 0.0)):.1%}",
    ]


def _format_phase_arbitrated_summary(components: dict) -> list[str]:
    shadow = (components or {}).get("phase_arbitrated_shadow_mode") or {}
    if not shadow:
        return []
    details = shadow.get("details") or {}
    final_model = shadow.get("final_model") or {}
    comparison = shadow.get("comparison_to_champion") or {}
    selected = str(details.get("selected_variant") or "unknown")
    selected_label = {
        "production_champion": "основная модель",
        "shadow_safe_blend": "дневной safe-blend",
        "shadow_intraday_ml": "ML remaining-upside",
        "shadow_seasonal_intraday": "вечерняя seasonal intraday",
    }.get(selected, selected)
    return [
        "",
        "<b>Экспериментальная модель</b>",
        "Главный challenger: phase-arbitrated. Не влияет на основной прогноз, только сравнивается.",
        f"Выбранный компонент: <b>{escape(selected_label)}</b>",
        f"Ожидаемый максимум: <b>{float(final_model.get('expected_tmax_c', 0.0)):.1f} °C</b> ({_signed_c(comparison.get('expected_tmax_delta_c'))} к основной)",
        f"Главные корзины: {_format_compact_bins(final_model.get('probabilities_by_integer_c', {}), limit=6)}",
        f"Причина выбора: {escape(str(details.get('selection_reason', 'не указана')))}",
    ]


def _format_model_disagreement(components: dict) -> list[str]:
    audit = (components or {}).get("model_disagreement") or {}
    if audit.get("status") != "evaluated":
        return []
    summary = audit.get("summary") or {}
    variants = audit.get("variants") or {}
    severity = audit.get("severity", "none")
    if severity == "none" and len(variants) < 3:
        return []
    severity_label = {
        "none": "расхождение небольшое",
        "watch": "нужно наблюдать",
        "high": "сильное расхождение",
    }.get(severity, escape(str(severity)))
    lines = [
        "",
        "<b>Расхождение моделей</b>",
        f"Статус: <b>{severity_label}</b>",
        f"Разброс ожидаемого Tmax: {float(summary.get('expected_tmax_spread_c', 0.0)):.1f} °C",
        f"Разброс P(Tmax ≥ 25 °C): {float(summary.get('ge_25_probability_spread', 0.0)):.1%}",
        f"Разброс P(Tmax ≥ 30 °C): {float(summary.get('ge_30_probability_spread', 0.0)):.1%}",
    ]
    for name, label in (
        ("production_champion", "Champion"),
        ("shadow_seasonal_intraday", "Phase shadow"),
        ("shadow_intraday_ml", "ML shadow"),
    ):
        item = variants.get(name)
        if item:
            thresholds = item.get("threshold_probabilities", {})
            lines.append(
                f"{label}: {float(item.get('expected_tmax_c', 0.0)):.1f} °C, "
                f"корзина {int(item.get('most_likely_integer_c', 0))} °C, "
                f"P≥30 {float(thresholds.get('ge_30', 0.0)):.1%}"
            )
    return lines


def _format_blended_shadow_summary(components: dict) -> list[str]:
    shadow = (components or {}).get("blended_shadow_mode") or {}
    if not shadow:
        return []
    details = shadow.get("details") or {}
    final_model = shadow.get("final_model") or {}
    comparison = shadow.get("comparison_to_champion") or {}
    reasons = ", ".join(str(reason) for reason in details.get("reasons", [])) or "нет"
    return [
        "",
        "<b>Безопасный blended shadow</b>",
        "Не влияет на основной прогноз. Смешивает только гладкие champion и phase-shadow распределения.",
        f"Ожидаемый максимум: <b>{float(final_model.get('expected_tmax_c', 0.0)):.1f} °C</b> ({_signed_c(comparison.get('expected_tmax_delta_c'))} к основному)",
        f"Вес phase-shadow: <b>{float(details.get('blend_weight', 0.0)):.1%}</b>",
        f"ML-сигнал учтен как ограничитель веса: {_yes_no(details.get('ml_signal_used'))}",
        "Рваное ML-распределение напрямую не смешивается.",
        f"Главные корзины: {_format_compact_bins(final_model.get('probabilities_by_integer_c', {}), limit=6)}",
        f"Причины веса: {escape(reasons)}",
    ]


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


def _format_source_compatibility(sources: dict) -> list[str]:
    if not sources:
        return []
    status_labels = {
        "exact_match": "совпадает с обучающим",
        "known_compatible": "отличается, но признан совместимым",
        "unknown_mismatch": "отличается, совместимость не подтверждена",
        "forbidden_mismatch": "запрещен для этой роли",
        "missing": "нет данных",
    }
    source_labels = {"metar": "METAR", "taf": "TAF", "nwp": "NWP"}
    lines = ["", "<b>Контроль источников</b>"]
    for kind in ("metar", "taf", "nwp"):
        item = sources.get(kind) or {}
        status = str(item.get("status", "missing"))
        runtime = item.get("runtime_source_id")
        runtime_text = "нет данных" if not runtime else f"<code>{escape(str(runtime))}</code>"
        lines.append(f"{source_labels[kind]}: {runtime_text}, {status_labels.get(status, escape(status))}")
    return lines


def format_metar_event_message(payload: dict, comparison: dict, reasons: list[str] | None = None) -> str:
    forecast_components = payload.get("forecast_components", {}) or {}
    intraday = forecast_components.get("intraday_update", {}) or {}
    shadow = forecast_components.get("shadow_mode", {}) or {}
    latest_metar = payload.get("latest_metar_record") or {}
    previous = comparison.get("previous") or {}
    current = comparison.get("current") or {}
    deltas = comparison.get("deltas") or {}
    thresholds = payload.get("threshold_probabilities", {}) or {}
    lines = [
        f"<b>METAR-обновление прогноза: {escape(str(payload.get('airport', 'EDDM')))}</b>",
        f"Дата: <b>{escape(str(payload.get('target_date_local', 'не указана')))}</b>",
        f"Выпуск: {_format_local_time(payload.get('issue_time_utc'))}",
        "Причина: появился новый METAR, пересчитан intraday-прогноз.",
        "",
        "<b>Что изменилось</b>",
    ]
    if previous:
        lines.extend(
            [
                f"Ожидаемый максимум: <b>{float(payload.get('expected_tmax_c', 0.0)):.1f} °C</b> ({_signed_c(deltas.get('expected_tmax_delta_c'))})",
                f"Самая вероятная корзина: <b>{int(payload.get('most_likely_integer_c', 0))} °C</b> ({_format_bin_change(current, previous)})",
                f"P(Tmax ≥ 20 °C): {float(thresholds.get('ge_20', 0.0)):.1%} ({_signed_pct(deltas.get('ge_20_delta'))})",
                f"P(Tmax ≥ 25 °C): {float(thresholds.get('ge_25', 0.0)):.1%} ({_signed_pct(deltas.get('ge_25_delta'))})",
                f"P(Tmax ≥ 30 °C): {float(thresholds.get('ge_30', 0.0)):.1%} ({_signed_pct(deltas.get('ge_30_delta'))})",
            ]
        )
    else:
        lines.extend(
            [
                f"Ожидаемый максимум: <b>{float(payload.get('expected_tmax_c', 0.0)):.1f} °C</b>",
                f"Самая вероятная корзина: <b>{int(payload.get('most_likely_integer_c', 0))} °C</b>",
                f"P(Tmax ≥ 25 °C): {float(thresholds.get('ge_25', 0.0)):.1%}",
                f"P(Tmax ≥ 30 °C): {float(thresholds.get('ge_30', 0.0)):.1%}",
            ]
        )

    lines.extend(
        [
            "",
            "<b>METAR-сигнал</b>",
            f"Последняя температура: {float(intraday.get('last_metar_temp_c', 0.0)):.1f} °C",
            f"Наблюдаемый максимум: {float(intraday.get('observed_max_so_far_c', 0.0)):.1f} °C",
            f"Падение от максимума: {float(intraday.get('drop_from_observed_max_c', 0.0)):.1f} °C",
            f"Вероятность, что пик уже был: {float(intraday.get('peak_passed_probability', 0.0)):.1%}",
            f"Вес intraday champion: {float(intraday.get('intraday_blend_weight', 0.0)):.1%}",
        ]
    )
    lines.extend(["", *_format_latest_metar_record(latest_metar)])
    lines.extend(["", *_format_distribution_change(payload, previous)])
    lines.extend(_format_source_compatibility(payload.get("source_compatibility", {})))
    lines.extend(_format_growth_potential_summary(forecast_components))
    shadow_final = {}
    shadow_intraday = shadow.get("intraday_update") or {}
    if shadow_final:
        lines.extend(
            [
                "",
                "<b>Shadow-сценарий</b>",
                f"Ожидаемый максимум: {float(shadow_final.get('expected_tmax_c', 0.0)):.1f} °C",
                f"Вес seasonal intraday: {float(shadow_intraday.get('intraday_blend_weight', 0.0)):.1%}",
                f"Фаза shadow: {escape(str(shadow_intraday.get('forecast_phase', 'неизвестно')))}",
                f"Распределение: {_format_compact_bins(shadow_final.get('probabilities_by_integer_c', {}), limit=6)}",
                f"P(Tmax ≥ 30 °C): {float((shadow_final.get('threshold_probabilities') or {}).get('ge_30', 0.0)):.1%}",
            ]
        )
        lines.extend(_format_shadow_interpretation(shadow_intraday))
        if shadow_intraday.get("survival_adjustment_active"):
            lines.extend(
                [
                    "Сезонная поправка после 17:00: <b>активна</b>",
                    f"Историческая вероятность нового пика: {float(shadow_intraday.get('seasonal_survival_prior', 0.0)):.1%}",
                    f"Вероятность повышения до поправки: {float(shadow_intraday.get('survival_original_upside_probability', 0.0)):.1%}",
                    f"После поправки cap_blend × {float(shadow_intraday.get('survival_adjustment_strength', 0.0)):.2f}: <b>{float(shadow_intraday.get('survival_adjusted_upside_probability', 0.0)):.1%}</b>",
                ]
            )
    if reasons:
        lines.extend(["", "<b>Почему отправлено</b>", ", ".join(_translate_reason(reason) for reason in reasons)])
    lines.extend(
        [
            "",
            "<b>Технически</b>",
            f"Модель: <code>{escape(str(payload.get('model_version', 'не указана')))}</code>",
            f"ID прогноза: <code>{escape(str(payload.get('forecast_id', 'не указан')))}</code>",
        ]
    )
    return "\n".join(lines)


def _format_shadow_interpretation(intraday: dict) -> list[str]:
    if not intraday:
        return []
    phase = str(intraday.get("forecast_phase") or "")
    scenario = str(intraday.get("scenario_tracking") or "")
    phase_labels = {
        "morning_prior": "Интерпретация: утренний prior, текущий METAR не должен слишком рано закрывать потенциал дневного прогрева.",
        "midday_update": "Интерпретация: дневное уточнение, METAR уже важен, но ICON/TAF еще могут оставить апсайд.",
        "late_nowcast": "Интерпретация: поздний nowcast, наблюдения и история времени пика получают высокий вес.",
    }
    scenario_labels = {
        "temporary_disruption_possible": "Сценарий: возможный временный погодный сбой, после которого прогрев еще возможен.",
        "heating_cutoff_likely": "Сценарий: вероятное завершение прогрева или прохождение дневного пика.",
        "multi_source_adverse_weather": "Сценарий: METAR, TAF и ICON одновременно указывают на неблагоприятную погоду.",
        "taf_and_metar_adverse": "Сценарий: TAF и METAR согласованно показывают неблагоприятную погоду.",
        "nwp_still_supports_higher_tmax": "Сценарий: ICON еще поддерживает более высокий максимум.",
        "near_observed_track": "Сценарий: день идет близко к текущей наблюдаемой траектории.",
    }
    lines = []
    if phase in phase_labels:
        lines.append(phase_labels[phase])
    if scenario in scenario_labels:
        lines.append(scenario_labels[scenario])
    if intraday.get("nwp_adverse_weather_signal"):
        components = ", ".join(str(item) for item in intraday.get("nwp_adverse_weather_components", [])) or "есть"
        lines.append(f"ICON adverse-weather signal: {escape(components)}")
    return lines


def _signed_pp(value: Any) -> str:
    return "\u043d\u0435\u0442 \u0434\u0430\u043d\u043d\u044b\u0445" if value is None else f"{float(value) * 100:+.1f} \u043f.\u043f."


def _format_latest_metar_record(record: dict) -> list[str]:
    if not record:
        return []
    lines = ["<b>\u0418\u0441\u043f\u043e\u043b\u044c\u0437\u043e\u0432\u0430\u043d\u043d\u044b\u0439 METAR</b>"]
    observation_time = record.get("observation_time_utc")
    if observation_time:
        lines.append(f"\u0412\u0440\u0435\u043c\u044f: {_format_local_time(observation_time)}")
    if record.get("temperature_c") is not None:
        lines.append(f"\u0422\u0435\u043c\u043f\u0435\u0440\u0430\u0442\u0443\u0440\u0430 \u0432 \u0441\u044b\u0440\u043e\u043c METAR: {float(record.get('temperature_c')):.1f} \u00b0C")
    raw_metar = record.get("raw_metar")
    if raw_metar:
        lines.append(f"\u0421\u044b\u0440\u0430\u044f \u0441\u0442\u0440\u043e\u043a\u0430: <code>{escape(str(raw_metar))}</code>")
    return lines


def _format_distribution_change(payload: dict, previous: dict) -> list[str]:
    current_distribution = _probability_mapping(payload.get("probabilities_by_integer_c") or {})
    if not current_distribution:
        return ["<b>Распределение по градусам</b>", "Нет данных по корзинам."]

    previous_distribution = _probability_mapping((previous or {}).get("probabilities_by_integer_c") or {})
    bins = sorted(set(current_distribution) | set(previous_distribution))
    if not previous_distribution:
        return [
            "<b>Распределение по градусам</b>",
            "\n".join(f"{bin_c:+d} °C: <b>{current_distribution.get(bin_c, 0.0):.1%}</b>" for bin_c in bins),
        ]

    unchanged = all(abs(current_distribution.get(bin_c, 0.0) - previous_distribution.get(bin_c, 0.0)) < 0.0005 for bin_c in bins)
    lines = ["<b>Распределение по градусам</b>"]
    if unchanged:
        lines.append("По сравнению с предыдущим METAR распределение не изменилось.")
    rows = []
    for bin_c in bins:
        current = current_distribution.get(bin_c, 0.0)
        previous_probability = previous_distribution.get(bin_c, 0.0)
        if current < 0.001 and previous_probability < 0.001:
            continue
        rows.append(f"{bin_c:+d} °C: <b>{current:.1%}</b> ({_signed_pp(current - previous_probability)})")
    lines.append("\n".join(rows) if rows else "Все показанные корзины остались на 0.0%.")
    return lines


def _probability_mapping(values: dict) -> dict[int, float]:
    result: dict[int, float] = {}
    for key, value in values.items():
        try:
            result[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(result.items()))


def _format_bin_change(current: dict, previous: dict) -> str:
    current_bin = current.get("most_likely_integer_c")
    previous_bin = previous.get("most_likely_integer_c")
    if current_bin is None or previous_bin is None:
        return "предыдущая неизвестна"
    if int(current_bin) == int(previous_bin):
        return "без изменений"
    return f"было {int(previous_bin)} °C"


def _translate_reason(reason: str) -> str:
    labels = {
        "first_new_metar_forecast": "первый METAR-прогноз за день",
        "expected_tmax_changed": "изменился ожидаемый Tmax",
        "most_likely_bin_changed": "изменилась главная корзина",
        "ge_20_probability_changed": "заметно изменилась вероятность ≥20 °C",
        "ge_25_probability_changed": "заметно изменилась вероятность ≥25 °C",
        "ge_30_probability_changed": "заметно изменилась вероятность ≥30 °C",
        "temperature_dropped_from_observed_max": "температура заметно упала от максимума",
        "peak_likely_passed": "вероятно, дневной пик уже пройден",
        "shadow_differs_from_champion": "shadow заметно расходится с основной моделью",
    }
    return labels.get(reason, escape(str(reason)))


def format_daily_model_report_message(report: dict) -> str:
    mode = str(report.get("mode", "preliminary_metar"))
    is_final = mode == "dwd_final"
    title = "Финальный дневной разбор моделей" if is_final else "Вечерний предварительный разбор моделей"
    actual_label = "Факт DWD" if is_final else "Предварительный максимум по METAR"
    lines = [
        f"<b>{escape(title)}: {escape(str(report.get('airport', 'EDDM')))}</b>",
        f"Дата: <b>{escape(str(report.get('target_date_local', 'не указана')))}</b>",
        f"{actual_label}: <b>{float(report.get('actual_tmax_c', 0.0)):.1f} °C</b>",
        f"Источник факта: {escape(str(report.get('truth_source', 'не указан')))}",
        "",
    ]
    analysis = report.get("analysis") or []
    if analysis:
        lines.extend(["<b>Короткий вывод</b>", *[escape(str(item)) for item in analysis], ""])
    summary = report.get("summary_by_variant") or []
    if summary:
        lines.append("<b>Сравнение моделей</b>")
        for item in summary[:6]:
            lines.append(
                f"{escape(str(item.get('forecast_variant')))}: "
                f"MAE {float(item.get('mae_expected', 0.0)):.2f} °C, "
                f"bias {float(item.get('bias_expected', 0.0)):+.2f} °C, "
                f"P(факт. корзина) {float(item.get('mean_probability_actual_integer_bin', 0.0)):.1%}, "
                f"покрытие {float(item.get('coverage_ratio', 0.0)):.0%}"
            )
    hourly = report.get("hourly_comparison") or []
    if hourly:
        lines.extend(["", "<b>По часам</b>"])
        for item in hourly[:12]:
            variants = item.get("variants") or []
            champion = _find_variant_summary(variants, "production_champion")
            challenger = _find_variant_summary(variants, "shadow_phase_arbitrated")
            if champion and challenger:
                lines.append(
                    f"{int(item.get('local_hour', 0)):02d}:00: "
                    f"champion MAE {float(champion.get('mae_expected', 0.0)):.2f} °C, "
                    f"experimental MAE {float(challenger.get('mae_expected', 0.0)):.2f} °C; "
                    f"лучше {escape(str(item.get('best_variant')))}"
                )
    best = report.get("best_variant") or {}
    worst = report.get("worst_variant") or {}
    if best or worst:
        lines.append("")
        lines.append("<b>Итог</b>")
        if best:
            lines.append(f"Лучше: <b>{escape(str(best.get('forecast_variant')))}</b>")
        if worst:
            lines.append(f"Хуже: <b>{escape(str(worst.get('forecast_variant')))}</b>")
    if not is_final:
        lines.extend(
            [
                "",
                "Это не финальная оценка качества: после прихода DWD truth будет отдельный отчет.",
            ]
        )
    return "\n".join(lines)


def _find_variant_summary(rows: list[dict], name: str) -> dict | None:
    for row in rows:
        if row.get("forecast_variant") == name:
            return row
    return None


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
