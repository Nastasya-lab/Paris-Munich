from __future__ import annotations

import argparse
import html
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from weather_tmax_bot.data.previous_runs import build_previous_day1_snapshots, fetch_previous_day1_hourly
from weather_tmax_bot.features.metar_upside_dataset import build_current_metar_upside_features
from weather_tmax_bot.models.multinwp_tmax import RCSS_NWP_MODELS, blend_nwp_features
from weather_tmax_bot.models.distribution import project_unimodal_distribution
from weather_tmax_bot.notifications.telegram import notify_if_configured
from weather_tmax_bot.operations.refresh import refresh_awc_live


AIRPORT = "RCSS"
TIMEZONE = "Asia/Taipei"
LATITUDE = 25.069722
LONGITUDE = 121.5525
MODEL_PATH = Path("data/models/rcss_metar_tmax_multinwp_d1_v1.joblib")
METADATA_PATH = Path("data/models/rcss_metar_tmax_multinwp_d1_v1.metadata.json")
METAR_PATH = Path("data/forecasts/awc_metar_live_RCSS.parquet")
REPORT_PATH = Path("data/reports/latest_rcss_prediction.json")
HISTORY_PATH = Path("data/logs/rcss_forecast_history.jsonl")
LIVE_CACHE_DIR = Path("data/cache/rcss_previous_day1_live")
TRAINED_LOCAL_HOUR_MIN = 6
TRAINED_LOCAL_HOUR_MAX = 20


def main() -> None:
    args = _parse_args()
    refresh = refresh_awc_live(AIRPORT)
    issue = _issue_time(args.issue_time)
    target = date.fromisoformat(args.target_date) if args.target_date else issue.astimezone(ZoneInfo(TIMEZONE)).date()
    if issue.astimezone(ZoneInfo(TIMEZONE)).date() != target:
        raise ValueError("RCSS live forecast currently supports the current local day only")
    actual_local_hour = issue.astimezone(ZoneInfo(TIMEZONE)).hour
    model_local_hour, forecast_mode = forecast_mode_for_hour(actual_local_hour)

    metar = pd.read_parquet(METAR_PATH)
    feature_row = build_current_metar_upside_features(
        metar,
        airport_icao=AIRPORT,
        target_date_local=target,
        issue_time_utc=issue,
        timezone_name=TIMEZONE,
    )
    nwp_features, nwp_errors = load_live_nwp_features(
        target,
        issue,
        cache_dir=LIVE_CACHE_DIR,
        model_local_hour=model_local_hour,
    )
    feature_row.update(nwp_features)
    feature_row["actual_local_issue_hour"] = actual_local_hour
    feature_row["local_issue_hour"] = float(model_local_hour)
    feature_row["month"] = target.month
    day_of_year = target.timetuple().tm_yday
    feature_row["doy_sin"] = float(np.sin(2 * np.pi * day_of_year / 366.0))
    feature_row["doy_cos"] = float(np.cos(2 * np.pi * day_of_year / 366.0))

    model = joblib.load(args.model_path)
    metadata = json.loads(Path(args.metadata_path).read_text(encoding="utf-8"))
    source_status = model.source_status(feature_row)
    if not source_status["usable"]:
        raise RuntimeError(
            f"RCSS forecast requires at least {source_status['minimum_required']} NWP sources; "
            f"available={source_status['available_models']}, errors={nwp_errors}"
        )
    blend = blend_nwp_features(feature_row, model.nwp_weights, prefixes=model.nwp_prefixes)
    if forecast_mode == "early_nwp_residual":
        residual_row = residual_feature_row(model, feature_row, blend)
        current_max = float(residual_row["current_metar_max_c"])
        residual_row["nwp_model_minus_current_max_c"] = float(residual_row["model_tmax_c"] - current_max)
        residual_row["nwp_future_minus_current_max_c"] = float(
            residual_row["model_future_temp_max_c"] - current_max
        )
        distribution = project_unimodal_distribution(model.ensemble.residual_distribution(residual_row))
    else:
        distribution = model.predict_distribution(feature_row)
    payload = {
        "airport": AIRPORT,
        "city": "Taipei",
        "target_date_local": target.isoformat(),
        "timezone": TIMEZONE,
        "issue_time_utc": issue.isoformat(),
        "issue_time_local": issue.astimezone(ZoneInfo(TIMEZONE)).isoformat(),
        "update_trigger": args.update_trigger,
        "forecast_mode": forecast_mode,
        "actual_local_issue_hour": actual_local_hour,
        "model_local_issue_hour": model_local_hour,
        "model_version": model.model_version,
        "forecast": distribution.to_payload(),
        "latest_metar_record": {
            "observation_time_utc": feature_row.get("latest_metar_time_utc"),
            "temperature_c": feature_row.get("latest_metar_temp_c"),
            "current_max_c": feature_row.get("current_metar_max_c"),
            "raw_metar": feature_row.get("latest_metar_raw"),
        },
        "nwp": {
            "production_source": model.residual_nwp_prefix or "weighted_multi_nwp",
            "weights": model.nwp_weights,
            "available_models": source_status["available_models"],
            "degraded": source_status["degraded"],
            "errors": nwp_errors,
            "individual_tmax_c": {
                prefix: feature_row.get(f"{prefix}_tmax_c") for prefix in model.nwp_prefixes
            },
            "blend_tmax_c": blend["model_tmax_c"],
            "spread_c": blend["nwp_tmax_spread_c"],
            "contract": "Open-Meteo Previous Runs, fixed 24-hour lead",
        },
        "refresh": refresh,
        "walk_forward_metrics": metadata.get("walk_forward_metrics"),
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    text = format_forecast_message(payload)
    if args.notify:
        payload["telegram"] = notify_if_configured(text)
    _write_report(payload, args.report_path)
    print(json.dumps(payload, indent=2, default=str))


def load_live_nwp_features(
    target: date,
    issue: datetime,
    *,
    cache_dir: Path,
    model_local_hour: int | None = None,
) -> tuple[dict, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(RCSS_NWP_MODELS)) as pool:
        futures = {
            pool.submit(_load_one_model, model, prefix, target, issue, cache_dir, model_local_hour): (model, prefix)
            for model, prefix in RCSS_NWP_MODELS.items()
        }
        for future in as_completed(futures):
            model, prefix = futures[future]
            try:
                results[prefix] = future.result()
            except Exception as exc:
                errors[model] = str(exc)
    features = {}
    for prefix in RCSS_NWP_MODELS.values():
        features.update(results.get(prefix, {}))
    return features, errors


def _load_one_model(
    model: str,
    prefix: str,
    target: date,
    issue: datetime,
    cache_dir: Path,
    model_local_hour: int | None,
) -> dict:
    hourly = fetch_previous_day1_hourly(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        model=model,
        start_date=(target - pd.Timedelta(days=1)).isoformat(),
        end_date=target.isoformat(),
        cache_dir=cache_dir,
        chunk_days=2,
    )
    local_hour = model_local_hour if model_local_hour is not None else issue.astimezone(ZoneInfo(TIMEZONE)).hour
    snapshots = build_previous_day1_snapshots(
        hourly,
        timezone_name=TIMEZONE,
        issue_hours=[local_hour],
        prefix=prefix,
    )
    row = snapshots[snapshots["target_date_local"].eq(target.isoformat())]
    if row.empty:
        raise RuntimeError(f"No complete {model} D-1 trajectory for {target}")
    return row.iloc[-1].to_dict()


def format_forecast_message(payload: dict) -> str:
    forecast = payload["forecast"]
    metar = payload["latest_metar_record"]
    nwp = payload["nwp"]
    local_issue = pd.Timestamp(payload["issue_time_local"]).strftime("%d.%m.%Y %H:%M")
    metar_time = pd.Timestamp(metar["observation_time_utc"]).tz_convert(TIMEZONE).strftime("%d.%m.%Y %H:%M")
    trigger = "новый METAR" if payload["update_trigger"] == "new_metar" else "плановый выпуск"
    mode_labels = {
        "trained_intraday": "основной обученный intraday-режим",
        "late_clamped_intraday": "поздний режим: временной профиль 20:00",
        "early_nwp_residual": "ранний режим: консервативная NWP-residual PMF",
    }
    lines = [
        "<b>RCSS Taipei Tmax forecast</b>",
        f"Дата прогноза: <b>{html.escape(payload['target_date_local'])}</b>",
        f"Выпуск: <b>{local_issue} по Тайбэю</b>",
        "",
        "<b>Источник обновления</b>",
        f"Триггер: {trigger}",
        f"Режим: {mode_labels.get(payload.get('forecast_mode'), payload.get('forecast_mode'))}",
        "",
        "<b>Использованный METAR</b>",
        f"Время: {metar_time} по Тайбэю",
        f"Температура в сыром METAR: {float(metar['temperature_c']):.1f} °C",
        f"Текущий максимум: {float(metar['current_max_c']):.1f} °C",
        f"Сырая строка: <code>{html.escape(str(metar.get('raw_metar') or 'нет данных'))}</code>",
        "",
        "<b>Production</b>",
        f"Модель: <code>{html.escape(payload['model_version'])}</code>",
        f"Ожидаемый Tmax: <b>{float(forecast['expected_tmax_c']):.1f} °C</b>",
        f"Самый вероятный Tmax: <b>{int(forecast['most_likely_integer_c']):+d} °C</b>",
        f"Интервал 80%: {forecast['intervals']['80'][0]:.0f}...{forecast['intervals']['80'][1]:.0f} °C",
        "Вероятности:",
    ]
    for value, probability in sorted(forecast["probabilities_by_integer_c"].items(), key=lambda item: int(item[0])):
        if float(probability) >= 0.01:
            lines.append(f"{int(value):+d} °C: <b>{100 * float(probability):.1f}%</b>")
    lines.extend(["", "<b>NWP D-1</b>"])
    lines.append("Production anchor: JMA GSM")
    lines.append("Supporting aggregate: available multi-NWP sources")
    labels = {
        "jma_msm": "JMA MSM",
        "jma_gsm": "JMA GSM",
        "icon_global": "ICON Global",
        "ecmwf": "ECMWF IFS",
    }
    for prefix, value in nwp["individual_tmax_c"].items():
        if value is not None and pd.notna(value):
            lines.append(f"{labels[prefix]}: {float(value):+.1f} °C")
    lines.append(f"Диагностический blend Tmax: {float(nwp['blend_tmax_c']):+.1f} °C")
    lines.append(f"Разброс моделей: {float(nwp['spread_c']):.1f} °C")
    if nwp["degraded"]:
        lines.append(f"Статус источников: ограниченный ({', '.join(nwp['available_models'])})")
    return "\n".join(lines)


def forecast_mode_for_hour(local_hour: int) -> tuple[int, str]:
    if local_hour < TRAINED_LOCAL_HOUR_MIN:
        return TRAINED_LOCAL_HOUR_MIN, "early_nwp_residual"
    if local_hour > TRAINED_LOCAL_HOUR_MAX:
        return TRAINED_LOCAL_HOUR_MAX, "late_clamped_intraday"
    return int(local_hour), "trained_intraday"


def residual_feature_row(model, feature_row: dict, blend: dict) -> dict:
    row = dict(feature_row)
    row.update(blend)
    if model.residual_nwp_prefix:
        prefix = model.residual_nwp_prefix
        row["model_tmax_c"] = float(row[f"{prefix}_tmax_c"])
        row["model_future_temp_max_c"] = float(row[f"{prefix}_future_temp_max_c"])
    return row


def _write_report(payload: dict, report_path: str | Path) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, default=str, ensure_ascii=False) + "\n")


def _issue_time(value: str) -> datetime:
    if value == "now":
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live RCSS multi-NWP METAR Tmax forecast.")
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--update-trigger", choices=["scheduled_forecast", "new_metar"], default="scheduled_forecast")
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--metadata-path", default=str(METADATA_PATH))
    parser.add_argument("--report-path", default=str(REPORT_PATH))
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    main()
