from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from weather_tmax_bot.bot.forecast_log import log_forecast
from weather_tmax_bot.data.nwp import NWPArchive
from weather_tmax_bot.data.open_meteo import fetch_open_meteo_live_extract
from weather_tmax_bot.features.metar_upside_dataset import build_current_metar_upside_features
from weather_tmax_bot.features.nwp_features import build_nwp_features
from weather_tmax_bot.models.metar_intraday_survival import apply_metar_intraday_survival_layer
from weather_tmax_bot.notifications.telegram import notify_if_configured
from weather_tmax_bot.operations.refresh import refresh_awc_live
from weather_tmax_bot.utils.time import parse_issue_time, to_local_date


AIRPORT = "LFPB"
TIMEZONE = "Europe/Paris"
LATITUDE = 48.969444
LONGITUDE = 2.441389
MODEL_PATH = Path("data/models/lfpb_metar_tmax_upside_v1.joblib")
METADATA_PATH = Path("data/models/lfpb_metar_tmax_upside_v1.metadata.json")
LIVE_NWP_PATH = Path("data/forecasts/open_meteo_archive_LFPB.parquet")
HISTORICAL_NWP_PATH = Path("data/forecasts/open_meteo_single_runs_icon_d2_LFPB.parquet")
SURVIVAL_DATASET_PATH = Path("data/processed/metar_upside_dataset_LFPB_icon_d2.parquet")


def main() -> None:
    args = _parse_args()
    refresh_summary = None
    issue_is_now = args.issue_time in (None, "now")
    if args.auto_refresh and issue_is_now:
        refresh_summary = {"awc": refresh_awc_live(args.airport)}
        if args.refresh_nwp:
            refresh_summary["open_meteo_nwp"] = _refresh_open_meteo_live(args.airport, None)
    issue_time_utc = parse_issue_time(args.issue_time)
    target_date = date.fromisoformat(args.target_date) if args.target_date else to_local_date(issue_time_utc, TIMEZONE)
    if args.auto_refresh and not issue_is_now:
        refresh_summary = {"awc": refresh_awc_live(args.airport)}
        if args.refresh_nwp:
            refresh_summary["open_meteo_nwp"] = _refresh_open_meteo_live(args.airport, target_date)
    metar = _load_metar(args.airport)
    model = joblib.load(args.model_path)
    metadata = _load_json(args.metadata_path)
    feature_row = build_current_metar_upside_features(
        metar,
        airport_icao=args.airport,
        target_date_local=target_date,
        issue_time_utc=issue_time_utc,
        timezone_name=TIMEZONE,
    )
    nwp_features = _load_nwp_features(target_date, issue_time_utc)
    _add_nwp_relative_features(nwp_features, feature_row)
    feature_row.update(nwp_features)
    feature_row["max_feature_knowledge_time_utc"] = _max_timestamp_string(
        feature_row.get("max_feature_knowledge_time_utc"),
        feature_row.get("max_nwp_knowledge_time_utc"),
    )
    if hasattr(model, "residuals_by_hour") and pd.isna(feature_row.get("model_tmax_c")):
        raise FileNotFoundError(
            "ICON-aware LFPB METAR Tmax model requires NWP features; "
            "run with --auto-refresh --refresh-nwp or provide an Open-Meteo archive."
        )
    base_distribution = model.predict_distribution(feature_row)
    survival_adjustment = apply_metar_intraday_survival_layer(
        base_distribution,
        feature_row,
        historical_dataset_path=SURVIVAL_DATASET_PATH,
    )
    distribution = survival_adjustment.distribution
    forecast_id = None
    if args.log:
        forecast_id = log_forecast(
            airport=args.airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date,
            distribution=distribution,
            feature_snapshot={
                **feature_row,
                "data_sources_used": ["awc.metar.live.LFPB", "iem.metar.archive.LFPB", feature_row.get("latest_nwp_source_id")],
                "target": "METAR_Tmax",
                "model_family": "metar_tmax_remaining_upside",
                "intraday_survival_layer": survival_adjustment.details,
                "base_forecast_before_intraday_survival": base_distribution.to_payload(),
            },
            model_version=metadata.get("model_version", "lfpb_metar_tmax_upside_v1"),
        )
    payload = {
        "forecast_id": forecast_id,
        "airport": args.airport,
        "target": "METAR_Tmax",
        "target_description": "daily maximum temperature reported by METAR",
        "target_date_local": target_date.isoformat(),
        "timezone": TIMEZONE,
        "issue_time_utc": issue_time_utc.isoformat(),
        "model_version": metadata.get("model_version", "lfpb_metar_tmax_upside_v1"),
        "calibration": (metadata.get("calibration_metadata") or {}).get("calibration_method", "unknown"),
        "calibration_attached": _calibration_attached(model),
        "forecast": distribution.to_payload(),
        "base_forecast_before_intraday_survival": base_distribution.to_payload(),
        "intraday_survival_layer": survival_adjustment.details,
        "metar_signal": {
            "latest_metar_time_utc": feature_row.get("latest_metar_time_utc"),
            "latest_metar_temp_c": feature_row.get("latest_metar_temp_c"),
            "current_metar_max_c": feature_row.get("current_metar_max_c"),
            "drop_from_current_max_c": feature_row.get("drop_from_current_max_c"),
            "metar_count_so_far": feature_row.get("metar_count_so_far"),
            "temp_trend_1h": feature_row.get("temp_trend_1h"),
            "temp_trend_3h": feature_row.get("temp_trend_3h"),
            "has_rain_recent_metar": feature_row.get("has_rain_recent_metar"),
            "latest_metar_raw": feature_row.get("latest_metar_raw"),
        },
        "data_lineage": {
            "max_feature_knowledge_time_utc": feature_row.get("max_feature_knowledge_time_utc"),
            "source": "AWC live METAR if refreshed; local AWC/IEM METAR archive fallback",
            "latest_nwp_source_id": feature_row.get("latest_nwp_source_id"),
            "max_nwp_knowledge_time_utc": feature_row.get("max_nwp_knowledge_time_utc"),
            "model_tmax_c": feature_row.get("model_tmax_c"),
            "model_future_temp_max_c": feature_row.get("model_future_temp_max_c"),
            "leakage_check_passed": feature_row.get("leakage_check_passed"),
        },
        "refresh_summary": refresh_summary,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    text = _format_message(payload)
    print(text)
    print(f"\nWrote {report_path}")
    if args.notify:
        result = notify_if_configured(text)
        print(json.dumps({"telegram": result}, indent=2, default=str))


def _load_metar(airport: str) -> pd.DataFrame:
    frames = []
    for path in [
        Path(f"data/forecasts/awc_metar_live_{airport}.parquet"),
        Path(f"data/interim/metar_iem_{airport}.parquet"),
    ]:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No METAR data found for {airport}; run with --auto-refresh or download historical METAR first")
    frame = pd.concat(frames, ignore_index=True)
    if "raw_record_hash" in frame.columns:
        frame = frame.drop_duplicates(subset=["raw_record_hash"], keep="last")
    else:
        frame = frame.drop_duplicates(subset=["observation_time_utc", "raw_metar"], keep="last")
    return frame


def _refresh_open_meteo_live(airport: str, target_date_local: date | None) -> dict:
    target = target_date_local or date.today()
    rows = fetch_open_meteo_live_extract(
        airport_icao=airport,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        target_date_local=target,
        timezone_name=TIMEZONE,
    )
    if rows.empty:
        return {"rows_fetched": 0, "archive_rows": _parquet_rows(LIVE_NWP_PATH)}
    NWPArchive(LIVE_NWP_PATH).append_extract(rows)
    return {"rows_fetched": len(rows), "archive_rows": _parquet_rows(LIVE_NWP_PATH)}


def _load_nwp_features(target_date_local: date, issue_time_utc) -> dict:
    frames = []
    for path in [LIVE_NWP_PATH, HISTORICAL_NWP_PATH]:
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return {"nwp_missing": True, "model_tmax_c": None}
    frame = pd.concat(frames, ignore_index=True)
    if "airport_icao" in frame.columns:
        frame = frame[frame["airport_icao"].fillna(AIRPORT) == AIRPORT].copy()
    frame = frame[frame["target_date_local"].astype(str) == target_date_local.isoformat()].copy()
    return build_nwp_features(frame, issue_time_utc)


def _add_nwp_relative_features(nwp_features: dict, metar_features: dict) -> None:
    model_tmax = nwp_features.get("model_tmax_c")
    current_max = metar_features.get("current_metar_max_c")
    future = nwp_features.get("model_future_temp_max_c")
    nwp_features["nwp_model_minus_current_max_c"] = (
        None if pd.isna(model_tmax) or pd.isna(current_max) else float(model_tmax) - float(current_max)
    )
    nwp_features["nwp_future_minus_current_max_c"] = (
        None if pd.isna(future) or pd.isna(current_max) else float(future) - float(current_max)
    )


def _parquet_rows(path: Path) -> int:
    return 0 if not path.exists() else len(pd.read_parquet(path))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _calibration_attached(model) -> bool:
    direct = getattr(model, "calibrator", None)
    if direct is not None:
        return True
    base = getattr(model, "base_model", None)
    if getattr(base, "calibrator", None) is not None:
        return True
    ml_model = getattr(model, "ml_model", None)
    return bool(getattr(ml_model, "calibrator", None))


def _format_message(payload: dict) -> str:
    forecast = payload["forecast"]
    signal = payload["metar_signal"]
    survival = payload.get("intraday_survival_layer") or {}
    thresholds = forecast["threshold_probabilities"]
    bins = {
        int(bin_c): float(probability)
        for bin_c, probability in forecast["probabilities_by_integer_c"].items()
        if float(probability) >= 0.01
    }
    bin_text = "\n".join(f"{bin_c:+d} °C: <b>{probability:.1%}</b>" for bin_c, probability in sorted(bins.items()))
    if not bin_text:
        bin_text = "Нет корзин выше 1%."
    return "\n".join(
        [
            f"<b>Прогноз METAR Tmax: {payload['airport']}</b>",
            f"Дата: <b>{payload['target_date_local']}</b>",
            f"Цель: максимум температуры, который покажут METAR за день",
            f"Выпуск UTC: <code>{payload['issue_time_utc']}</code>",
            f"ID прогноза: <code>{payload.get('forecast_id') or 'не логировался'}</code>",
            "",
            "<b>Температурный прогноз</b>",
            f"Ожидаемый METAR Tmax: <b>{forecast['expected_tmax_c']:.1f} °C</b>",
            f"Медиана: {forecast['median_tmax_c']:.1f} °C",
            f"Самая вероятная корзина: <b>{forecast['most_likely_integer_c']} °C</b>",
            f"Интервал 80%: {forecast['intervals']['80'][0]:.1f}...{forecast['intervals']['80'][1]:.1f} °C",
            "",
            "<b>Вероятности по градусам</b>",
            bin_text,
            "",
            "<b>Вероятности событий</b>",
            f"Не ниже +20 °C: {thresholds['ge_20']:.1%}",
            f"Не ниже +25 °C: {thresholds['ge_25']:.1%}",
            f"Не ниже +30 °C: {thresholds['ge_30']:.1%}",
            "",
            "<b>METAR-сигнал</b>",
            f"Последняя температура: {float(signal['latest_metar_temp_c']):.1f} °C",
            f"Текущий максимум по METAR: {float(signal['current_metar_max_c']):.1f} °C",
            f"Падение от максимума: {float(signal['drop_from_current_max_c']):.1f} °C",
            f"METAR за день: {int(signal['metar_count_so_far'])}",
            f"Тренд 1ч: {_fmt_float(signal.get('temp_trend_1h'))} °C",
            f"Тренд 3ч: {_fmt_float(signal.get('temp_trend_3h'))} °C",
            f"Дождь недавно: {'да' if signal.get('has_rain_recent_metar') else 'нет'}",
            "",
            "<b>Intraday survival</b>",
            f"Слой активен: {'да' if survival.get('active') else 'нет'}",
            f"Шанс роста минимум на +1 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_1c'))}",
            f"Шанс роста минимум на +2 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_2c'))}",
            f"Шанс роста минимум на +3 °C: {_fmt_percent(survival.get('adjusted_probability_upside_ge_3c'))}",
            f"До коррекции +1 °C: {_fmt_percent(survival.get('original_probability_upside_ge_1c'))}",
            f"Вес коррекции: {_fmt_percent(survival.get('effective_strength'))}",
            "",
            "<b>Калибровка</b>",
            f"Статус: {'включена' if payload.get('calibration_attached') else 'не включена'}",
            f"Метод: <code>{payload.get('calibration')}</code>",
            "",
            "<b>Данные</b>",
            f"Последний METAR: <code>{signal.get('latest_metar_time_utc')}</code>",
            f"ICON-D2 Tmax: {_fmt_float(payload['data_lineage'].get('model_tmax_c'))} °C",
            f"ICON-D2 future max: {_fmt_float(payload['data_lineage'].get('model_future_temp_max_c'))} °C",
            f"Max knowledge time: <code>{payload['data_lineage'].get('max_feature_knowledge_time_utc')}</code>",
            f"Leakage check: {'ok' if payload['data_lineage'].get('leakage_check_passed') else 'failed'}",
            "",
            f"<code>{signal.get('latest_metar_raw')}</code>",
        ]
    )


def _fmt_float(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):+.1f}"


def _fmt_percent(value) -> str:
    if value is None or pd.isna(value):
        return "н/д"
    return f"{float(value):.1%}"


def _max_timestamp_string(*values) -> str | None:
    stamps = [pd.Timestamp(value) for value in values if value is not None and not pd.isna(value)]
    if not stamps:
        return None
    return max(stamps).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict LFPB daily METAR Tmax distribution.")
    parser.add_argument("--airport", default=AIRPORT)
    parser.add_argument("--target-date", default=None)
    parser.add_argument("--issue-time", default="now")
    parser.add_argument("--auto-refresh", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--refresh-nwp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=METADATA_PATH)
    parser.add_argument("--report-path", default="data/reports/latest_lfpb_metar_tmax_prediction.json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
