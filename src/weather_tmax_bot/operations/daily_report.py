from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from weather_tmax_bot.evaluation.metrics import crps_discrete, nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.notifications.telegram import format_daily_model_report_message, notify_if_configured

LOCAL_TZ = ZoneInfo("Europe/Berlin")
REPORT_VARIANTS = {"production_champion"}


def run_daily_model_report(
    *,
    airport: str = "EDDM",
    target_date_local: date | None = None,
    mode: str = "preliminary_metar",
    notify: bool = True,
    force: bool = False,
    earliest_local_hour: int | None = None,
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    variant_monitoring_path: str | Path = "data/reports/forecast_variant_monitoring.parquet",
    sent_registry_path: str | Path = "data/reports/daily_report_sent.json",
    output_path: str | Path = "data/reports/latest_daily_model_report.json",
) -> dict:
    target = target_date_local or datetime.now(LOCAL_TZ).date()
    if earliest_local_hour is not None and datetime.now(LOCAL_TZ).hour < earliest_local_hour and not force:
        return {
            "status": "not_ready",
            "reason": "before_earliest_local_hour",
            "airport": airport,
            "target_date_local": target.isoformat(),
            "mode": mode,
            "earliest_local_hour": earliest_local_hour,
        }

    report = build_daily_model_report(
        airport=airport,
        target_date_local=target,
        mode=mode,
        forecast_log_path=forecast_log_path,
        variant_monitoring_path=variant_monitoring_path,
    )
    _write_json(output_path, report)
    report["report_path"] = str(output_path)

    dedupe_key = _dedupe_key(airport, target, mode)
    already_sent = dedupe_key in _read_sent_registry(sent_registry_path)
    should_notify = notify and report["status"] == "ok" and (force or not already_sent)
    if should_notify:
        report["telegram_notification"] = notify_if_configured(format_daily_model_report_message(report))
        if (report["telegram_notification"] or {}).get("sent"):
            _mark_sent(sent_registry_path, dedupe_key)
    else:
        report["telegram_notification"] = {
            "sent": False,
            "reason": "notify_disabled_or_already_sent" if notify else "notify_disabled",
            "already_sent": already_sent,
        }
    return report


def build_daily_model_report(
    *,
    airport: str,
    target_date_local: date,
    mode: str,
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    variant_monitoring_path: str | Path = "data/reports/forecast_variant_monitoring.parquet",
) -> dict:
    if mode not in {"preliminary_metar", "dwd_final"}:
        raise ValueError("mode must be preliminary_metar or dwd_final")
    if mode == "dwd_final":
        return _build_final_report(
            airport=airport,
            target_date_local=target_date_local,
            variant_monitoring_path=variant_monitoring_path,
        )
    return _build_preliminary_report(
        airport=airport,
        target_date_local=target_date_local,
        forecast_log_path=forecast_log_path,
        requested_mode=mode,
    )


def _build_final_report(*, airport: str, target_date_local: date, variant_monitoring_path: str | Path) -> dict:
    path = Path(variant_monitoring_path)
    if not path.exists():
        return _empty_report(airport, target_date_local, "dwd_final", "variant_monitoring_missing")
    frame = pd.read_parquet(path)
    if frame.empty:
        return _empty_report(airport, target_date_local, "dwd_final", "variant_monitoring_empty")
    rows = frame[
        (frame["airport"].astype(str) == airport)
        & (frame["target_date_local"].astype(str) == target_date_local.isoformat())
    ].copy()
    if "forecast_variant" in rows.columns:
        rows = rows[rows["forecast_variant"].isin(REPORT_VARIANTS)].copy()
    if rows.empty:
        return _empty_report(airport, target_date_local, "dwd_final", "no_scored_rows_for_date")
    actual = float(rows["actual_tmax_c"].dropna().iloc[0])
    return _report_from_scored_rows(
        airport=airport,
        target_date_local=target_date_local,
        mode="dwd_final",
        truth_source="DWD 10-minute truth",
        actual_tmax_c=actual,
        rows=rows,
    )


def _build_preliminary_report(
    *, airport: str, target_date_local: date, forecast_log_path: str | Path, requested_mode: str
) -> dict:
    records = [
        record
        for record in _iter_forecast_log(Path(forecast_log_path))
        if record.get("airport") == airport and record.get("target_date_local") == target_date_local.isoformat()
    ]
    if not records:
        return _empty_report(airport, target_date_local, requested_mode, "no_forecasts_for_date")
    actual = _preliminary_metar_max(records)
    if actual is None:
        return _empty_report(airport, target_date_local, requested_mode, "no_metar_max_in_forecast_log")
    variant_rows = []
    for record in records:
        variant_rows.extend(_score_record_variants(record, actual))
    scored = pd.DataFrame(variant_rows)
    if scored.empty:
        return _empty_report(airport, target_date_local, requested_mode, "no_variant_distributions")
    return _report_from_scored_rows(
        airport=airport,
        target_date_local=target_date_local,
        mode="preliminary_metar",
        truth_source="operational METAR max from forecast log",
        actual_tmax_c=actual,
        rows=scored,
    )


def _report_from_scored_rows(
    *,
    airport: str,
    target_date_local: date,
    mode: str,
    truth_source: str,
    actual_tmax_c: float,
    rows: pd.DataFrame,
) -> dict:
    summary = _summary_by_variant(rows)
    best = _select_variant(summary, best=True)
    worst = _select_variant(summary, best=False)
    return {
        "status": "ok",
        "airport": airport,
        "target_date_local": target_date_local.isoformat(),
        "mode": mode,
        "truth_source": truth_source,
        "actual_tmax_c": actual_tmax_c,
        "forecast_count": int(rows["forecast_id"].nunique()) if "forecast_id" in rows else int(len(rows)),
        "variant_record_count": int(len(rows)),
        "summary_by_variant": summary,
        "hourly_comparison": _hourly_comparison(rows),
        "best_variant": best,
        "worst_variant": worst,
        "analysis": _analysis_text(summary, best, worst, mode),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _summary_by_variant(rows: pd.DataFrame) -> list[dict]:
    grouped = []
    total_forecasts = max(1, int(rows["forecast_id"].nunique())) if "forecast_id" in rows else max(1, len(rows))
    for variant, group in rows.groupby("forecast_variant", dropna=False):
        errors = group["error_expected_c"].astype(float)
        grouped.append(
            {
                "forecast_variant": str(variant),
                "rows": int(len(group)),
                "coverage_ratio": float(len(group) / total_forecasts),
                "mae_expected": float(errors.abs().mean()),
                "bias_expected": float(errors.mean()),
                "mean_nll": float(group["nll"].astype(float).mean()),
                "mean_crps": float(group["crps"].astype(float).mean()),
                "mean_probability_actual_integer_bin": float(
                    group["probability_actual_integer_bin"].astype(float).mean()
                ),
                "mean_probability_above_actual_integer_bin": float(
                    group["probability_above_actual_integer_bin"].astype(float).mean()
                ),
                "first_issue_time_utc": str(group["issue_time_utc"].min()) if "issue_time_utc" in group else None,
                "last_issue_time_utc": str(group["issue_time_utc"].max()) if "issue_time_utc" in group else None,
            }
        )
    return sorted(grouped, key=lambda item: (item["mae_expected"], item["mean_crps"], item["mean_nll"]))


def _select_variant(summary: list[dict], *, best: bool) -> dict | None:
    candidates = [item for item in summary if item["coverage_ratio"] >= 0.5] or summary
    if not candidates:
        return None
    key = lambda item: (item["mae_expected"], item["mean_crps"], item["mean_nll"])
    return (min if best else max)(candidates, key=key)


def _analysis_text(summary: list[dict], best: dict | None, worst: dict | None, mode: str) -> list[str]:
    lines = []
    if mode == "preliminary_metar":
        lines.append("Это предварительный отчет: факт взят из оперативного METAR, финальная DWD-оценка может немного отличаться.")
    else:
        lines.append("Это финальный отчет по DWD truth: его можно использовать для накопления статистики качества.")
    if best:
        lines.append(
            f"Лучше всего по среднему MAE сейчас выглядит {best['forecast_variant']} "
            f"({best['mae_expected']:.2f} °C, покрытие {best['coverage_ratio']:.0%})."
        )
    if worst:
        lines.append(
            f"Слабее всего выглядела {worst['forecast_variant']} "
            f"({worst['mae_expected']:.2f} °C, bias {worst['bias_expected']:+.2f} °C)."
        )
    for item in summary:
        if item["forecast_variant"] == "base_prior" and item["bias_expected"] > 0.75:
            lines.append("Базовый NWP-prior заметно завышал температуру; intraday-коррекции были полезны.")
        if item["forecast_variant"] == "shadow_safe_blend" and item["coverage_ratio"] < 0.9:
            lines.append("Safe blend выглядит перспективно, но покрывал не весь день, поэтому его нельзя переоценивать по одному отчету.")
    return lines


def _score_record_variants(record: dict, actual: float) -> list[dict]:
    rows = []
    champion = _distribution_from_probabilities(record.get("probability_distribution") or {})
    if champion is not None:
        rows.append(_score_distribution(record, actual, "production_champion", champion))
    metadata = record.get("raw_input_metadata", {}) or {}
    for name, payload in (metadata.get("forecast_variants", {}) or {}).items():
        if name == "production_champion":
            continue
        if name not in REPORT_VARIANTS:
            continue
        dist = _variant_distribution(payload)
        if dist is not None:
            rows.append(_score_distribution(record, actual, str(name), dist))
    return rows


def _score_distribution(record: dict, actual: float, variant: str, dist: TmaxDistribution) -> dict:
    actual_bin = int(round(actual))
    issue_time = record.get("issue_time_utc")
    local_hour = _local_issue_hour(issue_time)
    return {
        "forecast_id": record.get("forecast_id"),
        "airport": record.get("airport"),
        "target_date_local": record.get("target_date_local"),
        "issue_time_utc": record.get("issue_time_utc"),
        "local_issue_hour": local_hour,
        "forecast_variant": variant,
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "error_expected_c": dist.expected_tmax_c - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "probability_actual_integer_bin": float(dist.probabilities[dist.bins_c == actual_bin].sum()),
        "probability_above_actual_integer_bin": float(dist.probabilities[dist.bins_c > actual_bin].sum()),
    }


def _hourly_comparison(rows: pd.DataFrame) -> list[dict]:
    if rows.empty or "local_issue_hour" not in rows.columns:
        return []
    frame = rows.copy()
    frame["local_hour_floor"] = pd.to_numeric(frame["local_issue_hour"], errors="coerce").floordiv(1).astype("Int64")
    frame = frame[frame["local_hour_floor"].notna()]
    out = []
    for hour, group in frame.groupby("local_hour_floor"):
        variants = _summary_by_variant(group)
        if not variants:
            continue
        out.append(
            {
                "local_hour": int(hour),
                "best_variant": variants[0]["forecast_variant"],
                "variants": variants,
            }
        )
    return sorted(out, key=lambda item: item["local_hour"])


def _local_issue_hour(value) -> float | None:
    if not value:
        return None
    try:
        parsed = pd.Timestamp(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize("UTC")
        local = parsed.tz_convert(LOCAL_TZ)
        return float(local.hour + local.minute / 60)
    except (TypeError, ValueError):
        return None


def _variant_distribution(payload: dict) -> TmaxDistribution | None:
    distribution = (payload or {}).get("distribution") or payload
    return _distribution_from_probabilities((distribution or {}).get("probabilities_by_integer_c") or {})


def _distribution_from_probabilities(probabilities: dict) -> TmaxDistribution | None:
    if not probabilities:
        return None
    try:
        return TmaxDistribution([int(key) for key in probabilities], [float(value) for value in probabilities.values()])
    except (TypeError, ValueError):
        return None


def _preliminary_metar_max(records: list[dict]) -> float | None:
    values = []
    for record in records:
        metadata = record.get("raw_input_metadata", {}) or {}
        components = metadata.get("forecast_components", {}) or {}
        intraday = components.get("intraday_update", {}) or {}
        latest_metar = metadata.get("latest_metar_record") or {}
        for value in (
            intraday.get("observed_max_so_far_c"),
            intraday.get("last_metar_temp_c"),
            metadata.get("observed_max_so_far_from_metar"),
            metadata.get("last_metar_temp_c"),
            latest_metar.get("temperature_c"),
        ):
            if value is not None:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
    return max(values) if values else None


def _iter_forecast_log(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _empty_report(airport: str, target_date_local: date, mode: str, reason: str) -> dict:
    return {
        "status": "no_data",
        "reason": reason,
        "airport": airport,
        "target_date_local": target_date_local.isoformat(),
        "mode": mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _read_sent_registry(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(payload.get("sent", []))


def _mark_sent(path: str | Path, key: str) -> None:
    sent = _read_sent_registry(path)
    sent.add(key)
    _write_json(path, {"sent": sorted(sent)})


def _dedupe_key(airport: str, target_date_local: date, mode: str) -> str:
    return f"{airport}:{target_date_local.isoformat()}:{mode}"


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
