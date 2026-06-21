from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


AIRPORT_TIMEZONES = {
    "EDDM": "Europe/Berlin",
    "LFPB": "Europe/Paris",
}

DEFAULT_WINDOWS_DAYS = (7, 14, 30)
DEFAULT_START_HOUR = 10.0
DEFAULT_END_HOUR = 17.0
TEMP_RE = r"([-+]?[0-9]+(?:[\.,][0-9]+)?)\s*\u00b0C"


@dataclass(frozen=True)
class LiveBaselineReport:
    rows: pd.DataFrame
    overall: pd.DataFrame
    by_phase: pd.DataFrame
    by_hour: pd.DataFrame
    by_day: pd.DataFrame
    rolling: pd.DataFrame
    alerts: pd.DataFrame
    bias_audit: pd.DataFrame
    metadata: dict


def build_live_baseline_report(
    monitoring: pd.DataFrame,
    *,
    start_hour: float = DEFAULT_START_HOUR,
    end_hour: float = DEFAULT_END_HOUR,
    windows_days: tuple[int, ...] = DEFAULT_WINDOWS_DAYS,
) -> LiveBaselineReport:
    rows = prepare_scored_rows(monitoring)
    rows = rows[rows["local_issue_hour"].between(start_hour, end_hour, inclusive="both")].copy()
    rows = rows[rows["error_expected_c"].notna()].copy()
    if rows.empty:
        metadata = _metadata(monitoring, rows, start_hour, end_hour, windows_days)
        empty = pd.DataFrame()
        return LiveBaselineReport(empty, empty, empty, empty, empty, empty, empty, empty, metadata)

    rows["phase"] = rows["local_issue_hour"].map(_phase)
    rows["local_hour"] = rows["local_issue_hour"].map(lambda value: int(math.floor(float(value))))
    rows["abs_error_expected_c"] = rows["error_expected_c"].abs()
    rows["abs_error_median_c"] = rows["error_median_c"].abs() if "error_median_c" in rows else pd.NA
    if "most_likely_integer_c" in rows and "actual_tmax_c" in rows:
        rows["mode_error_c"] = pd.to_numeric(rows["most_likely_integer_c"], errors="coerce") - pd.to_numeric(
            rows["actual_tmax_c"], errors="coerce"
        )
        rows["abs_error_mode_c"] = rows["mode_error_c"].abs()
        rows["mode_hit"] = rows["mode_error_c"].round().eq(0)
    else:
        rows["mode_error_c"] = pd.NA
        rows["abs_error_mode_c"] = pd.NA
        rows["mode_hit"] = pd.NA
    rows["within_1c_expected"] = rows["abs_error_expected_c"] <= 1.0
    rows["large_error_gt_2c"] = rows["abs_error_expected_c"] > 2.0

    overall = _summarize(rows, ["airport", "model_version"])
    by_phase = _summarize(rows, ["airport", "model_version", "phase"])
    by_hour = _summarize(rows, ["airport", "model_version", "local_hour"])
    by_day = _summarize(rows, ["airport", "target_date_local"])
    rolling = _rolling_summaries(rows, windows_days)
    alerts = _alerts(rows, rolling)
    bias_audit = build_bias_correction_audit(rows)
    metadata = _metadata(monitoring, rows, start_hour, end_hour, windows_days)
    return LiveBaselineReport(rows, overall, by_phase, by_hour, by_day, rolling, alerts, bias_audit, metadata)


def build_bias_correction_audit(
    rows: pd.DataFrame,
    *,
    min_rows: int = 20,
    min_distinct_days: int = 7,
    min_abs_bias_c: float = 0.25,
    min_mae_improvement_c: float = 0.05,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    required = {"airport", "model_version", "phase", "error_expected_c", "expected_tmax_c", "actual_tmax_c"}
    if not required.issubset(rows.columns):
        return pd.DataFrame()
    audit = _bias_summary(rows, ["airport", "model_version", "phase"])
    if audit.empty:
        return audit
    audit["recommendation"] = audit.apply(
        lambda row: _bias_recommendation(
            row,
            min_rows=min_rows,
            min_distinct_days=min_distinct_days,
            min_abs_bias_c=min_abs_bias_c,
            min_mae_improvement_c=min_mae_improvement_c,
        ),
        axis=1,
    )
    return audit.sort_values(["airport", "phase", "model_version"]).reset_index(drop=True)


def prepare_scored_rows(monitoring: pd.DataFrame) -> pd.DataFrame:
    if monitoring.empty:
        return monitoring.copy()
    df = monitoring.copy()
    if "airport" not in df.columns:
        raise ValueError("monitoring must contain airport")
    if "target_date_local" not in df.columns:
        raise ValueError("monitoring must contain target_date_local")
    if "error_expected_c" not in df.columns:
        raise ValueError("monitoring must contain error_expected_c")

    df["target_date_local"] = pd.to_datetime(df["target_date_local"], errors="coerce").dt.date.astype(str)
    if "local_issue_hour" in df.columns:
        df["local_issue_hour"] = pd.to_numeric(df["local_issue_hour"], errors="coerce")
    else:
        df["local_issue_hour"] = pd.NA
    missing_hour = df["local_issue_hour"].isna()
    if missing_hour.any() and "issue_time_utc" in df.columns:
        df.loc[missing_hour, "local_issue_hour"] = [
            _local_hour(issue_time, airport)
            for issue_time, airport in zip(df.loc[missing_hour, "issue_time_utc"], df.loc[missing_hour, "airport"])
        ]
    for column in [
        "actual_tmax_c",
        "expected_tmax_c",
        "median_tmax_c",
        "most_likely_integer_c",
        "error_expected_c",
        "error_median_c",
        "nll",
        "crps",
        "probability_actual_integer_bin",
        "coverage_80",
        "interval_80_width_c",
    ]:
        if column in df.columns and column != "coverage_80":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "coverage_80" in df.columns:
        df["coverage_80"] = df["coverage_80"].astype("boolean")
    return df


def monitoring_from_telegram_exports(paths_by_airport: dict[str, str | Path]) -> pd.DataFrame:
    forecast_rows = []
    observed_max_by_key: dict[tuple[str, str], float] = {}
    for airport, path_value in paths_by_airport.items():
        airport = airport.upper()
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        for message in data.get("messages", []):
            text = _telegram_text(message)
            if airport not in text:
                continue
            target_date = _first_date(text)
            if target_date is None:
                continue
            observed = _observed_max_from_message(text)
            if observed is not None:
                key = (airport, target_date)
                observed_max_by_key[key] = max(observed_max_by_key.get(key, -999.0), observed)
            forecast = _forecast_from_telegram_message(airport, message, text, target_date)
            if forecast is not None:
                forecast_rows.append(forecast)
    for row in forecast_rows:
        actual = observed_max_by_key.get((row["airport"], row["target_date_local"]))
        if actual is None:
            continue
        row["actual_tmax_c"] = actual
        row["error_expected_c"] = row["expected_tmax_c"] - actual
        if pd.notna(row.get("median_tmax_c")):
            row["error_median_c"] = row["median_tmax_c"] - actual
        row["probability_actual_integer_bin"] = row["probabilities_by_integer_c"].get(int(round(actual)), 0.0)
        row["coverage_80"] = _coverage_from_interval(row.get("interval_80_low_c"), row.get("interval_80_high_c"), actual)
        row["interval_80_width_c"] = _interval_width_from_bounds(row.get("interval_80_low_c"), row.get("interval_80_high_c"))
    frame = pd.DataFrame(forecast_rows)
    if frame.empty:
        return frame
    frame = frame[frame["actual_tmax_c"].notna()].copy()
    frame["probabilities_by_integer_c"] = frame["probabilities_by_integer_c"].map(
        lambda item: json.dumps(item, sort_keys=True)
    )
    return frame


def write_live_baseline_report(report: LiveBaselineReport, output_dir: str | Path = "data/reports") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": out / "live_baseline_10_17_rows.parquet",
        "overall": out / "live_baseline_10_17_overall.csv",
        "by_phase": out / "live_baseline_10_17_by_phase.csv",
        "by_hour": out / "live_baseline_10_17_by_hour.csv",
        "by_day": out / "live_baseline_10_17_by_day.csv",
        "rolling": out / "live_baseline_10_17_rolling.csv",
        "alerts": out / "live_baseline_10_17_alerts.csv",
        "bias_audit": out / "live_baseline_10_17_bias_audit.csv",
        "metadata": out / "live_baseline_10_17_metadata.json",
        "markdown": out / "live_baseline_10_17.md",
    }
    report.rows.to_parquet(paths["rows"], index=False)
    report.overall.to_csv(paths["overall"], index=False, encoding="utf-8")
    report.by_phase.to_csv(paths["by_phase"], index=False, encoding="utf-8")
    report.by_hour.to_csv(paths["by_hour"], index=False, encoding="utf-8")
    report.by_day.to_csv(paths["by_day"], index=False, encoding="utf-8")
    report.rolling.to_csv(paths["rolling"], index=False, encoding="utf-8")
    report.alerts.to_csv(paths["alerts"], index=False, encoding="utf-8")
    report.bias_audit.to_csv(paths["bias_audit"], index=False, encoding="utf-8")
    paths["metadata"].write_text(json.dumps(report.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    return {key: str(value) for key, value in paths.items()}


def _telegram_text(message: dict) -> str:
    value = message.get("text", "")
    if isinstance(value, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in value)
    return str(value)


def _first_date(text: str) -> str | None:
    match = re.search(r"(20[0-9]{2}-[0-9]{2}-[0-9]{2})", text)
    return match.group(1) if match else None


def _observed_max_from_message(text: str) -> float | None:
    labels = [
        "\u041d\u0430\u0431\u043b\u044e\u0434\u0430\u0435\u043c\u044b\u0439 \u043c\u0430\u043a\u0441\u0438\u043c\u0443\u043c",
        "\u0422\u0435\u043a\u0443\u0449\u0438\u0439 \u043c\u0430\u043a\u0441\u0438\u043c\u0443\u043c \u043f\u043e METAR",
    ]
    for label in labels:
        match = re.search(re.escape(label) + r":\s*" + TEMP_RE, text)
        if match:
            return _float(match.group(1))
    return None


def _forecast_from_telegram_message(airport: str, message: dict, text: str, target_date: str) -> dict | None:
    if airport == "EDDM":
        if not _looks_like_munich_forecast(text):
            return None
        issue = _munich_issue_time(text)
        model_version = _first_line_value(text, "\u041c\u043e\u0434\u0435\u043b\u044c")
    elif airport == "LFPB":
        if "METAR Tmax: LFPB" not in text:
            return None
        issue = _paris_issue_time(text)
        model_version = "lfpb_telegram_forecast"
    else:
        return None
    if issue is None:
        return None
    expected = _first_temp(text)
    if expected is None:
        return None
    median = _line_temp(text, "\u041c\u0435\u0434\u0438\u0430\u043d\u0430")
    mode = _mode_from_text(text)
    low, high = _interval_80(text)
    probabilities = _probabilities_by_integer(text)
    return {
        "forecast_id": str(message.get("id")),
        "airport": airport,
        "target_date_local": target_date,
        "issue_time_utc": issue.astimezone(ZoneInfo("UTC")).isoformat(),
        "local_issue_hour": issue.hour + issue.minute / 60,
        "model_version": model_version or f"{airport.lower()}_telegram_forecast",
        "expected_tmax_c": expected,
        "median_tmax_c": median,
        "most_likely_integer_c": mode,
        "interval_80_low_c": low,
        "interval_80_high_c": high,
        "probabilities_by_integer_c": probabilities,
    }


def _looks_like_munich_forecast(text: str) -> bool:
    return "EDDM" in text and (
        "\u041f\u0440\u043e\u0433\u043d\u043e\u0437 \u0433\u043e\u0442\u043e\u0432" in text
        or "METAR-\u043e\u0431\u043d\u043e\u0432\u043b\u0435\u043d\u0438\u0435" in text
    )


def _munich_issue_time(text: str) -> datetime | None:
    match = re.search(r"([0-9]{2})\.([0-9]{2})\.([0-9]{4})\s+([0-9]{2}):([0-9]{2})", text)
    if not match:
        return None
    return datetime(
        int(match.group(3)),
        int(match.group(2)),
        int(match.group(1)),
        int(match.group(4)),
        int(match.group(5)),
        tzinfo=ZoneInfo("Europe/Berlin"),
    )


def _paris_issue_time(text: str) -> datetime | None:
    match = re.search(r"UTC:\s*([0-9T:\.\-]+(?:\+00:00|Z))", text)
    if not match:
        return None
    return datetime.fromisoformat(match.group(1).replace("Z", "+00:00")).astimezone(ZoneInfo("Europe/Paris"))


def _first_temp(text: str) -> float | None:
    match = re.search(TEMP_RE, text)
    return _float(match.group(1)) if match else None


def _line_temp(text: str, label: str) -> float | None:
    match = re.search(re.escape(label) + r":\s*" + TEMP_RE, text)
    return _float(match.group(1)) if match else None


def _first_line_value(text: str, label: str) -> str | None:
    match = re.search(re.escape(label) + r":\s*([^\n]+)", text)
    return match.group(1).strip() if match else None


def _mode_from_text(text: str) -> int | None:
    label = "\u0421\u0430\u043c\u0430\u044f \u0432\u0435\u0440\u043e\u044f\u0442\u043d\u0430\u044f \u043a\u043e\u0440\u0437\u0438\u043d\u0430"
    match = re.search(re.escape(label) + r":\s*([-+]?[0-9]+)\s*\u00b0C", text)
    return int(match.group(1)) if match else None


def _interval_80(text: str) -> tuple[float | None, float | None]:
    label = "\u0418\u043d\u0442\u0435\u0440\u0432\u0430\u043b 80%"
    match = re.search(
        re.escape(label) + r":\s*([-+]?[0-9]+(?:[\.,][0-9]+)?)\.\.\.([-+]?[0-9]+(?:[\.,][0-9]+)?)\s*\u00b0C",
        text,
    )
    if not match:
        return None, None
    return _float(match.group(1)), _float(match.group(2))


def _probabilities_by_integer(text: str) -> dict[int, float]:
    return {
        int(degree): _float(probability) / 100.0
        for degree, probability in re.findall(r"\+?([-+]?[0-9]+)\s*\u00b0C:\s*([0-9]+(?:[\.,][0-9]+)?)%", text)
    }


def _coverage_from_interval(low, high, actual: float) -> bool | None:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return None
    return bool(float(low) <= actual <= float(high))


def _interval_width_from_bounds(low, high) -> float | None:
    if low is None or high is None or pd.isna(low) or pd.isna(high):
        return None
    return float(high) - float(low)


def _float(value: str) -> float:
    return float(str(value).replace(",", "."))


def _summarize(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    return rows.groupby(group_cols, dropna=False, observed=True).apply(_summary_row, include_groups=False).reset_index()


def _summary_row(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": int(len(group)),
            "distinct_days": _distinct_days(group),
            "mae_expected": _mean(group["abs_error_expected_c"]),
            "bias_expected": _mean(group["error_expected_c"]),
            "rmse_expected": _rmse(group["error_expected_c"]),
            "mae_median": _mean(group.get("abs_error_median_c")),
            "mae_mode": _mean(group.get("abs_error_mode_c")),
            "mode_hit_rate": _mean(group.get("mode_hit")),
            "within_1c_expected_rate": _mean(group["within_1c_expected"]),
            "large_error_gt_2c_rate": _mean(group["large_error_gt_2c"]),
            "mean_nll": _mean(group.get("nll")),
            "mean_crps": _mean(group.get("crps")),
            "coverage_80": _mean(group.get("coverage_80")),
            "mean_p_actual_bin": _mean(group.get("probability_actual_integer_bin")),
            "mean_interval_80_width_c": _mean(group.get("interval_80_width_c")),
        }
    )


def _bias_summary(rows: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return rows.groupby(group_cols, dropna=False, observed=True).apply(_bias_row, include_groups=False).reset_index()


def _bias_row(group: pd.DataFrame) -> pd.Series:
    errors = pd.to_numeric(group["error_expected_c"], errors="coerce")
    expected = pd.to_numeric(group["expected_tmax_c"], errors="coerce")
    actual = pd.to_numeric(group["actual_tmax_c"], errors="coerce")
    valid = errors.notna() & expected.notna() & actual.notna()
    errors = errors[valid]
    expected = expected[valid]
    actual = actual[valid]
    if errors.empty:
        return pd.Series(
            {
                "rows": 0,
                "distinct_days": 0,
                "bias_expected": None,
                "current_mae_expected": None,
                "bias_corrected_mae_expected": None,
                "mae_improvement_c": None,
                "suggested_bias_correction_c": None,
                "large_error_gt_2c_rate_before": None,
                "large_error_gt_2c_rate_after": None,
            }
        )
    bias = float(errors.mean())
    corrected_errors = expected - bias - actual
    return pd.Series(
        {
            "rows": int(len(errors)),
            "distinct_days": _distinct_days(group.loc[valid]),
            "bias_expected": bias,
            "current_mae_expected": float(errors.abs().mean()),
            "bias_corrected_mae_expected": float(corrected_errors.abs().mean()),
            "mae_improvement_c": float(errors.abs().mean() - corrected_errors.abs().mean()),
            "suggested_bias_correction_c": float(-bias),
            "large_error_gt_2c_rate_before": float((errors.abs() > 2.0).mean()),
            "large_error_gt_2c_rate_after": float((corrected_errors.abs() > 2.0).mean()),
        }
    )


def _bias_recommendation(
    row: pd.Series,
    *,
    min_rows: int,
    min_distinct_days: int,
    min_abs_bias_c: float,
    min_mae_improvement_c: float,
) -> str:
    rows = int(row.get("rows") or 0)
    days = int(row.get("distinct_days") or 0)
    bias = row.get("bias_expected")
    improvement = row.get("mae_improvement_c")
    large_before = row.get("large_error_gt_2c_rate_before")
    large_after = row.get("large_error_gt_2c_rate_after")
    if rows < min_rows or days < min_distinct_days:
        return "watch_more_data"
    if pd.isna(bias) or abs(float(bias)) < min_abs_bias_c:
        return "no_action_small_bias"
    if pd.isna(improvement) or float(improvement) < min_mae_improvement_c:
        return "no_action_mae_not_improved"
    if pd.notna(large_before) and pd.notna(large_after) and float(large_after) > float(large_before) + 0.01:
        return "no_action_large_errors_worse"
    return "shadow_bias_correction"


def _distinct_days(group: pd.DataFrame) -> int:
    if "target_date_local" not in group.columns:
        return 1
    return int(group["target_date_local"].nunique())


def _rolling_summaries(rows: pd.DataFrame, windows_days: tuple[int, ...]) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    df["_target_date"] = pd.to_datetime(df["target_date_local"], errors="coerce")
    latest = df["_target_date"].max()
    frames = []
    for window in windows_days:
        start = latest - pd.Timedelta(days=window - 1)
        subset = df[df["_target_date"].between(start, latest, inclusive="both")].drop(columns=["_target_date"])
        if subset.empty:
            continue
        summary = _summarize(subset, ["airport", "model_version"])
        summary.insert(0, "window_days", int(window))
        summary.insert(1, "window_start", start.date().isoformat())
        summary.insert(2, "window_end", latest.date().isoformat())
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _alerts(rows: pd.DataFrame, rolling: pd.DataFrame) -> pd.DataFrame:
    alerts = []
    if not rolling.empty:
        for _, row in rolling.iterrows():
            prefix = {
                "airport": row["airport"],
                "model_version": row["model_version"],
                "window_days": row["window_days"],
            }
            if pd.notna(row.get("mae_expected")) and float(row["mae_expected"]) > 1.5:
                alerts.append({**prefix, "severity": "warning", "reason": "mae_expected_gt_1_5c", "value": row["mae_expected"]})
            if pd.notna(row.get("bias_expected")) and abs(float(row["bias_expected"])) > 0.8:
                alerts.append({**prefix, "severity": "warning", "reason": "abs_bias_expected_gt_0_8c", "value": row["bias_expected"]})
            if pd.notna(row.get("coverage_80")) and float(row["coverage_80"]) < 0.7:
                alerts.append({**prefix, "severity": "warning", "reason": "coverage_80_lt_70pct", "value": row["coverage_80"]})
    large = rows[rows["large_error_gt_2c"]].copy()
    for _, row in large.iterrows():
        alerts.append(
            {
                "airport": row["airport"],
                "model_version": row.get("model_version"),
                "window_days": None,
                "severity": "day",
                "reason": "forecast_error_gt_2c",
                "target_date_local": row["target_date_local"],
                "local_issue_hour": row["local_issue_hour"],
                "value": row["error_expected_c"],
            }
        )
    return pd.DataFrame(alerts)


def _metadata(
    monitoring: pd.DataFrame,
    rows: pd.DataFrame,
    start_hour: float,
    end_hour: float,
    windows_days: tuple[int, ...],
) -> dict:
    metadata = {
        "source_rows": int(len(monitoring)),
        "evaluated_rows": int(len(rows)),
        "local_hour_window": [start_hour, end_hour],
        "rolling_windows_days": list(windows_days),
        "airports": sorted(rows["airport"].dropna().astype(str).unique().tolist()) if "airport" in rows else [],
    }
    if not rows.empty:
        metadata.update(
            {
                "first_target_date_local": str(pd.to_datetime(rows["target_date_local"]).min().date()),
                "latest_target_date_local": str(pd.to_datetime(rows["target_date_local"]).max().date()),
            }
        )
    return metadata


def _markdown(report: LiveBaselineReport) -> str:
    lines = [
        "# Live Baseline 10-17 Local",
        "",
        f"- Source rows: `{report.metadata['source_rows']}`",
        f"- Evaluated rows: `{report.metadata['evaluated_rows']}`",
        f"- Local window: `{report.metadata['local_hour_window'][0]}-{report.metadata['local_hour_window'][1]}`",
        "",
    ]
    if report.overall.empty:
        lines.append("No scored forecasts available in the selected window.")
        return "\n".join(lines) + "\n"
    lines.extend(["## Overall", "", _table(report.overall)])
    if not report.rolling.empty:
        lines.extend(["", "## Rolling", "", _table(report.rolling)])
    if not report.bias_audit.empty:
        lines.extend(["", "## Bias Audit", "", _table(report.bias_audit)])
    if not report.alerts.empty:
        lines.extend(["", "## Alerts", "", _table(report.alerts.head(50))])
    return "\n".join(lines) + "\n"


def _table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    rendered = df.copy()
    for column in rendered.columns:
        if pd.api.types.is_float_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
        else:
            rendered[column] = rendered[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in rendered.iterrows():
        lines.append("| " + " | ".join(_escape_markdown_cell(row[column]) for column in rendered.columns) + " |")
    return "\n".join(lines)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _phase(hour: float) -> str:
    if hour < 12:
        return "morning_10_12"
    if hour < 14:
        return "midday_12_14"
    if hour < 16:
        return "afternoon_14_16"
    return "late_16_17"


def _local_hour(issue_time_utc, airport: str) -> float | None:
    if pd.isna(issue_time_utc):
        return None
    timestamp = pd.Timestamp(issue_time_utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    tz = ZoneInfo(AIRPORT_TIMEZONES.get(str(airport).upper(), "UTC"))
    local = timestamp.tz_convert(tz)
    return float(local.hour + local.minute / 60)


def _mean(values) -> float | None:
    if values is None:
        return None
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    return float(series.astype(float).mean())


def _rmse(values) -> float | None:
    series = pd.Series(values).dropna()
    if series.empty:
        return None
    numeric = series.astype(float)
    return float(math.sqrt((numeric * numeric).mean()))
