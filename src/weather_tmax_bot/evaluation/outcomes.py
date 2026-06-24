from __future__ import annotations

import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution

LOCAL_TZ = ZoneInfo("Europe/Berlin")
REPORT_VARIANTS = {"production_champion", "shadow_unimodal_pmf"}
METAR_TARGET_PATHS = {
    "EDDM": Path("data/processed/metar_tmax_target_EDDM.parquet"),
    "LFPB": Path("data/processed/metar_tmax_target_LFPB.parquet"),
}


def update_forecast_outcomes(
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    target_path: str | Path = "data/processed/daily_target.parquet",
    output_path: str | Path = "data/reports/forecast_monitoring.parquet",
    variant_output_path: str | Path | None = None,
    metar_target_paths: dict[str, str | Path] | None = None,
    metar_archive_dir: str | Path = "data",
) -> pd.DataFrame:
    log_path = Path(forecast_log_path)
    if not log_path.exists():
        return pd.DataFrame()
    records = list(_iter_forecast_log(log_path))
    truth_index = _load_truth_index(
        target_path=target_path,
        metar_target_paths=metar_target_paths,
        metar_archive_dir=metar_archive_dir,
        metar_airports=_metar_target_airports(records),
    )
    rows = []
    variant_rows = []
    for record in records:
        truth = _truth_for_record(record, truth_index)
        if truth is None or truth.get("quality_flags") != "ok":
            continue
        actual = float(truth["actual_tmax_c"])
        dist = _distribution_from_record(record)
        base_row = _score_distribution_row(
            record,
            actual,
            dist,
            "production_champion",
            metadata=_variant_metadata_from_record(record, "production_champion"),
            truth=truth,
        )
        rows.append({key: value for key, value in base_row.items() if key not in {"forecast_variant", "variant_description"}})
        variant_rows.extend(_variant_rows_from_record(record, actual, champion_dist=dist, truth=truth))
    out = pd.DataFrame(rows)
    if not out.empty:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
    variants = pd.DataFrame(variant_rows)
    if not variants.empty:
        variant_output = Path(variant_output_path) if variant_output_path is not None else Path(output_path).parent / "forecast_variant_monitoring.parquet"
        variant_output.parent.mkdir(parents=True, exist_ok=True)
        variants.to_parquet(variant_output, index=False)
    return out


def build_forecast_outcome_status(
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    target_path: str | Path = "data/processed/daily_target.parquet",
    output_path: str | Path | None = "data/reports/forecast_outcome_status.parquet",
    metar_target_paths: dict[str, str | Path] | None = None,
    metar_archive_dir: str | Path = "data",
) -> pd.DataFrame:
    log_path = Path(forecast_log_path)
    if not log_path.exists():
        return pd.DataFrame()
    records = list(_iter_forecast_log(log_path))
    truth_index = _load_truth_index(
        target_path=target_path,
        metar_target_paths=metar_target_paths,
        metar_archive_dir=metar_archive_dir,
        metar_airports=_metar_target_airports(records),
    )
    rows = []
    for record in records:
        truth = _truth_for_record(record, truth_index)
        if truth is None:
            status = "pending_truth"
            actual = None
            quality = None
        else:
            actual = truth.get("actual_tmax_c")
            quality = truth.get("quality_flags")
            status = "scored" if quality == "ok" else "truth_bad_quality"
        target_kind = _target_kind_from_record(record)
        rows.append(
            {
                "forecast_id": record["forecast_id"],
                "airport": record["airport"],
                "issue_time_utc": record["issue_time_utc"],
                "target_date_local": record["target_date_local"],
                "model_version": record.get("model_version"),
                **_operational_metadata(record),
                "outcome_status": status,
                "actual_tmax_c": actual,
                "target_quality_flags": quality,
                "target_kind": target_kind,
                "truth_source": None if truth is None else truth.get("truth_source"),
            }
        )
    out = pd.DataFrame(rows)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)
    return out


def _distribution_from_record(record: dict) -> TmaxDistribution:
    probabilities = record["probability_distribution"]
    return _distribution_from_probabilities(probabilities)


def _distribution_from_probabilities(probabilities: dict) -> TmaxDistribution:
    bins = [int(k) for k in probabilities.keys()]
    probs = [float(v) for v in probabilities.values()]
    return TmaxDistribution(bins, probs)


def _score_distribution_row(
    record: dict,
    actual: float,
    dist: TmaxDistribution,
    forecast_variant: str,
    description: str | None = None,
    metadata: dict | None = None,
    truth: dict | None = None,
) -> dict:
    expected = dist.expected_tmax_c
    median = dist.median_tmax_c
    issue_local = _issue_time_local(record.get("issue_time_utc"))
    return {
        "forecast_id": record["forecast_id"],
        "forecast_variant": forecast_variant,
        "variant_description": description,
        "variant_version": (metadata or {}).get("variant_version"),
        "forecast_phase": (metadata or {}).get("forecast_phase"),
        "scenario_tracking": (metadata or {}).get("scenario_tracking"),
        "phase_reason": (metadata or {}).get("phase_reason"),
        "phase_season": (metadata or {}).get("phase_season"),
        "local_issue_hour": _local_issue_hour(issue_local, metadata),
        "issue_time_local": None if issue_local is None else issue_local.isoformat(),
        "intraday_blend_weight": (metadata or {}).get("intraday_blend_weight"),
        "peak_passed_probability": (metadata or {}).get("peak_passed_probability"),
        "observed_max_so_far_c": (metadata or {}).get("observed_max_so_far_c"),
        "drop_from_observed_max_c": (metadata or {}).get("drop_from_observed_max_c"),
        "metar_weather_break_signal": (metadata or {}).get("metar_weather_break_signal"),
        "taf_adverse_weather_signal": (metadata or {}).get("taf_adverse_weather_signal"),
        "nwp_adverse_weather_signal": (metadata or {}).get("nwp_adverse_weather_signal"),
        "nwp_future_heating_signal": (metadata or {}).get("nwp_future_heating_signal"),
        "airport": record["airport"],
        "issue_time_utc": record["issue_time_utc"],
        "target_date_local": record["target_date_local"],
        "model_version": record.get("model_version"),
        **_operational_metadata(record),
        "target_kind": (truth or {}).get("target_kind", _target_kind_from_record(record)),
        "truth_source": (truth or {}).get("truth_source"),
        "target_quality_flags": (truth or {}).get("quality_flags"),
        "actual_tmax_c": actual,
        "expected_tmax_c": expected,
        "median_tmax_c": median,
        "most_likely_integer_c": dist.most_likely_integer_c,
        "error_expected_c": expected - actual,
        "error_median_c": median - actual,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
        "brier_ge_20": brier(dist.threshold_ge(20), actual >= 20),
        "brier_ge_25": brier(dist.threshold_ge(25), actual >= 25),
        "brier_ge_30": brier(dist.threshold_ge(30), actual >= 30),
        "probability_actual_integer_bin": _probability_at_actual_bin(dist, actual),
        "probability_above_actual_integer_bin": _probability_above_actual_bin(dist, actual),
        "coverage_50": _interval_contains(dist, actual, 0.50),
        "coverage_80": _interval_contains(dist, actual, 0.80),
        "coverage_90": _interval_contains(dist, actual, 0.90),
        "interval_80_width_c": _interval_width(dist, 0.80),
    }


def _variant_rows_from_record(record: dict, actual: float, *, champion_dist: TmaxDistribution, truth: dict | None = None) -> list[dict]:
    rows = [
        _score_distribution_row(
            record,
            actual,
            champion_dist,
            "production_champion",
            "Operational distribution returned to users.",
            metadata=_variant_metadata_from_record(record, "production_champion"),
            truth=truth,
        )
    ]
    metadata = record.get("raw_input_metadata", {}) or {}
    for name, payload in (metadata.get("forecast_variants", {}) or {}).items():
        if name == "production_champion":
            continue
        if name not in REPORT_VARIANTS:
            continue
        dist = _distribution_from_variant_payload(payload)
        if dist is None:
            continue
        rows.append(
            _score_distribution_row(
                record,
                actual,
                dist,
                name,
                _variant_description(payload),
                metadata=_variant_metadata(payload),
                truth=truth,
            )
        )
    return rows


def _load_truth_index(
    *,
    target_path: str | Path,
    metar_target_paths: dict[str, str | Path] | None,
    metar_archive_dir: str | Path,
    metar_airports: set[str] | None = None,
) -> dict[tuple[str, str, str], dict]:
    index: dict[tuple[str, str, str], dict] = {}
    dwd_path = Path(target_path)
    if dwd_path.exists():
        targets = pd.read_parquet(dwd_path)
        if not targets.empty:
            targets = targets.copy()
            targets["target_date_local"] = targets["target_date_local"].astype(str)
            for _, row in targets.iterrows():
                airport = str(row.get("airport_icao", "EDDM"))
                day = str(row["target_date_local"])
                index[("DWD_Tmax", airport, day)] = {
                    "target_kind": "DWD_Tmax",
                    "actual_tmax_c": row.get("tmax_c"),
                    "quality_flags": row.get("quality_flags"),
                    "truth_source": row.get("source_id", "DWD 10-minute truth"),
                }
    paths = {**METAR_TARGET_PATHS, **(metar_target_paths or {})}
    if metar_airports is not None:
        paths = {airport: path for airport, path in paths.items() if airport in metar_airports}
    for airport, path_value in paths.items():
        path = Path(path_value)
        if path.exists():
            metar_targets = pd.read_parquet(path)
            _add_metar_target_rows(index, airport, metar_targets, truth_source=f"{airport} METAR Tmax target")
    _add_metar_archive_truth(index, Path(metar_archive_dir), airports=metar_airports)
    return index


def _add_metar_target_rows(index: dict, airport: str, frame: pd.DataFrame, *, truth_source: str) -> None:
    if frame.empty:
        return
    data = frame.copy()
    data["target_date_local"] = data["target_date_local"].astype(str)
    for _, row in data.iterrows():
        day = str(row["target_date_local"])
        actual = row.get("metar_tmax_c", row.get("tmax_c"))
        index[("METAR_Tmax", airport, day)] = {
            "target_kind": "METAR_Tmax",
            "actual_tmax_c": actual,
            "quality_flags": row.get("quality_flags", "ok"),
            "truth_source": row.get("source_id", truth_source),
        }


def _add_metar_archive_truth(index: dict, root: Path, *, airports: set[str] | None = None) -> None:
    airport_codes = set(airports or {airport for _, airport, _ in index.keys()})
    if not airport_codes:
        return
    for airport in sorted(airport_codes):
        frames = []
        for path in (
            root / "interim" / f"metar_iem_{airport}.parquet",
            root / "forecasts" / f"awc_metar_live_{airport}.parquet",
        ):
            if path.exists():
                frames.append(pd.read_parquet(path))
        if not frames:
            continue
        metar = pd.concat(frames, ignore_index=True)
        if "raw_record_hash" in metar.columns:
            metar = metar.drop_duplicates(subset=["raw_record_hash"], keep="last")
        _add_metar_target_rows(index, airport, _build_metar_truth_from_archive(metar, airport), truth_source=f"{airport} METAR archive")


def _build_metar_truth_from_archive(metar: pd.DataFrame, airport: str) -> pd.DataFrame:
    if metar.empty or "observation_time_utc" not in metar.columns or "temperature_c" not in metar.columns:
        return pd.DataFrame(columns=["airport_icao", "target_date_local", "metar_tmax_c", "quality_flags", "source_id"])
    timezone = "Europe/Berlin" if airport.upper() == "EDDM" else "Europe/Paris"
    df = metar.copy()
    df["observation_time_utc"] = pd.to_datetime(df["observation_time_utc"], utc=True, errors="coerce")
    df["temperature_c"] = pd.to_numeric(df["temperature_c"], errors="coerce")
    df = df.dropna(subset=["observation_time_utc", "temperature_c"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["airport_icao", "target_date_local", "metar_tmax_c", "quality_flags", "source_id"])
    df["target_date_local"] = df["observation_time_utc"].dt.tz_convert(timezone).dt.date.astype(str)
    rows = []
    for day, group in df.groupby("target_date_local", sort=True):
        rows.append(
            {
                "airport_icao": airport,
                "target_date_local": day,
                "metar_tmax_c": float(group["temperature_c"].max()),
                "quality_flags": "ok" if len(group) >= 8 else "low_coverage",
                "source_id": f"metar.archive.{airport}",
            }
        )
    return pd.DataFrame(rows)


def _truth_for_record(record: dict, truth_index: dict[tuple[str, str, str], dict]) -> dict | None:
    return truth_index.get((_target_kind_from_record(record), str(record["airport"]), str(record["target_date_local"])))


def _metar_target_airports(records: list[dict]) -> set[str]:
    return {str(record["airport"]) for record in records if _target_kind_from_record(record) == "METAR_Tmax"}


def _target_kind_from_record(record: dict) -> str:
    metadata = record.get("raw_input_metadata", {}) or {}
    target = str(metadata.get("target") or record.get("target") or "")
    if target == "METAR_Tmax":
        return "METAR_Tmax"
    return "DWD_Tmax"


def _distribution_from_variant_payload(payload: dict) -> TmaxDistribution | None:
    distribution = (payload or {}).get("distribution") or payload
    probabilities = (distribution or {}).get("probabilities_by_integer_c")
    if not probabilities:
        return None
    return _distribution_from_probabilities(probabilities)


def _fallback_shadow_distribution(metadata: dict) -> TmaxDistribution | None:
    components = metadata.get("forecast_components", {}) or {}
    shadow = components.get("shadow_mode", {}) or {}
    final_model = shadow.get("final_model", {}) or {}
    probabilities = final_model.get("probabilities_by_integer_c")
    if not probabilities:
        return None
    return _distribution_from_probabilities(probabilities)


def _variant_description(payload: dict) -> str | None:
    if isinstance(payload, dict):
        return payload.get("description")
    return None


def _variant_metadata(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    return payload.get("metadata") or {}


def _variant_metadata_from_record(record: dict, variant_name: str) -> dict:
    metadata = record.get("raw_input_metadata", {}) or {}
    payload = (metadata.get("forecast_variants", {}) or {}).get(variant_name, {})
    if payload:
        return _variant_metadata(payload)
    components = metadata.get("forecast_components", {}) or {}
    if variant_name == "production_champion":
        return components.get("intraday_update", {}) or {}
    return {}


def _fallback_shadow_metadata(metadata: dict) -> dict:
    components = metadata.get("forecast_components", {}) or {}
    shadow = components.get("shadow_mode", {}) or {}
    details = shadow.get("intraday_update", {}) or {}
    if shadow.get("name") and "variant_version" not in details:
        details = {"variant_version": shadow.get("name"), **details}
    return details


def _probability_at_actual_bin(dist: TmaxDistribution, actual: float) -> float:
    actual_bin = int(round(actual))
    return float(dist.probabilities[dist.bins_c == actual_bin].sum())


def _probability_above_actual_bin(dist: TmaxDistribution, actual: float) -> float:
    actual_bin = int(round(actual))
    return float(dist.probabilities[dist.bins_c > actual_bin].sum())


def _interval_contains(dist: TmaxDistribution, actual: float, central_mass: float) -> bool:
    low, high = dist.interval(central_mass)
    return bool(low <= actual <= high)


def _interval_width(dist: TmaxDistribution, central_mass: float) -> float:
    low, high = dist.interval(central_mass)
    return float(high - low)


def _issue_time_local(value) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert(LOCAL_TZ)


def _local_issue_hour(issue_local: pd.Timestamp | None, metadata: dict | None) -> float | None:
    metadata_hour = (metadata or {}).get("local_issue_hour")
    if metadata_hour is not None:
        return float(metadata_hour)
    if issue_local is None:
        return None
    return float(issue_local.hour + issue_local.minute / 60)


def _iter_forecast_log(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _operational_metadata(record: dict) -> dict:
    metadata = record.get("raw_input_metadata", {}) or {}
    quality = metadata.get("forecast_quality", {}) or {}
    acceptance = metadata.get("forecast_acceptance", {}) or {}
    compatibility = metadata.get("source_compatibility", {}) or {}
    disagreement = metadata.get("model_disagreement", {}) or (metadata.get("forecast_components", {}) or {}).get("model_disagreement", {}) or {}
    disagreement_summary = disagreement.get("summary", {}) or {}
    metar_source = metadata.get("latest_metar_source_id")
    taf_source = metadata.get("latest_taf_source_id")
    nwp_source = metadata.get("latest_nwp_source_id")
    metar_compat = compatibility.get("metar", {})
    taf_compat = compatibility.get("taf", {})
    nwp_compat = compatibility.get("nwp", {})
    return {
        "latest_metar_source_id": metar_source,
        "latest_taf_source_id": taf_source,
        "latest_nwp_source_id": nwp_source,
        "metar_missing": bool(metadata.get("metar_missing", False)),
        "taf_missing": bool(metadata.get("taf_missing", False)),
        "nwp_missing": bool(metadata.get("nwp_missing", False)),
        "metar_source_mismatch": _is_source_mismatch(metar_compat),
        "taf_source_mismatch": _is_source_mismatch(taf_compat),
        "nwp_source_mismatch": _is_source_mismatch(nwp_compat),
        "metar_source_compatibility_status": metar_compat.get("status"),
        "taf_source_compatibility_status": taf_compat.get("status"),
        "nwp_source_compatibility_status": nwp_compat.get("status"),
        "forecast_quality_status": quality.get("status"),
        "forecast_quality_reasons": ", ".join(quality.get("reasons", [])) if isinstance(quality.get("reasons"), list) else None,
        "forecast_quality_cautions": ", ".join(quality.get("cautions", [])) if isinstance(quality.get("cautions"), list) else None,
        "model_disagreement_status": disagreement.get("status"),
        "model_disagreement_severity": disagreement.get("severity"),
        "model_disagreement_reasons": ", ".join(disagreement.get("reasons", []))
        if isinstance(disagreement.get("reasons"), list)
        else None,
        "model_disagreement_expected_spread_c": disagreement_summary.get("expected_tmax_spread_c"),
        "model_disagreement_ge_25_spread": disagreement_summary.get("ge_25_probability_spread"),
        "model_disagreement_ge_30_spread": disagreement_summary.get("ge_30_probability_spread"),
        "forecast_accepted": acceptance.get("accepted"),
        "forecast_acceptance_blocking_reasons": ", ".join(acceptance.get("blocking_reasons", []))
        if isinstance(acceptance.get("blocking_reasons"), list)
        else None,
        "forecast_acceptance_cautions": ", ".join(acceptance.get("cautions", [])) if isinstance(acceptance.get("cautions"), list) else None,
        "max_feature_knowledge_time_utc": record.get("max_feature_knowledge_time_utc"),
    }


def _is_source_mismatch(compatibility: dict) -> bool:
    return compatibility.get("status") in {"known_compatible", "unknown_mismatch", "forbidden_mismatch"}
