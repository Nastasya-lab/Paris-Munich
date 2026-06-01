from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from weather_tmax_bot.evaluation.metrics import brier, crps_discrete, nll_integer_bin
from weather_tmax_bot.models.distribution import TmaxDistribution


def update_forecast_outcomes(
    forecast_log_path: str | Path = "data/logs/forecast_log.jsonl",
    target_path: str | Path = "data/processed/daily_target.parquet",
    output_path: str | Path = "data/reports/forecast_monitoring.parquet",
    variant_output_path: str | Path | None = None,
) -> pd.DataFrame:
    log_path = Path(forecast_log_path)
    if not log_path.exists():
        return pd.DataFrame()
    targets = pd.read_parquet(target_path)
    targets = targets[targets["quality_flags"] == "ok"].copy()
    targets["target_date_local"] = targets["target_date_local"].astype(str)
    target_map = targets.set_index(["airport_icao", "target_date_local"])["tmax_c"].to_dict()
    rows = []
    variant_rows = []
    for record in _iter_forecast_log(log_path):
        key = (record["airport"], record["target_date_local"])
        if key not in target_map:
            continue
        actual = float(target_map[key])
        dist = _distribution_from_record(record)
        base_row = _score_distribution_row(record, actual, dist, "production_champion")
        rows.append({key: value for key, value in base_row.items() if key not in {"forecast_variant", "variant_description"}})
        variant_rows.extend(_variant_rows_from_record(record, actual, champion_dist=dist))
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
) -> pd.DataFrame:
    log_path = Path(forecast_log_path)
    if not log_path.exists():
        return pd.DataFrame()
    targets = pd.read_parquet(target_path)
    targets["target_date_local"] = targets["target_date_local"].astype(str)
    target_by_key = targets.set_index(["airport_icao", "target_date_local"]).to_dict(orient="index")
    rows = []
    for record in _iter_forecast_log(log_path):
        key = (record["airport"], record["target_date_local"])
        target = target_by_key.get(key)
        if target is None:
            status = "pending_truth"
            actual = None
            quality = None
        else:
            actual = target.get("tmax_c")
            quality = target.get("quality_flags")
            status = "scored" if quality == "ok" else "truth_bad_quality"
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


def _score_distribution_row(record: dict, actual: float, dist: TmaxDistribution, forecast_variant: str, description: str | None = None) -> dict:
    expected = dist.expected_tmax_c
    median = dist.median_tmax_c
    return {
        "forecast_id": record["forecast_id"],
        "forecast_variant": forecast_variant,
        "variant_description": description,
        "airport": record["airport"],
        "issue_time_utc": record["issue_time_utc"],
        "target_date_local": record["target_date_local"],
        "model_version": record.get("model_version"),
        **_operational_metadata(record),
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
    }


def _variant_rows_from_record(record: dict, actual: float, *, champion_dist: TmaxDistribution) -> list[dict]:
    rows = [_score_distribution_row(record, actual, champion_dist, "production_champion", "Operational distribution returned to users.")]
    metadata = record.get("raw_input_metadata", {}) or {}
    for name, payload in (metadata.get("forecast_variants", {}) or {}).items():
        if name == "production_champion":
            continue
        dist = _distribution_from_variant_payload(payload)
        if dist is None:
            continue
        rows.append(_score_distribution_row(record, actual, dist, name, _variant_description(payload)))
    if not any(row["forecast_variant"] == "shadow_seasonal_intraday" for row in rows):
        fallback = _fallback_shadow_distribution(metadata)
        if fallback is not None:
            rows.append(_score_distribution_row(record, actual, fallback, "shadow_seasonal_intraday", "Shadow distribution reconstructed from legacy forecast_components."))
    return rows


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


def _probability_at_actual_bin(dist: TmaxDistribution, actual: float) -> float:
    actual_bin = int(round(actual))
    return float(dist.probabilities[dist.bins_c == actual_bin].sum())


def _iter_forecast_log(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _operational_metadata(record: dict) -> dict:
    metadata = record.get("raw_input_metadata", {}) or {}
    quality = metadata.get("forecast_quality", {}) or {}
    acceptance = metadata.get("forecast_acceptance", {}) or {}
    compatibility = metadata.get("source_compatibility", {}) or {}
    metar_source = metadata.get("latest_metar_source_id")
    taf_source = metadata.get("latest_taf_source_id")
    metar_compat = compatibility.get("metar", {})
    taf_compat = compatibility.get("taf", {})
    return {
        "latest_metar_source_id": metar_source,
        "latest_taf_source_id": taf_source,
        "latest_nwp_source_id": metadata.get("latest_nwp_source_id"),
        "metar_missing": bool(metadata.get("metar_missing", False)),
        "taf_missing": bool(metadata.get("taf_missing", False)),
        "nwp_missing": bool(metadata.get("nwp_missing", False)),
        "metar_source_mismatch": bool(metar_source and metar_source != "iem.metar.archive.EDDM"),
        "taf_source_mismatch": bool(taf_source and taf_source != "iem.taf.archive.EDDM"),
        "metar_source_compatibility_status": metar_compat.get("status"),
        "taf_source_compatibility_status": taf_compat.get("status"),
        "forecast_quality_status": quality.get("status"),
        "forecast_quality_reasons": ", ".join(quality.get("reasons", [])) if isinstance(quality.get("reasons"), list) else None,
        "forecast_quality_cautions": ", ".join(quality.get("cautions", [])) if isinstance(quality.get("cautions"), list) else None,
        "forecast_accepted": acceptance.get("accepted"),
        "forecast_acceptance_blocking_reasons": ", ".join(acceptance.get("blocking_reasons", []))
        if isinstance(acceptance.get("blocking_reasons"), list)
        else None,
        "forecast_acceptance_cautions": ", ".join(acceptance.get("cautions", [])) if isinstance(acceptance.get("cautions"), list) else None,
        "max_feature_knowledge_time_utc": record.get("max_feature_knowledge_time_utc"),
    }
