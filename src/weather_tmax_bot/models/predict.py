from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from weather_tmax_bot.features.build_features import build_feature_row
from weather_tmax_bot.models.baselines import ClimatologyBaseline
from weather_tmax_bot.models.extrapolation import detect_feature_extrapolation
from weather_tmax_bot.models.model_registry import load_model, resolve_active_artifacts
from weather_tmax_bot.temporal.freshness import assess_feature_freshness
from weather_tmax_bot.temporal.source_compatibility import assess_source_compatibility
from weather_tmax_bot.utils.time import local_day_bounds_utc


def predict_with_climatology(
    target_date: date,
    daily_target_path: str | Path = "data/processed/daily_target.parquet",
    observed_max_so_far: float | None = None,
):
    if Path(daily_target_path).exists():
        targets = pd.read_parquet(daily_target_path)
    else:
        targets = _synthetic_targets()
    return ClimatologyBaseline().fit(targets).predict_distribution(target_date, observed_max_so_far)


def predict_best_available(
    airport: str,
    target_date: date,
    issue_time_utc: datetime,
    daily_target_path: str | Path = "data/processed/daily_target.parquet",
    model_path: str | Path = "data/models/quantile_mvp.joblib",
) -> tuple[object, dict]:
    warnings = []
    active = resolve_active_artifacts(fallback_model_path=model_path)
    resolved_model_path = active["model_path"]
    if resolved_model_path is not None and Path(resolved_model_path).exists():
        metar = _load_metar_for_issue(airport, target_date, issue_time_utc)
        taf = _load_taf_for_issue(airport, issue_time_utc)
        nwp = _load_nwp_for_issue(target_date, issue_time_utc)
        feature_row = build_feature_row(
            airport_icao=airport,
            issue_time_utc=issue_time_utc,
            target_date_local=target_date,
            metar=metar,
            taf=taf,
            nwp=nwp,
        )
        model = load_model(resolved_model_path)
        observed_max = feature_row.get("observed_max_so_far_from_metar")
        try:
            dist = model.predict_distribution(pd.DataFrame([feature_row]), observed_max_so_far=observed_max)
        except ValueError as exc:
            warnings.append(f"Active model could not use current features: {exc}; climatology fallback used.")
            dist = predict_with_climatology(target_date, daily_target_path=daily_target_path, observed_max_so_far=observed_max)
        extrapolation = detect_feature_extrapolation(feature_row, model)
        feature_row["extrapolation"] = extrapolation
        warnings.extend(extrapolation["warnings"])
        calibrator_path = active["calibrator_path"]
        if calibrator_path is not None and calibrator_path.exists():
            calibrator = load_model(calibrator_path)
            dist = calibrator.transform(dist).truncate_below(observed_max)
            warnings.append("Validation-fitted spread calibration applied.")
        if model.__class__.__name__ == "QuantileTmaxModel":
            warnings.append("Quantile MVP model used; calibration layer is still preliminary.")
        else:
            warnings.append(f"{model.__class__.__name__} model used; monitor NWP source availability and residual calibration.")
        freshness = assess_feature_freshness(feature_row, issue_time_utc)
        feature_row["freshness"] = freshness["statuses"]
        warnings.extend(freshness["warnings"])
        source_compatibility = assess_source_compatibility(feature_row)
        feature_row["source_compatibility"] = source_compatibility["sources"]
        warnings.extend(source_compatibility["warnings"])
        if feature_row.get("nwp_missing", True):
            warnings.append("NWP forecast-as-issued archive not yet available.")
        if feature_row.get("taf_missing", True):
            warnings.append("TAF missing for this as-of feature view.")
        return dist, {
            "model_version": active.get("active_model_version") or Path(resolved_model_path).stem,
            "feature_snapshot": feature_row,
            "warnings": warnings,
        }
    dist = predict_with_climatology(target_date, daily_target_path=daily_target_path)
    warnings.append("Quantile model unavailable; climatology MVP distribution used.")
    return dist, {"model_version": "climatology_mvp", "feature_snapshot": {}, "warnings": warnings}


def _synthetic_targets() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2025-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    seasonal = 13 + 12 * pd.Series((2 * 3.14159 * (doy - 200) / 366)).map(lambda x: __import__("math").cos(x)).to_numpy()
    deterministic_variation = 2.5 * pd.Series((2 * 3.14159 * (doy * 7 + dates.year.to_numpy()) / 29)).map(
        lambda x: __import__("math").sin(x)
    ).to_numpy()
    return pd.DataFrame(
        {
            "airport_icao": "EDDM",
            "target_date_local": dates.date.astype(str),
            "tmax_c": seasonal + deterministic_variation,
            "source_id": "synthetic.climatology.placeholder",
        }
    )


def _load_optional(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _load_metar_for_issue(airport: str, target_date: date, issue_time_utc: datetime) -> pd.DataFrame:
    frames = []
    for path in (f"data/interim/metar_iem_{airport}.parquet", f"data/forecasts/awc_metar_live_{airport}.parquet"):
        df = _load_optional(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    metar = pd.concat(frames, ignore_index=True)
    start, _ = local_day_bounds_utc(target_date, "Europe/Berlin")
    context_start = min(pd.Timestamp(start), pd.Timestamp(issue_time_utc) - pd.Timedelta(hours=24))
    times = pd.to_datetime(metar["observation_time_utc"], utc=True)
    return metar[(times >= context_start) & (times <= pd.Timestamp(issue_time_utc))].copy()


def _load_nwp_for_issue(target_date: date, issue_time_utc: datetime) -> pd.DataFrame:
    frames = [
        frame
        for path in (
            "data/forecasts/open_meteo_archive.parquet",
            "data/forecasts/open_meteo_single_runs_icon_d2.parquet",
        )
        if not (frame := _load_optional(path)).empty
    ]
    if not frames:
        return pd.DataFrame()
    nwp = pd.concat(frames, ignore_index=True)
    nwp = nwp[nwp["target_date_local"].astype(str) == target_date.isoformat()].copy()
    if nwp.empty:
        return nwp
    availability = pd.to_datetime(nwp["model_availability_time_utc"], utc=True)
    return nwp[availability <= pd.Timestamp(issue_time_utc)].copy()


def _load_taf_for_issue(airport: str, issue_time_utc: datetime) -> pd.DataFrame:
    frames = []
    for path in (f"data/interim/taf_iem_{airport}.parquet", f"data/forecasts/awc_taf_live_{airport}.parquet"):
        df = _load_optional(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    taf = pd.concat(frames, ignore_index=True)
    if "knowledge_time_utc" not in taf.columns:
        return taf
    knowledge = pd.to_datetime(taf["knowledge_time_utc"], utc=True, errors="coerce")
    return taf[knowledge <= pd.Timestamp(issue_time_utc)].copy()
