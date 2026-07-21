from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from weather_tmax_bot.models.distribution import (
    TmaxDistribution,
    project_unimodal_distribution,
    temperature_scale_distribution,
)


LEMD_NWP_MODELS = {
    "icon_eu": "icon_eu",
    "ecmwf_ifs025": "ecmwf",
    "gfs_global": "gfs",
    "meteofrance_arpege_europe": "arpege",
}

NWP_AGGREGATES = (
    "tmax_c",
    "future_temp_max_c",
    "cloud_cover_mean",
    "future_cloud_cover_mean",
    "precip_sum",
    "future_precip_sum",
    "shortwave_radiation_sum",
    "future_shortwave_radiation_sum",
    "wind_speed_max",
    "future_wind_speed_max",
    "gust_max",
    "future_gust_max",
    "dewpoint_mean",
    "relative_humidity_mean",
    "surface_pressure_mean",
    "cape_max",
)


def prefixed_nwp_feature_columns(prefixes: list[str] | tuple[str, ...]) -> list[str]:
    return [f"{prefix}_{name}" for prefix in prefixes for name in NWP_AGGREGATES]


def blend_nwp_features(
    row: dict | pd.Series,
    weights: dict[str, float],
    *,
    prefixes: list[str] | tuple[str, ...] | None = None,
) -> dict[str, float | int | str]:
    """Build robust aggregate NWP features from all currently available models."""
    values = dict(row)
    model_prefixes = list(prefixes or weights)
    available = [
        prefix
        for prefix in model_prefixes
        if _finite(values.get(f"{prefix}_tmax_c"))
    ]
    if not available:
        raise ValueError("No usable NWP Tmax is available")
    normalized = _available_weights(weights, available)
    out: dict[str, float | int | str] = {
        "nwp_available_model_count": len(available),
        "nwp_available_models": ",".join(available),
    }
    for name in NWP_AGGREGATES:
        pairs = [
            (prefix, float(values[f"{prefix}_{name}"]))
            for prefix in available
            if _finite(values.get(f"{prefix}_{name}"))
        ]
        if not pairs:
            out[f"nwp_blend_{name}"] = float("nan")
            continue
        pair_weights = _available_weights(normalized, [prefix for prefix, _ in pairs])
        out[f"nwp_blend_{name}"] = float(sum(pair_weights[prefix] * value for prefix, value in pairs))
    tmax_values = np.asarray([float(values[f"{prefix}_tmax_c"]) for prefix in available], dtype=float)
    out["nwp_tmax_spread_c"] = float(tmax_values.max() - tmax_values.min())
    out["nwp_tmax_std_c"] = float(tmax_values.std(ddof=0))
    out["model_tmax_c"] = float(out["nwp_blend_tmax_c"])
    out["model_future_temp_max_c"] = float(out["nwp_blend_future_temp_max_c"])
    return out


@dataclass
class MultiNwpMetarTmaxModel:
    """Production wrapper for a calibrated METAR-upside and multi-NWP ensemble."""

    ensemble: object
    nwp_weights: dict[str, float]
    nwp_prefixes: list[str]
    model_version: str
    shape_strategy: str = "raw"
    unimodal_temperature: float = 0.67
    minimum_nwp_models: int = 2
    residual_nwp_prefix: str | None = None

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        row = dict(feature_row)
        blended = blend_nwp_features(row, self.nwp_weights, prefixes=self.nwp_prefixes)
        row.update(blended)
        if self.residual_nwp_prefix:
            row["model_tmax_c"] = float(row[f"{self.residual_nwp_prefix}_tmax_c"])
            row["model_future_temp_max_c"] = float(row[f"{self.residual_nwp_prefix}_future_temp_max_c"])
        current_max = float(row["current_metar_max_c"])
        row["nwp_model_minus_current_max_c"] = float(row["model_tmax_c"] - current_max)
        row["nwp_future_minus_current_max_c"] = float(row["model_future_temp_max_c"] - current_max)
        distribution = self.ensemble.predict_distribution(row)
        if self.shape_strategy == "unimodal":
            return project_unimodal_distribution(distribution)
        if self.shape_strategy == "temperature_unimodal":
            return project_unimodal_distribution(
                temperature_scale_distribution(distribution, self.unimodal_temperature)
            )
        if self.shape_strategy != "raw":
            raise ValueError(f"Unknown PMF shape strategy: {self.shape_strategy}")
        return distribution

    def source_status(self, feature_row: dict | pd.Series) -> dict:
        available = [
            prefix
            for prefix in self.nwp_prefixes
            if _finite(dict(feature_row).get(f"{prefix}_tmax_c"))
        ]
        return {
            "available_models": available,
            "available_model_count": len(available),
            "minimum_required": self.minimum_nwp_models,
            "degraded": len(available) < len(self.nwp_prefixes),
            "usable": len(available) >= self.minimum_nwp_models,
        }

    def to_metadata(self) -> dict:
        return {
            "model_version": self.model_version,
            "model_family": "multi_nwp_metar_remaining_upside",
            "nwp_weights": self.nwp_weights,
            "nwp_prefixes": self.nwp_prefixes,
            "shape_strategy": self.shape_strategy,
            "unimodal_temperature": self.unimodal_temperature,
            "minimum_nwp_models": self.minimum_nwp_models,
            "residual_nwp_prefix": self.residual_nwp_prefix,
        }


def _available_weights(weights: dict[str, float], available: list[str]) -> dict[str, float]:
    raw = np.asarray([max(0.0, float(weights.get(name, 0.0))) for name in available], dtype=float)
    if raw.sum() <= 0:
        raw = np.ones(len(available), dtype=float)
    raw /= raw.sum()
    return {name: float(value) for name, value in zip(available, raw, strict=True)}


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
