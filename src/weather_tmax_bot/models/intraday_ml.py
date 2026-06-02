from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer

from weather_tmax_bot.models.distribution import TmaxDistribution

DEFAULT_INTRADAY_ML_FEATURES = [
    "issue_hour_utc",
    "issue_hour_continuous_utc",
    "lead_to_local_day_end_hours",
    "month",
    "doy_sin",
    "doy_cos",
    "last_metar_temp_c",
    "last_metar_dewpoint_c",
    "last_metar_qnh_hpa",
    "temp_trend_1h",
    "temp_trend_3h",
    "temp_trend_6h",
    "observed_max_so_far_from_metar",
    "observed_min_so_far_from_metar",
    "pressure_trend_3h",
    "dewpoint_depression",
    "wind_u",
    "wind_v",
    "is_cavok",
    "has_precip_recent",
    "has_fog_recent",
    "has_thunder_recent",
    "metar_missing_last_1h",
    "metar_missing_last_3h",
    "taf_missing",
    "taf_has_rain",
    "taf_has_shower",
    "taf_has_thunder",
    "taf_has_fog",
    "taf_has_snow",
    "taf_wind_shift_flag",
    "taf_prob30_bad_weather",
    "taf_prob40_bad_weather",
    "taf_age_hours_at_issue",
    "nwp_missing",
    "model_tmax_c",
    "model_temp_at_08_local",
    "model_temp_at_11_local",
    "model_temp_at_14_local",
    "model_temp_at_17_local",
    "model_cloud_cover_mean",
    "model_precip_sum",
    "model_shortwave_radiation_sum",
    "model_wind_speed_max",
    "model_gust_max",
    "model_pressure_mean",
    "model_dewpoint_mean",
    "model_relative_humidity_mean",
]


@dataclass
class IntradayMLUpsideModel:
    """Ordinal remaining-upside model for same-day Tmax shadow forecasts."""

    max_upside_c: int = 20
    min_rows: int = 300
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_INTRADAY_ML_FEATURES))
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True))
    threshold_models: dict[int, HistGradientBoostingClassifier | None] = field(default_factory=dict)
    constant_probabilities: dict[int, float] = field(default_factory=dict)
    training_rows: int = 0
    fitted: bool = False

    def fit(self, dataset: pd.DataFrame) -> "IntradayMLUpsideModel":
        frame = prepare_intraday_ml_dataset(dataset)
        if len(frame) < self.min_rows:
            raise ValueError(f"intraday ML model requires at least {self.min_rows} valid rows")
        X = self.imputer.fit_transform(_numeric_feature_frame(frame, self.feature_columns))
        upside = frame["remaining_upside_c"].to_numpy(dtype=float)
        self.threshold_models = {}
        self.constant_probabilities = {}
        for threshold in range(1, self.max_upside_c + 1):
            labels = (upside >= threshold).astype(int)
            if np.unique(labels).size < 2:
                self.threshold_models[threshold] = None
                self.constant_probabilities[threshold] = float(labels.mean())
                continue
            model = HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=120,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )
            model.fit(X, labels)
            self.threshold_models[threshold] = model
        self.training_rows = len(frame)
        self.fitted = True
        return self

    def predict_upside_survival(self, feature_row: dict | pd.Series) -> dict[int, float]:
        if not self.fitted:
            raise ValueError("intraday ML model is not fitted")
        frame = pd.DataFrame([dict(feature_row)])
        X = self.imputer.transform(_numeric_feature_frame(frame, self.feature_columns))
        probabilities = []
        for threshold in range(1, self.max_upside_c + 1):
            model = self.threshold_models.get(threshold)
            probability = self.constant_probabilities.get(threshold, 0.0) if model is None else float(model.predict_proba(X)[0, 1])
            probabilities.append(probability)
        # Ordinal survival probabilities must never rise with the threshold.
        monotonic = np.minimum.accumulate(np.clip(probabilities, 0.0, 1.0))
        return {threshold: float(monotonic[threshold - 1]) for threshold in range(1, self.max_upside_c + 1)}

    def predict_distribution(self, feature_row: dict | pd.Series) -> tuple[TmaxDistribution, dict]:
        observed_max = _required_float(feature_row, "observed_max_so_far_from_metar")
        survival = self.predict_upside_survival(feature_row)
        survival_values = np.array([survival[threshold] for threshold in range(1, self.max_upside_c + 1)], dtype=float)
        probs = np.empty(self.max_upside_c + 1, dtype=float)
        probs[0] = 1.0 - survival_values[0]
        probs[1:-1] = survival_values[:-1] - survival_values[1:]
        probs[-1] = survival_values[-1]
        bins = np.rint(observed_max + np.arange(self.max_upside_c + 1)).astype(int)
        distribution = TmaxDistribution(bins, probs)
        return distribution, {
            "active": True,
            "name": "intraday_ml_core_challenger_v1",
            "training_rows": self.training_rows,
            "observed_max_so_far_c": observed_max,
            "probability_peak_already_passed": float(probs[0]),
            "probability_upside_ge_1c": survival[1],
            "probability_upside_ge_2c": survival[2],
            "probability_upside_ge_3c": survival[3],
            "upside_survival_probabilities": {str(key): value for key, value in survival.items()},
            "nwp_features_available": not bool(feature_row.get("nwp_missing", True)),
            "taf_features_available": not bool(feature_row.get("taf_missing", True)),
        }


def prepare_intraday_ml_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"tmax_c", "observed_max_so_far_from_metar", "last_metar_temp_c", "issue_time_utc", "target_date_local"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"dataset missing intraday ML columns: {missing}")
    frame = dataset.copy()
    frame["tmax_c"] = pd.to_numeric(frame["tmax_c"], errors="coerce")
    frame["observed_max_so_far_from_metar"] = pd.to_numeric(frame["observed_max_so_far_from_metar"], errors="coerce")
    frame["last_metar_temp_c"] = pd.to_numeric(frame["last_metar_temp_c"], errors="coerce")
    frame = frame[
        frame["tmax_c"].notna()
        & frame["observed_max_so_far_from_metar"].notna()
        & frame["last_metar_temp_c"].notna()
    ].copy()
    if "leakage_check_passed" in frame.columns:
        frame = frame[frame["leakage_check_passed"].fillna(False).astype(bool)].copy()
    frame["remaining_upside_c"] = (frame["tmax_c"] - frame["observed_max_so_far_from_metar"]).clip(lower=0.0)
    frame["peak_already_passed"] = frame["remaining_upside_c"] < 0.5
    for threshold in (1, 2, 3):
        frame[f"upside_ge_{threshold}c"] = frame["remaining_upside_c"] >= threshold
    return frame.reset_index(drop=True)


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        if values.dtype == bool:
            out[column] = values.astype(float)
        else:
            out[column] = pd.to_numeric(values, errors="coerce")
    return out


def _required_float(row: dict | pd.Series, key: str) -> float:
    value = row.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"intraday ML model requires {key}")
    return float(value)
