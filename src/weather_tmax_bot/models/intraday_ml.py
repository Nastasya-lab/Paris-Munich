from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
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
class IntradayMLSurvivalCalibrator:
    """Out-of-fold isotonic calibrator for ordinal remaining-upside probabilities."""

    max_upside_c: int = 20
    min_rows_per_threshold: int = 200
    prior_smoothing: float = 1.0
    max_prior_blend_weight: float = 0.35
    threshold_calibrators: dict[int, IsotonicRegression | None] = field(default_factory=dict)
    threshold_rows: dict[int, int] = field(default_factory=dict)
    hourly_prior_survival: dict[int, dict[int, float]] = field(default_factory=dict)
    global_prior_survival: dict[int, float] = field(default_factory=dict)
    prior_blend_weight: float = 0.0
    fitted: bool = False

    def fit(self, calibration_rows: pd.DataFrame) -> "IntradayMLSurvivalCalibrator":
        self.threshold_calibrators = {}
        self.threshold_rows = {}
        for threshold in range(1, self.max_upside_c + 1):
            prob_col = f"raw_probability_upside_ge_{threshold}c"
            actual_col = f"actual_upside_ge_{threshold}c"
            if prob_col not in calibration_rows.columns or actual_col not in calibration_rows.columns:
                self.threshold_calibrators[threshold] = None
                self.threshold_rows[threshold] = 0
                continue
            frame = calibration_rows[[prob_col, actual_col]].dropna()
            self.threshold_rows[threshold] = len(frame)
            if len(frame) < self.min_rows_per_threshold or frame[actual_col].nunique() < 2:
                self.threshold_calibrators[threshold] = None
                continue
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(frame[prob_col].to_numpy(dtype=float), frame[actual_col].to_numpy(dtype=float))
            self.threshold_calibrators[threshold] = model
        self.global_prior_survival = self._empirical_prior(calibration_rows)
        self.hourly_prior_survival = {
            int(hour): self._empirical_prior(frame)
            for hour, frame in calibration_rows.groupby("issue_hour_utc")
        }
        self.fitted = any(model is not None for model in self.threshold_calibrators.values())
        self.prior_blend_weight = self._fit_prior_blend_weight(calibration_rows) if self.fitted else 0.0
        return self

    def transform(self, survival: dict[int, float], issue_hour_utc: int | None = None) -> dict[int, float]:
        if not self.fitted:
            return {key: float(np.clip(value, 0.0, 1.0)) for key, value in survival.items()}
        calibrated = []
        for threshold in range(1, self.max_upside_c + 1):
            raw = float(np.clip(survival.get(threshold, 0.0), 0.0, 1.0))
            model = self.threshold_calibrators.get(threshold)
            value = raw if model is None else float(model.predict(np.array([raw], dtype=float))[0])
            calibrated.append(value)
        prior = self.hourly_prior_survival.get(int(issue_hour_utc)) if issue_hour_utc is not None else None
        prior = prior or self.global_prior_survival
        calibrated = [
            (1.0 - self.prior_blend_weight) * value + self.prior_blend_weight * prior.get(threshold, 0.0)
            for threshold, value in enumerate(calibrated, start=1)
        ]
        monotonic = np.minimum.accumulate(np.clip(calibrated, 0.0, 1.0))
        return {threshold: float(monotonic[threshold - 1]) for threshold in range(1, self.max_upside_c + 1)}

    def to_metadata(self) -> dict:
        return {
            "calibration_method": "out_of_fold_isotonic_survival",
            "max_upside_c": self.max_upside_c,
            "min_rows_per_threshold": self.min_rows_per_threshold,
            "fitted": self.fitted,
            "prior_blend_weight": self.prior_blend_weight,
            "prior_smoothing": self.prior_smoothing,
            "max_prior_blend_weight": self.max_prior_blend_weight,
            "threshold_rows": {str(key): int(value) for key, value in self.threshold_rows.items()},
            "calibrated_thresholds": [
                int(key) for key, model in self.threshold_calibrators.items() if model is not None
            ],
        }

    def _empirical_prior(self, frame: pd.DataFrame) -> dict[int, float]:
        upside = pd.to_numeric(frame["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
        rounded = np.clip(np.rint(upside), 0, self.max_upside_c).astype(int)
        counts = np.bincount(rounded, minlength=self.max_upside_c + 1).astype(float) + self.prior_smoothing
        probabilities = counts / counts.sum()
        return {
            threshold: float(probabilities[threshold:].sum())
            for threshold in range(1, self.max_upside_c + 1)
        }

    def _fit_prior_blend_weight(self, frame: pd.DataFrame) -> float:
        raw = np.column_stack(
            [
                pd.to_numeric(frame[f"raw_probability_upside_ge_{threshold}c"], errors="coerce")
                .fillna(0.0)
                .to_numpy(dtype=float)
                for threshold in range(1, self.max_upside_c + 1)
            ]
        )
        isotonic = np.empty_like(raw)
        for idx, threshold in enumerate(range(1, self.max_upside_c + 1)):
            model = self.threshold_calibrators.get(threshold)
            isotonic[:, idx] = raw[:, idx] if model is None else model.predict(raw[:, idx])
        priors = np.vstack(
            [
                [
                    self.hourly_prior_survival.get(int(hour), self.global_prior_survival).get(threshold, 0.0)
                    for threshold in range(1, self.max_upside_c + 1)
                ]
                for hour in frame["issue_hour_utc"]
            ]
        )
        actual_bins = np.clip(
            np.rint(pd.to_numeric(frame["remaining_upside_c"], errors="coerce").fillna(0.0)),
            0,
            self.max_upside_c,
        ).astype(int)
        best_weight = 0.0
        best_score = float("inf")
        for weight in np.linspace(0.0, self.max_prior_blend_weight, 8):
            survival = np.minimum.accumulate(
                np.clip((1.0 - weight) * isotonic + weight * priors, 0.0, 1.0),
                axis=1,
            )
            probabilities = _survival_matrix_to_probabilities(survival)
            score = float(np.mean(-np.log(np.maximum(probabilities[np.arange(len(frame)), actual_bins], 1e-12))))
            if score < best_score:
                best_score = score
                best_weight = float(weight)
        return best_weight

    def _transform_with_weight(
        self,
        survival: dict[int, float],
        issue_hour_utc: int | None,
        prior_blend_weight: float,
    ) -> dict[int, float]:
        prior = self.hourly_prior_survival.get(int(issue_hour_utc)) if issue_hour_utc is not None else None
        prior = prior or self.global_prior_survival
        calibrated = []
        for threshold in range(1, self.max_upside_c + 1):
            raw = float(np.clip(survival.get(threshold, 0.0), 0.0, 1.0))
            model = self.threshold_calibrators.get(threshold)
            value = raw if model is None else float(model.predict(np.array([raw], dtype=float))[0])
            calibrated.append((1.0 - prior_blend_weight) * value + prior_blend_weight * prior.get(threshold, 0.0))
        monotonic = np.minimum.accumulate(np.clip(calibrated, 0.0, 1.0))
        return {threshold: float(monotonic[threshold - 1]) for threshold in range(1, self.max_upside_c + 1)}


@dataclass
class IntradayMLUpsideModel:
    """Ordinal remaining-upside model for same-day Tmax shadow forecasts."""

    max_upside_c: int = 20
    min_rows: int = 300
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_INTRADAY_ML_FEATURES))
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True))
    threshold_models: dict[int, HistGradientBoostingClassifier | None] = field(default_factory=dict)
    constant_probabilities: dict[int, float] = field(default_factory=dict)
    calibrator: IntradayMLSurvivalCalibrator | None = None
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
                max_iter=80,
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
        raw_survival = self.predict_upside_survival(feature_row)
        survival = (
            self.calibrator.transform(raw_survival, issue_hour_utc=feature_row.get("issue_hour_utc"))
            if self.calibrator is not None
            else raw_survival
        )
        probs = _survival_to_probabilities(survival, self.max_upside_c)
        bins = np.rint(observed_max + np.arange(self.max_upside_c + 1)).astype(int)
        distribution = TmaxDistribution(bins, probs)
        return distribution, {
            "active": True,
            "name": "intraday_ml_core_challenger_v1",
            "training_rows": self.training_rows,
            "calibration_status": (
                "out_of_fold_isotonic_survival_calibrated"
                if self.calibrator is not None and self.calibrator.fitted
                else "uncalibrated"
            ),
            "calibration_metadata": None if self.calibrator is None else self.calibrator.to_metadata(),
            "observed_max_so_far_c": observed_max,
            "probability_peak_already_passed": float(probs[0]),
            "probability_upside_ge_1c": survival[1],
            "probability_upside_ge_2c": survival[2],
            "probability_upside_ge_3c": survival[3],
            "raw_probability_upside_ge_1c": raw_survival[1],
            "raw_probability_upside_ge_2c": raw_survival[2],
            "raw_probability_upside_ge_3c": raw_survival[3],
            "upside_survival_probabilities": {str(key): value for key, value in survival.items()},
            "raw_upside_survival_probabilities": {str(key): value for key, value in raw_survival.items()},
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


def _survival_to_probabilities(survival: dict[int, float], max_upside_c: int) -> np.ndarray:
    survival_values = np.array([survival[threshold] for threshold in range(1, max_upside_c + 1)], dtype=float)
    probs = np.empty(max_upside_c + 1, dtype=float)
    probs[0] = 1.0 - survival_values[0]
    probs[1:-1] = survival_values[:-1] - survival_values[1:]
    probs[-1] = survival_values[-1]
    return np.clip(probs, 0.0, 1.0)


def _survival_matrix_to_probabilities(survival: np.ndarray) -> np.ndarray:
    probs = np.empty((survival.shape[0], survival.shape[1] + 1), dtype=float)
    probs[:, 0] = 1.0 - survival[:, 0]
    probs[:, 1:-1] = survival[:, :-1] - survival[:, 1:]
    probs[:, -1] = survival[:, -1]
    return np.clip(probs, 0.0, 1.0)
