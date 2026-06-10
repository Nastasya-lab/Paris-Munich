from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

from weather_tmax_bot.models.distribution import TmaxDistribution


DEFAULT_METAR_TMAX_FEATURES = [
    "local_issue_hour",
    "current_metar_max_c",
    "latest_metar_temp_c",
    "drop_from_current_max_c",
    "metar_count_so_far",
    "metar_count_last_1h",
    "metar_count_last_3h",
    "temp_trend_1h",
    "temp_trend_3h",
    "temp_trend_6h",
    "has_rain_recent_metar",
    "has_thunder_recent_metar",
    "is_cavok_latest",
    "rain_mm_last_30m",
    "rain_mm_last_1h",
    "rain_mm_last_3h",
    "rain_mm_since_midnight",
    "rain_max_6min_last_3h",
    "month",
    "doy_sin",
    "doy_cos",
]


@dataclass
class MetarTmaxSurvivalCalibrator:
    """Out-of-fold isotonic calibrator for METAR Tmax remaining-upside probabilities."""

    max_upside_c: int = 12
    min_rows_per_threshold: int = 200
    max_prior_blend_weight: float = 0.35
    threshold_calibrators: dict[int, IsotonicRegression | None] = field(default_factory=dict)
    threshold_rows: dict[int, int] = field(default_factory=dict)
    global_prior_survival: dict[int, float] = field(default_factory=dict)
    hourly_prior_survival: dict[int, dict[int, float]] = field(default_factory=dict)
    seasonal_hourly_prior_survival: dict[str, dict[int, float]] = field(default_factory=dict)
    prior_blend_weight: float = 0.0
    fitted: bool = False

    def fit(self, calibration_rows: pd.DataFrame) -> "MetarTmaxSurvivalCalibrator":
        self.threshold_calibrators = {}
        self.threshold_rows = {}
        for threshold in range(1, self.max_upside_c + 1):
            prob_col = f"raw_probability_upside_ge_{threshold}c"
            actual_col = f"actual_upside_ge_{threshold}c"
            frame = calibration_rows[[prob_col, actual_col]].dropna() if {prob_col, actual_col}.issubset(calibration_rows.columns) else pd.DataFrame()
            self.threshold_rows[threshold] = len(frame)
            if len(frame) < self.min_rows_per_threshold or frame[actual_col].nunique() < 2:
                self.threshold_calibrators[threshold] = None
                continue
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(frame[prob_col].to_numpy(dtype=float), frame[actual_col].to_numpy(dtype=float))
            self.threshold_calibrators[threshold] = model
        self.global_prior_survival = self._empirical_prior(calibration_rows)
        self.hourly_prior_survival = {
            int(hour): self._empirical_prior(group)
            for hour, group in calibration_rows.groupby("local_issue_hour", dropna=True)
        }
        self.seasonal_hourly_prior_survival = {
            _context_key(hour, season): self._empirical_prior(group)
            for (hour, season), group in calibration_rows.groupby(["local_issue_hour", "season"], dropna=True)
        }
        self.fitted = any(model is not None for model in self.threshold_calibrators.values())
        self.prior_blend_weight = self._fit_prior_blend_weight(calibration_rows) if self.fitted else 0.0
        return self

    def transform(self, survival: dict[int, float], *, local_issue_hour=None, season=None) -> dict[int, float]:
        if not self.fitted:
            return survival
        calibrated = []
        prior = self._select_prior(local_issue_hour, season)
        for threshold in range(1, self.max_upside_c + 1):
            raw = float(np.clip(survival.get(threshold, 0.0), 0.0, 1.0))
            model = self.threshold_calibrators.get(threshold)
            value = raw if model is None else float(model.predict(np.array([raw], dtype=float))[0])
            value = (1.0 - self.prior_blend_weight) * value + self.prior_blend_weight * prior.get(threshold, 0.0)
            calibrated.append(value)
        monotonic = np.minimum.accumulate(np.clip(calibrated, 0.0, 1.0))
        return {threshold: float(monotonic[threshold - 1]) for threshold in range(1, self.max_upside_c + 1)}

    def to_metadata(self) -> dict:
        return {
            "calibration_method": "metar_tmax_out_of_fold_isotonic_survival",
            "max_upside_c": self.max_upside_c,
            "min_rows_per_threshold": self.min_rows_per_threshold,
            "fitted": self.fitted,
            "prior_blend_weight": self.prior_blend_weight,
            "threshold_rows": self.threshold_rows,
            "calibrated_thresholds": [
                int(threshold) for threshold, model in self.threshold_calibrators.items() if model is not None
            ],
            "hourly_prior_count": len(self.hourly_prior_survival),
            "seasonal_hourly_prior_count": len(self.seasonal_hourly_prior_survival),
        }

    def _empirical_prior(self, frame: pd.DataFrame) -> dict[int, float]:
        if frame.empty or "remaining_upside_c" not in frame.columns:
            return {threshold: 0.0 for threshold in range(1, self.max_upside_c + 1)}
        upside = pd.to_numeric(frame["remaining_upside_c"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(upside) == 0:
            return {threshold: 0.0 for threshold in range(1, self.max_upside_c + 1)}
        return {threshold: float((upside >= threshold).mean()) for threshold in range(1, self.max_upside_c + 1)}

    def _select_prior(self, local_issue_hour, season) -> dict[int, float]:
        if local_issue_hour is not None and season is not None:
            prior = self.seasonal_hourly_prior_survival.get(_context_key(local_issue_hour, season))
            if prior:
                return prior
        if local_issue_hour is not None:
            prior = self.hourly_prior_survival.get(int(local_issue_hour))
            if prior:
                return prior
        return self.global_prior_survival

    def _fit_prior_blend_weight(self, calibration_rows: pd.DataFrame) -> float:
        rows = calibration_rows.dropna(subset=["remaining_upside_c"]).copy()
        if rows.empty:
            return 0.0
        raw = np.column_stack(
            [
                pd.to_numeric(rows.get(f"raw_probability_upside_ge_{threshold}c"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
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
                    self._select_prior(row.get("local_issue_hour"), row.get("season")).get(threshold, 0.0)
                    for threshold in range(1, self.max_upside_c + 1)
                ]
                for _, row in rows.iterrows()
            ]
        )
        actual_bins = np.clip(np.rint(rows["remaining_upside_c"].to_numpy(dtype=float)).astype(int), 0, self.max_upside_c)
        best_weight = 0.0
        best_score = np.inf
        for weight in np.linspace(0.0, self.max_prior_blend_weight, 8):
            survival = np.minimum.accumulate(np.clip((1.0 - weight) * isotonic + weight * priors, 0.0, 1.0), axis=1)
            probabilities = _survival_matrix_to_probabilities(survival)
            score = float(np.mean(-np.log(np.maximum(probabilities[np.arange(len(rows)), actual_bins], 1e-12))))
            if score < best_score:
                best_score = score
                best_weight = float(weight)
        return best_weight


@dataclass
class MetarTmaxHybridModel:
    """Blend calibrated ML remaining-upside distribution with empirical hourly/seasonal priors."""

    base_model: MetarTmaxUpsideModel
    phase_priors: dict[str, np.ndarray]
    global_prior: np.ndarray
    blend_weight: float = 0.35
    model_version: str = "lfpb_metar_tmax_hybrid_v1"

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        base_dist = self.base_model.predict_distribution(feature_row)
        prior_dist = self.phase_prior_distribution(feature_row)
        return mix_distributions(base_dist, prior_dist, self.blend_weight)

    def phase_prior_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        current_max = _required_float(feature_row, "current_metar_max_c")
        hour = int(float(feature_row.get("local_issue_hour", 0)))
        season = feature_row.get("season") or _season(feature_row.get("target_date_local"))
        samples = self.phase_priors.get(_context_key(hour, season))
        if samples is None or len(samples) < 30:
            samples = self.phase_priors.get(_context_key(hour, "all"))
        if samples is None or len(samples) == 0:
            samples = self.global_prior
        if samples is None or len(samples) == 0:
            samples = np.array([0.0])
        rounded = np.rint(current_max + np.clip(np.asarray(samples, dtype=float), 0.0, self.base_model.max_upside_c)).astype(int)
        bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
        probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
        return TmaxDistribution(bins, probabilities)

    def to_metadata(self) -> dict:
        return {
            "model_version": self.model_version,
            "model_family": "metar_tmax_hybrid_remaining_upside",
            "blend_weight_phase_prior": self.blend_weight,
            "base_model_training_rows": self.base_model.training_rows,
            "phase_prior_context_count": len(self.phase_priors),
            "base_calibration": None if self.base_model.calibrator is None else self.base_model.calibrator.to_metadata(),
        }


@dataclass
class IconD2MetarTmaxEnsemble:
    """Blend an ICON-aware ML model with empirical ICON residual distributions."""

    ml_model: "MetarTmaxUpsideModel"
    residuals_by_hour: dict[int, np.ndarray]
    ml_weight: float = 0.50
    model_version: str = "lfpb_metar_tmax_icon_d2_v1"

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        ml_dist = self.ml_model.predict_distribution(feature_row)
        residual_dist = self.residual_distribution(feature_row)
        return mix_distributions(residual_dist, ml_dist, self.ml_weight)

    def residual_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        model_tmax = _required_float(feature_row, "model_tmax_c")
        current_max = _required_float(feature_row, "current_metar_max_c")
        hour = int(float(feature_row.get("local_issue_hour", -1)))
        samples = self.residuals_by_hour.get(hour)
        if samples is None or len(samples) < 20:
            samples = self.residuals_by_hour.get(-1, np.array([0.0]))
        rounded = np.rint(model_tmax + np.asarray(samples, dtype=float)).astype(int)
        bins = np.arange(int(rounded.min()), int(rounded.max()) + 1)
        probabilities = np.array([(rounded == bin_c).sum() for bin_c in bins], dtype=float)
        return TmaxDistribution(bins, probabilities).truncate_below(current_max)

    def to_metadata(self) -> dict:
        return {
            "model_version": self.model_version,
            "model_family": "icon_d2_metar_tmax_ensemble",
            "ml_weight": self.ml_weight,
            "residual_context_count": len(self.residuals_by_hour),
            "base_model_training_rows": self.ml_model.training_rows,
            "base_calibration": None if self.ml_model.calibrator is None else self.ml_model.calibrator.to_metadata(),
        }


@dataclass
class MetarTmaxUpsideModel:
    """Ordinal model for final daily METAR Tmax as current max + remaining upside."""

    max_upside_c: int = 12
    min_rows: int = 500
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_METAR_TMAX_FEATURES))
    max_iter: int = 70
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True))
    threshold_models: dict[int, HistGradientBoostingClassifier | None] = field(default_factory=dict)
    constant_probabilities: dict[int, float] = field(default_factory=dict)
    calibrator: MetarTmaxSurvivalCalibrator | None = None
    training_rows: int = 0
    fitted: bool = False

    def fit(self, dataset: pd.DataFrame) -> "MetarTmaxUpsideModel":
        frame = prepare_metar_tmax_dataset(dataset)
        if len(frame) < self.min_rows:
            raise ValueError(f"METAR Tmax model requires at least {self.min_rows} rows")
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
                learning_rate=0.05,
                max_iter=self.max_iter,
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
            raise ValueError("METAR Tmax model is not fitted")
        frame = pd.DataFrame([dict(feature_row)])
        X = self.imputer.transform(_numeric_feature_frame(frame, self.feature_columns))
        values = []
        for threshold in range(1, self.max_upside_c + 1):
            model = self.threshold_models.get(threshold)
            probability = self.constant_probabilities.get(threshold, 0.0) if model is None else float(model.predict_proba(X)[0, 1])
            values.append(probability)
        monotonic = np.minimum.accumulate(np.clip(values, 0.0, 1.0))
        return {threshold: float(monotonic[threshold - 1]) for threshold in range(1, self.max_upside_c + 1)}

    def predict_upside_survival_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("METAR Tmax model is not fitted")
        X = self.imputer.transform(_numeric_feature_frame(frame, self.feature_columns))
        columns = {}
        for threshold in range(1, self.max_upside_c + 1):
            model = self.threshold_models.get(threshold)
            if model is None:
                probability = self.constant_probabilities.get(threshold, 0.0)
                columns[threshold] = np.full(len(frame), probability, dtype=float)
            else:
                columns[threshold] = model.predict_proba(X)[:, 1].astype(float)
        matrix = np.column_stack([columns[threshold] for threshold in range(1, self.max_upside_c + 1)])
        monotonic = np.minimum.accumulate(np.clip(matrix, 0.0, 1.0), axis=1)
        return pd.DataFrame(
            monotonic,
            index=frame.index,
            columns=[f"probability_upside_ge_{threshold}c" for threshold in range(1, self.max_upside_c + 1)],
        )

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        current_max = _required_float(feature_row, "current_metar_max_c")
        raw_survival = self.predict_upside_survival(feature_row)
        season = feature_row.get("season") or _season(feature_row.get("target_date_local"))
        survival = (
            self.calibrator.transform(
                raw_survival,
                local_issue_hour=feature_row.get("local_issue_hour"),
                season=season,
            )
            if self.calibrator is not None
            else raw_survival
        )
        probs = survival_to_probabilities(survival, self.max_upside_c)
        bins = np.rint(current_max + np.arange(self.max_upside_c + 1)).astype(int)
        return TmaxDistribution(bins, probs)


def prepare_metar_tmax_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    required = {"final_metar_tmax_c", "current_metar_max_c", "remaining_upside_c", "issue_time_utc", "target_date_local"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise ValueError(f"dataset missing METAR Tmax columns: {missing}")
    frame = dataset.copy()
    frame["final_metar_tmax_c"] = pd.to_numeric(frame["final_metar_tmax_c"], errors="coerce")
    frame["current_metar_max_c"] = pd.to_numeric(frame["current_metar_max_c"], errors="coerce")
    frame["remaining_upside_c"] = pd.to_numeric(frame["remaining_upside_c"], errors="coerce").clip(lower=0.0)
    frame = frame.dropna(subset=["final_metar_tmax_c", "current_metar_max_c", "remaining_upside_c"]).copy()
    if "leakage_check_passed" in frame.columns:
        frame = frame[frame["leakage_check_passed"].fillna(False).astype(bool)].copy()
    frame["issue_time_utc"] = pd.to_datetime(frame["issue_time_utc"], utc=True, errors="coerce")
    frame["target_date_local"] = pd.to_datetime(frame["target_date_local"], errors="coerce")
    frame["month"] = frame["target_date_local"].dt.month
    doy = frame["target_date_local"].dt.dayofyear.fillna(1).to_numpy(dtype=float)
    frame["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    frame["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return frame.reset_index(drop=True)


def survival_to_probabilities(survival: dict[int, float], max_upside_c: int) -> np.ndarray:
    survival_values = np.array([survival.get(threshold, 0.0) for threshold in range(1, max_upside_c + 1)], dtype=float)
    probs = np.empty(max_upside_c + 1, dtype=float)
    probs[0] = 1.0 - survival_values[0]
    probs[1:-1] = survival_values[:-1] - survival_values[1:]
    probs[-1] = survival_values[-1]
    probs = np.clip(probs, 0.0, 1.0)
    total = probs.sum()
    return probs / total if total > 0 else np.r_[1.0, np.zeros(max_upside_c)]


def mix_distributions(base: TmaxDistribution, prior: TmaxDistribution, prior_weight: float) -> TmaxDistribution:
    weight = float(np.clip(prior_weight, 0.0, 1.0))
    bins = np.arange(min(base.bins_c.min(), prior.bins_c.min()), max(base.bins_c.max(), prior.bins_c.max()) + 1)
    base_probs = np.zeros(len(bins), dtype=float)
    prior_probs = np.zeros(len(bins), dtype=float)
    base_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(base.bins_c, base.probabilities)}
    prior_lookup = {int(bin_c): float(probability) for bin_c, probability in zip(prior.bins_c, prior.probabilities)}
    for idx, bin_c in enumerate(bins):
        base_probs[idx] = base_lookup.get(int(bin_c), 0.0)
        prior_probs[idx] = prior_lookup.get(int(bin_c), 0.0)
    return TmaxDistribution(bins, (1.0 - weight) * base_probs + weight * prior_probs)


def _survival_matrix_to_probabilities(survival: np.ndarray) -> np.ndarray:
    probs = np.empty((survival.shape[0], survival.shape[1] + 1), dtype=float)
    probs[:, 0] = 1.0 - survival[:, 0]
    probs[:, 1:-1] = survival[:, :-1] - survival[:, 1:]
    probs[:, -1] = survival[:, -1]
    probs = np.clip(probs, 0.0, 1.0)
    totals = probs.sum(axis=1)
    probs[totals > 0] = probs[totals > 0] / totals[totals > 0, None]
    probs[totals <= 0, 0] = 1.0
    return probs


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        if values.dtype == bool:
            data[column] = values.astype(float)
        else:
            data[column] = pd.to_numeric(values, errors="coerce")
    return pd.DataFrame(data, index=frame.index)


def _required_float(row: dict | pd.Series, key: str) -> float:
    value = row.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"METAR Tmax model requires {key}")
    return float(value)


def _season(value) -> str:
    if value is None or pd.isna(value):
        return "unknown"
    month = pd.Timestamp(value).month
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "autumn"


def _context_key(local_issue_hour, season) -> str:
    return f"{int(local_issue_hour)}|{season}"
