from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression

from weather_tmax_bot.models.distribution import TmaxDistribution
from weather_tmax_bot.models.metar_tmax_model import (
    DEFAULT_METAR_TMAX_FEATURES,
    prepare_metar_tmax_dataset,
    survival_to_probabilities,
)


@dataclass
class DiscreteHazardCalibrator:
    max_upside_c: int = 12
    min_rows_per_threshold: int = 120
    threshold_calibrators: dict[int, IsotonicRegression | None] = field(default_factory=dict)
    threshold_rows: dict[int, int] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, rows: pd.DataFrame) -> "DiscreteHazardCalibrator":
        self.threshold_calibrators = {}
        self.threshold_rows = {}
        for threshold in range(1, self.max_upside_c + 1):
            prob_col = f"raw_hazard_upside_ge_{threshold}c"
            actual_col = f"actual_hazard_upside_ge_{threshold}c"
            at_risk_col = f"actual_upside_ge_{threshold - 1}c"
            needed = {prob_col, actual_col, at_risk_col}
            frame = rows[list(needed)].dropna() if needed.issubset(rows.columns) else pd.DataFrame()
            frame = frame[frame[at_risk_col].astype(bool)].copy()
            self.threshold_rows[threshold] = len(frame)
            if len(frame) < self.min_rows_per_threshold or frame[actual_col].nunique() < 2:
                self.threshold_calibrators[threshold] = None
                continue
            model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            model.fit(frame[prob_col].to_numpy(dtype=float), frame[actual_col].to_numpy(dtype=float))
            self.threshold_calibrators[threshold] = model
        self.fitted = any(model is not None for model in self.threshold_calibrators.values())
        return self

    def transform(self, hazards: dict[int, float]) -> dict[int, float]:
        if not self.fitted:
            return hazards
        out = {}
        for threshold in range(1, self.max_upside_c + 1):
            raw = float(np.clip(hazards.get(threshold, 0.0), 0.0, 1.0))
            model = self.threshold_calibrators.get(threshold)
            out[threshold] = raw if model is None else float(model.predict(np.array([raw], dtype=float))[0])
        return out

    def to_metadata(self) -> dict:
        return {
            "calibration_method": "discrete_hazard_isotonic_by_threshold",
            "max_upside_c": self.max_upside_c,
            "min_rows_per_threshold": self.min_rows_per_threshold,
            "fitted": self.fitted,
            "threshold_rows": self.threshold_rows,
            "calibrated_thresholds": [
                int(threshold) for threshold, model in self.threshold_calibrators.items() if model is not None
            ],
        }


@dataclass
class DiscreteHazardUpsideModel:
    max_upside_c: int = 12
    min_rows: int = 500
    min_at_risk_rows: int = 80
    feature_columns: list[str] = field(default_factory=lambda: list(DEFAULT_METAR_TMAX_FEATURES))
    max_iter: int = 70
    imputer: SimpleImputer = field(default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True))
    hazard_models: dict[int, HistGradientBoostingClassifier | None] = field(default_factory=dict)
    constant_hazards: dict[int, float] = field(default_factory=dict)
    calibrator: DiscreteHazardCalibrator | None = None
    training_rows: int = 0
    fitted: bool = False

    def fit(self, dataset: pd.DataFrame) -> "DiscreteHazardUpsideModel":
        frame = prepare_metar_tmax_dataset(dataset)
        if len(frame) < self.min_rows:
            raise ValueError(f"Discrete hazard model requires at least {self.min_rows} rows")
        X_all = self.imputer.fit_transform(_numeric_feature_frame(frame, self.feature_columns))
        upside = frame["remaining_upside_c"].to_numpy(dtype=float)
        self.hazard_models = {}
        self.constant_hazards = {}
        for threshold in range(1, self.max_upside_c + 1):
            at_risk = upside >= float(threshold - 1)
            labels = (upside[at_risk] >= float(threshold)).astype(int)
            if len(labels) < self.min_at_risk_rows or np.unique(labels).size < 2:
                self.hazard_models[threshold] = None
                self.constant_hazards[threshold] = float(labels.mean()) if len(labels) else 0.0
                continue
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=self.max_iter,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=42,
            )
            model.fit(X_all[at_risk], labels)
            self.hazard_models[threshold] = model
        self.training_rows = len(frame)
        self.fitted = True
        return self

    def predict_hazards(self, feature_row: dict | pd.Series) -> dict[int, float]:
        if not self.fitted:
            raise ValueError("Discrete hazard model is not fitted")
        frame = pd.DataFrame([dict(feature_row)])
        X = self.imputer.transform(_numeric_feature_frame(frame, self.feature_columns))
        hazards = {}
        for threshold in range(1, self.max_upside_c + 1):
            model = self.hazard_models.get(threshold)
            probability = self.constant_hazards.get(threshold, 0.0) if model is None else float(model.predict_proba(X)[0, 1])
            hazards[threshold] = float(np.clip(probability, 0.0, 1.0))
        return self.calibrator.transform(hazards) if self.calibrator is not None else hazards

    def predict_hazard_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Discrete hazard model is not fitted")
        X = self.imputer.transform(_numeric_feature_frame(frame, self.feature_columns))
        columns = {}
        for threshold in range(1, self.max_upside_c + 1):
            model = self.hazard_models.get(threshold)
            if model is None:
                columns[threshold] = np.full(len(frame), self.constant_hazards.get(threshold, 0.0), dtype=float)
            else:
                columns[threshold] = model.predict_proba(X)[:, 1].astype(float)
        return pd.DataFrame(
            columns,
            index=frame.index,
        ).rename(columns={threshold: f"hazard_upside_ge_{threshold}c" for threshold in columns})

    def predict_upside_survival(self, feature_row: dict | pd.Series) -> dict[int, float]:
        hazards = self.predict_hazards(feature_row)
        survival = {}
        running = 1.0
        for threshold in range(1, self.max_upside_c + 1):
            running *= float(np.clip(hazards.get(threshold, 0.0), 0.0, 1.0))
            survival[threshold] = running
        return survival

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        current_max = _required_float(feature_row, "current_metar_max_c")
        survival = self.predict_upside_survival(feature_row)
        probs = survival_to_probabilities(survival, self.max_upside_c)
        bins = np.rint(current_max + np.arange(self.max_upside_c + 1)).astype(int)
        return TmaxDistribution(bins, probs)

    def to_metadata(self) -> dict:
        return {
            "model_family": "discrete_hazard_remaining_upside",
            "max_upside_c": self.max_upside_c,
            "min_rows": self.min_rows,
            "min_at_risk_rows": self.min_at_risk_rows,
            "feature_count": len(self.feature_columns),
            "training_rows": self.training_rows,
            "calibration": None if self.calibrator is None else self.calibrator.to_metadata(),
        }


def hazard_calibration_rows(model: DiscreteHazardUpsideModel, frame: pd.DataFrame) -> pd.DataFrame:
    raw = model.predict_hazard_frame(frame)
    rows = []
    for index, row in frame.iterrows():
        out = {
            "target_date_local": str(row["target_date_local"]),
            "issue_time_utc": pd.Timestamp(row["issue_time_utc"]).isoformat(),
            "local_issue_hour": int(row["local_issue_hour"]),
            "remaining_upside_c": float(row["remaining_upside_c"]),
        }
        for threshold in range(1, model.max_upside_c + 1):
            out[f"raw_hazard_upside_ge_{threshold}c"] = float(raw.loc[index, f"hazard_upside_ge_{threshold}c"])
            out[f"actual_hazard_upside_ge_{threshold}c"] = float(row["remaining_upside_c"] >= threshold)
            out[f"actual_upside_ge_{threshold - 1}c"] = float(row["remaining_upside_c"] >= threshold - 1)
        rows.append(out)
    return pd.DataFrame(rows)


def _numeric_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = {}
    for column in columns:
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        data[column] = values.astype(float) if values.dtype == bool else pd.to_numeric(values, errors="coerce")
    return pd.DataFrame(data, index=frame.index)


def _required_float(row: dict | pd.Series, key: str) -> float:
    value = row.get(key)
    if value is None or pd.isna(value):
        raise ValueError(f"Discrete hazard model requires {key}")
    return float(value)
