from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weather_tmax_bot.models.distribution import (
    TmaxDistribution,
    project_unimodal_distribution,
    temperature_scale_distribution,
)


REGRESSION_COMPONENTS = ("ridge", "hist_gradient_boosting", "extra_trees")


@dataclass
class RegressionResidualTmaxModel:
    """Residual regressors with calibration-only error distributions."""

    feature_columns: list[str]
    selected_components: tuple[str, ...] = REGRESSION_COMPONENTS
    shape_strategy: str = "raw"
    unimodal_temperature: float = 0.67
    estimators: dict[str, object] = field(default_factory=dict)
    imputer: SimpleImputer = field(
        default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    )
    residuals_by_hour: dict[int, np.ndarray] = field(default_factory=dict)
    fitted: bool = False

    def fit(self, train: pd.DataFrame, calibration: pd.DataFrame) -> "RegressionResidualTmaxModel":
        X_train = self.imputer.fit_transform(_numeric_features(train, self.feature_columns))
        target = (
            pd.to_numeric(train["final_metar_tmax_c"], errors="coerce")
            - pd.to_numeric(train["model_tmax_c"], errors="coerce")
        ).to_numpy(dtype=float)
        self.estimators = {
            "ridge": make_pipeline(StandardScaler(), Ridge(alpha=8.0)),
            "hist_gradient_boosting": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=160,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=42,
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=240,
                min_samples_leaf=5,
                max_features=0.75,
                n_jobs=1,
                random_state=42,
            ),
        }
        for estimator in self.estimators.values():
            estimator.fit(X_train, target)

        self.fitted = True
        self._calibrate_residuals(calibration)
        return self

    def with_components(
        self,
        components: tuple[str, ...],
        calibration: pd.DataFrame,
    ) -> "RegressionResidualTmaxModel":
        model = RegressionResidualTmaxModel(
            feature_columns=self.feature_columns,
            selected_components=components,
            shape_strategy=self.shape_strategy,
            unimodal_temperature=self.unimodal_temperature,
            estimators=self.estimators,
            imputer=self.imputer,
            residuals_by_hour=self.residuals_by_hour,
            fitted=self.fitted,
        )
        model._calibrate_residuals(calibration)
        return model

    def _calibrate_residuals(self, calibration: pd.DataFrame) -> None:
        frame = calibration.copy()
        frame["regression_error_c"] = (
            pd.to_numeric(frame["final_metar_tmax_c"], errors="coerce")
            - self.predict_expected_frame(frame)
        )
        self.residuals_by_hour = {
            -1: frame["regression_error_c"].dropna().to_numpy(dtype=float)
        }
        for hour, group in frame.groupby("local_issue_hour"):
            values = group["regression_error_c"].dropna().to_numpy(dtype=float)
            if len(values) >= 20:
                self.residuals_by_hour[int(hour)] = values

    def predict_expected_frame(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.estimators:
            raise ValueError("Regression Tmax model is not fitted")
        X = self.imputer.transform(_numeric_features(frame, self.feature_columns))
        corrections = np.column_stack(
            [self.estimators[name].predict(X) for name in self.selected_components]
        )
        baseline = pd.to_numeric(frame["model_tmax_c"], errors="coerce").to_numpy(dtype=float)
        current_max = pd.to_numeric(frame["current_metar_max_c"], errors="coerce").to_numpy(dtype=float)
        return np.maximum(current_max, baseline + corrections.mean(axis=1))

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        frame = pd.DataFrame([dict(feature_row)])
        expected = float(self.predict_expected_frame(frame)[0])
        current_max = float(feature_row["current_metar_max_c"])
        hour = int(float(feature_row.get("local_issue_hour", -1)))
        residuals = self.residuals_by_hour.get(hour, self.residuals_by_hour.get(-1))
        if residuals is None or len(residuals) == 0:
            residuals = np.asarray([0.0], dtype=float)
        values = np.rint(np.maximum(current_max, expected + residuals)).astype(int)
        bins = np.arange(int(values.min()), int(values.max()) + 1)
        probabilities = np.asarray([(values == value).sum() for value in bins], dtype=float)
        distribution = TmaxDistribution(bins, probabilities)
        if self.shape_strategy == "temperature_unimodal":
            return project_unimodal_distribution(
                temperature_scale_distribution(distribution, self.unimodal_temperature)
            )
        if self.shape_strategy == "unimodal":
            return project_unimodal_distribution(distribution)
        return distribution

    def component_predictions(self, feature_row: dict | pd.Series) -> dict[str, float]:
        frame = pd.DataFrame([dict(feature_row)])
        X = self.imputer.transform(_numeric_features(frame, self.feature_columns))
        baseline = float(feature_row["model_tmax_c"])
        current_max = float(feature_row["current_metar_max_c"])
        return {
            name: max(current_max, baseline + float(estimator.predict(X)[0]))
            for name, estimator in self.estimators.items()
        }


def _numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: pd.to_numeric(
                frame[column] if column in frame else pd.Series(np.nan, index=frame.index),
                errors="coerce",
            )
            for column in columns
        },
        index=frame.index,
    )
