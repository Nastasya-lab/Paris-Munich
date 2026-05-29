from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from weather_tmax_bot.models.distribution import TmaxDistribution, quantiles_to_distribution


DEFAULT_QUANTILES = [
    0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99,
]


class QuantileTmaxModel:
    def __init__(self, quantiles: list[float] | None = None, random_state: int = 42):
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.random_state = random_state
        self.models: dict[float, GradientBoostingRegressor] = {}
        self.feature_columns: list[str] = []
        self.feature_fill_values: pd.Series | None = None
        self.feature_ranges: dict[str, dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "QuantileTmaxModel":
        numeric = X.select_dtypes(include=["number", "bool"]).copy()
        self.feature_columns = numeric.columns.tolist()
        self.feature_fill_values = numeric.median(numeric_only=True).fillna(0)
        self.feature_ranges = {
            col: {"min": float(numeric[col].min()), "max": float(numeric[col].max())}
            for col in self.feature_columns
            if numeric[col].notna().any()
        }
        numeric = numeric.fillna(self.feature_fill_values).fillna(0)
        for q in self.quantiles:
            model = GradientBoostingRegressor(loss="quantile", alpha=q, random_state=self.random_state)
            model.fit(numeric, y)
            self.models[q] = model
        return self

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        numeric = X.reindex(columns=self.feature_columns).fillna(np.nan)
        fill_values = self.feature_fill_values
        if fill_values is None:
            fill_values = numeric.median(numeric_only=True).fillna(0)
        numeric = numeric.fillna(fill_values).fillna(0)
        return np.column_stack([self.models[q].predict(numeric) for q in self.quantiles])

    def predict_distribution(self, X_one: pd.DataFrame, observed_max_so_far: float | None = None) -> TmaxDistribution:
        values = self.predict_quantiles(X_one)[0]
        return quantiles_to_distribution(self.quantiles, values).truncate_below(observed_max_so_far)
