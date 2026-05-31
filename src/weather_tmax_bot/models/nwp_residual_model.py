from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution, empirical_distribution


@dataclass
class NWPResidualDistributionModel:
    min_group_rows: int = 30
    residuals_all: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    residuals_by_month: dict[int, np.ndarray] = field(default_factory=dict)
    residuals_by_month_issue_hour: dict[tuple[int, int], np.ndarray] = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: ["model_tmax_c", "month", "issue_hour_utc", "nwp_missing"])

    def fit(self, dataset: pd.DataFrame) -> "NWPResidualDistributionModel":
        required = {"tmax_c", "model_tmax_c", "month", "issue_hour_utc"}
        missing = required.difference(dataset.columns)
        if missing:
            raise ValueError(f"dataset missing required NWP residual columns: {sorted(missing)}")
        df = dataset.copy()
        if "nwp_missing" in df.columns:
            df = df[df["nwp_missing"] == False].copy()  # noqa: E712 - pandas boolean filtering.
        df["tmax_c"] = pd.to_numeric(df["tmax_c"], errors="coerce")
        df["model_tmax_c"] = pd.to_numeric(df["model_tmax_c"], errors="coerce")
        df = df[df["tmax_c"].notna() & df["model_tmax_c"].notna()].copy()
        if len(df) < self.min_group_rows:
            raise ValueError("not enough NWP rows to fit residual distribution model")
        df["residual_c"] = df["tmax_c"] - df["model_tmax_c"]
        self.residuals_all = df["residual_c"].to_numpy(dtype=float)
        self.residuals_by_month = {
            int(month): group["residual_c"].to_numpy(dtype=float)
            for month, group in df.groupby("month")
            if len(group) >= self.min_group_rows
        }
        self.residuals_by_month_issue_hour = {
            (int(month), int(hour)): group["residual_c"].to_numpy(dtype=float)
            for (month, hour), group in df.groupby(["month", "issue_hour_utc"])
            if len(group) >= self.min_group_rows
        }
        return self

    def predict_distribution(self, X_one: pd.DataFrame, observed_max_so_far: float | None = None) -> TmaxDistribution:
        row = X_one.iloc[0]
        if bool(row.get("nwp_missing", False)) or pd.isna(row.get("model_tmax_c")):
            raise ValueError("NWP residual model requires model_tmax_c and nwp_missing=False")
        model_tmax = float(row["model_tmax_c"])
        month = int(row.get("month", 0)) if pd.notna(row.get("month")) else 0
        issue_hour = int(row.get("issue_hour_utc", -1)) if pd.notna(row.get("issue_hour_utc")) else -1
        residuals = self._residuals_for(month, issue_hour)
        return empirical_distribution(model_tmax + residuals).truncate_below(observed_max_so_far)

    def _residuals_for(self, month: int, issue_hour: int) -> np.ndarray:
        return (
            self.residuals_by_month_issue_hour.get((month, issue_hour))
            if (month, issue_hour) in self.residuals_by_month_issue_hour
            else self.residuals_by_month.get(month, self.residuals_all)
        )
