from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from weather_tmax_bot.models.distribution import TmaxDistribution, project_unimodal_distribution


@dataclass
class HfIconEuResidualPmfShadowModel:
    residual_samples_by_run_hour: dict[int, np.ndarray] = field(default_factory=dict)
    global_residual_samples: np.ndarray | None = None
    model_version: str = "lfpb_hf_icon_eu_residual_pmf_shadow_v1"
    minimum_run_hour_samples: int = 30

    def fit(self, frame: pd.DataFrame) -> "HfIconEuResidualPmfShadowModel":
        rows = frame.dropna(subset=["metar_tmax_c", "model_tmax_c", "model_run_hour"]).copy()
        if rows.empty:
            raise ValueError("HF ICON-EU shadow model requires metar_tmax_c, model_tmax_c, and model_run_hour")
        residual = rows["metar_tmax_c"].astype(float) - rows["model_tmax_c"].astype(float)
        self.global_residual_samples = residual.to_numpy(dtype=float)
        self.residual_samples_by_run_hour = {}
        for run_hour, group in rows.groupby(rows["model_run_hour"].astype(int)):
            group_residual = group["metar_tmax_c"].astype(float) - group["model_tmax_c"].astype(float)
            if len(group_residual) >= self.minimum_run_hour_samples:
                self.residual_samples_by_run_hour[int(run_hour)] = group_residual.to_numpy(dtype=float)
        return self

    def predict_distribution(self, feature_row: dict | pd.Series) -> TmaxDistribution:
        row = dict(feature_row)
        model_tmax = _float_or_none(row.get("model_tmax_c"))
        if model_tmax is None:
            raise ValueError("HF ICON-EU shadow prediction requires model_tmax_c")
        residuals = self._residual_samples(row.get("model_run_hour"))
        centered = residuals - float(np.mean(residuals))
        samples = model_tmax + float(np.mean(residuals)) + centered
        bins = np.arange(int(np.floor(samples.min())) - 1, int(np.ceil(samples.max())) + 2)
        rounded = np.rint(samples).astype(int)
        probabilities = np.array([(rounded == bin_c).mean() for bin_c in bins], dtype=float)
        distribution = project_unimodal_distribution(TmaxDistribution(bins, probabilities))
        return distribution.truncate_below(_float_or_none(row.get("current_metar_max_c")))

    def to_metadata(self) -> dict:
        return {
            "model_family": "hf_icon_eu_residual_pmf_shadow",
            "model_version": self.model_version,
            "run_hours": sorted(int(hour) for hour in self.residual_samples_by_run_hour),
            "samples_by_run_hour": {
                str(int(hour)): int(len(samples))
                for hour, samples in sorted(self.residual_samples_by_run_hour.items())
            },
            "global_samples": 0 if self.global_residual_samples is None else int(len(self.global_residual_samples)),
            "minimum_run_hour_samples": int(self.minimum_run_hour_samples),
        }

    def _residual_samples(self, run_hour_raw) -> np.ndarray:
        if run_hour_raw is not None and not pd.isna(run_hour_raw):
            samples = self.residual_samples_by_run_hour.get(int(run_hour_raw))
            if samples is not None and len(samples) > 0:
                return samples
        if self.global_residual_samples is None or len(self.global_residual_samples) == 0:
            raise ValueError("HF ICON-EU shadow model is not fitted")
        return self.global_residual_samples


def prepare_hf_icon_eu_training_frame(frame: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["target_date_local"] = pd.to_datetime(rows["target_date_local"], errors="coerce").dt.date
    rows["model_run_time_utc"] = pd.to_datetime(rows["model_run_time_utc"], utc=True, errors="coerce")
    rows["model_run_hour"] = rows["model_run_time_utc"].dt.hour
    rows["model_tmax_c"] = pd.to_numeric(rows["model_tmax_c"], errors="coerce")
    truth = target.copy()
    truth["target_date_local"] = pd.to_datetime(truth["target_date_local"], errors="coerce").dt.date
    truth = truth[["target_date_local", "metar_tmax_c"]]
    rows = rows.merge(truth, on="target_date_local", how="left")
    rows["metar_tmax_c"] = pd.to_numeric(rows["metar_tmax_c"], errors="coerce")
    rows = rows.dropna(subset=["target_date_local", "model_run_time_utc", "model_run_hour", "model_tmax_c", "metar_tmax_c"])
    return rows.sort_values(["target_date_local", "model_run_hour"]).reset_index(drop=True)


def build_hf_icon_eu_live_feature_row(nwp_features: dict, base_feature_row: dict, issue_time_utc) -> dict:
    row = dict(nwp_features)
    issue = pd.Timestamp(issue_time_utc)
    if issue.tzinfo is None:
        issue = issue.tz_localize("UTC")
    run_time = pd.to_datetime(row.get("model_run_time_utc") or row.get("forecast_reference_time"), utc=True, errors="coerce")
    if pd.isna(run_time):
        row["model_run_hour"] = _latest_supported_training_run_hour(issue.hour)
    else:
        row["model_run_hour"] = int(run_time.hour)
    row["current_metar_max_c"] = base_feature_row.get("current_metar_max_c")
    row["latest_metar_temp_c"] = base_feature_row.get("latest_metar_temp_c")
    row["local_issue_hour"] = base_feature_row.get("local_issue_hour")
    row["target_date_local"] = base_feature_row.get("target_date_local")
    return row


def _latest_supported_training_run_hour(issue_hour_utc: int) -> int:
    for hour in [12, 6, 0]:
        if issue_hour_utc >= hour:
            return hour
    return 12


def _float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
