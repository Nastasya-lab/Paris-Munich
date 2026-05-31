from __future__ import annotations

from pathlib import Path

import pandas as pd

from weather_tmax_bot.data.storage import write_parquet
from weather_tmax_bot.evaluation.metrics import bias, crps_discrete, mae, nll_integer_bin, rmse
from weather_tmax_bot.models.quantile_model import QuantileTmaxModel


NWP_PREFIXES = ("model_",)
NWP_COLUMNS = {
    "nwp_missing",
    "latest_nwp_model_name",
    "latest_nwp_source_id",
    "max_nwp_knowledge_time_utc",
}


def main() -> None:
    dataset = pd.read_parquet("data/processed/training_dataset.parquet")
    dataset["target_date_local"] = pd.to_datetime(dataset["target_date_local"]).dt.date
    start = pd.to_datetime("2025-09-01").date()
    test_start = pd.to_datetime("2025-12-01").date()
    test_end = pd.to_datetime("2025-12-31").date()
    df = dataset[
        (dataset["target_date_local"] >= start)
        & (dataset["target_date_local"] <= test_end)
        & (dataset["nwp_missing"] == False)  # noqa: E712 - pandas boolean filtering.
    ].copy()
    train = df[df["target_date_local"] < test_start].copy()
    test = df[df["target_date_local"] >= test_start].copy()
    if train.empty or test.empty:
        raise SystemExit("NWP-aware holdout requires non-empty train and test slices")

    with_nwp = QuantileTmaxModel().fit(train.drop(columns=["tmax_c"]), train["tmax_c"])
    without_nwp = QuantileTmaxModel().fit(_drop_nwp(train.drop(columns=["tmax_c"])), train["tmax_c"])

    rows = []
    for _, row in test.iterrows():
        actual = float(row["tmax_c"])
        dist_with = with_nwp.predict_distribution(pd.DataFrame([row.drop(labels=["tmax_c"])]))
        dist_without = without_nwp.predict_distribution(_drop_nwp(pd.DataFrame([row.drop(labels=["tmax_c"])])))
        rows.append(_score("quantile_with_nwp", row, dist_with, actual))
        rows.append(_score("quantile_without_nwp", row, dist_without, actual))
        rows.append(_score_point("raw_nwp_model_tmax", row, actual))

    scored = pd.DataFrame(rows)
    summary = (
        scored.groupby("model_variant")
        .apply(_summary, include_groups=False)
        .reset_index()
        .sort_values("mae_expected")
    )
    write_parquet(scored, "data/reports/nwp_aware_holdout_rows.parquet")
    write_parquet(summary, "data/reports/nwp_aware_holdout_summary.parquet")
    _write_doc(summary, train, test)
    print(f"Wrote NWP-aware holdout comparison for {len(test)} test rows")


def _drop_nwp(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in df.columns if col in NWP_COLUMNS or col.startswith(NWP_PREFIXES)]
    return df.drop(columns=drop_cols, errors="ignore")


def _score(variant: str, row: pd.Series, dist, actual: float) -> dict:
    return {
        "model_variant": variant,
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "actual_tmax_c": actual,
        "expected_tmax_c": dist.expected_tmax_c,
        "median_tmax_c": dist.median_tmax_c,
        "nll": nll_integer_bin(dist, actual),
        "crps": crps_discrete(dist, actual),
    }


def _score_point(variant: str, row: pd.Series, actual: float) -> dict:
    predicted = float(row["model_tmax_c"])
    return {
        "model_variant": variant,
        "target_date_local": row["target_date_local"].isoformat(),
        "issue_hour_utc": int(row["issue_hour_utc"]),
        "actual_tmax_c": actual,
        "expected_tmax_c": predicted,
        "median_tmax_c": predicted,
        "nll": None,
        "crps": None,
    }


def _summary(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": len(group),
            "mae_expected": mae(group["actual_tmax_c"], group["expected_tmax_c"]),
            "rmse_expected": rmse(group["actual_tmax_c"], group["expected_tmax_c"]),
            "bias_expected": bias(group["actual_tmax_c"], group["expected_tmax_c"]),
            "mean_nll": float(group["nll"].dropna().mean()) if group["nll"].notna().any() else None,
            "mean_crps": float(group["crps"].dropna().mean()) if group["crps"].notna().any() else None,
        }
    )


def _write_doc(summary: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    lines = [
        "# NWP-aware holdout comparison",
        "",
        "Experimental comparison using only rows where Open-Meteo Single Runs ICON-D2 is available.",
        "",
        f"- train period: `{train['target_date_local'].min()}` to `{train['target_date_local'].max()}`",
        f"- test period: `{test['target_date_local'].min()}` to `{test['target_date_local'].max()}`",
        f"- train rows: `{len(train)}`",
        f"- test rows: `{len(test)}`",
        "",
        "| model_variant | rows | mae_expected | rmse_expected | bias_expected | mean_nll | mean_crps |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                str(row[col])
                for col in ["model_variant", "rows", "mae_expected", "rmse_expected", "bias_expected", "mean_nll", "mean_crps"]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "This is a short seasonal slice, not enough for final promotion by itself.",
            "It is useful for deciding whether to continue toward a production NWP-aware model.",
        ]
    )
    Path("docs/nwp_aware_holdout.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
