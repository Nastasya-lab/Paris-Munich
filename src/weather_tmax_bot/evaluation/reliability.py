from __future__ import annotations

import pandas as pd


def reliability_table(probabilities, outcomes, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"probability": probabilities, "outcome": outcomes})
    df["bucket"] = pd.cut(df["probability"], bins=bins, include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        mean_probability=("probability", "mean"),
        observed_frequency=("outcome", "mean"),
        count=("outcome", "size"),
    ).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    return out
