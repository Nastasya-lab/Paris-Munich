from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path("data/cache/matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_pit_histogram(pit_values, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.hist(pit_values, bins=10, range=(0, 1))
    plt.xlabel("PIT")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p


def save_reliability_curve(table: pd.DataFrame, path: str | Path, title: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(table["mean_probability"], table["observed_frequency"], s=30 + table["count"].fillna(0))
    plt.xlabel("Forecast probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return p
