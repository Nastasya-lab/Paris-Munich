from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    ensure_parent(p)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))


def append_jsonl(line: str, path: str | Path) -> Path:
    p = Path(path)
    ensure_parent(p)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")
    return p
