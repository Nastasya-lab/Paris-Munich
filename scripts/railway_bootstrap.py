from __future__ import annotations

import os
import shutil
from pathlib import Path


DATA_DIRS = ("raw", "interim", "processed", "forecasts", "cache", "models", "reports", "logs")


def main() -> None:
    data_dir = Path(os.getenv("WEATHER_TMAX_DATA_DIR", "data"))
    seed_dir = Path(os.getenv("WEATHER_TMAX_SEED_DATA_DIR", "seed_data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_DIRS:
        (data_dir / name).mkdir(parents=True, exist_ok=True)
    if seed_dir.exists():
        _seed_if_missing(seed_dir, data_dir)
    print(f"Railway bootstrap complete: data_dir={data_dir.resolve()}")


def _seed_if_missing(seed_dir: Path, data_dir: Path) -> None:
    for name in DATA_DIRS:
        source = seed_dir / name
        target = data_dir / name
        if not source.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            destination = target / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination)
            else:
                shutil.copy2(item, destination)


if __name__ == "__main__":
    main()
