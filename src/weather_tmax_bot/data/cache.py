from __future__ import annotations

from pathlib import Path


class LocalCache:
    def __init__(self, root: str | Path = "data/cache"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, source_id: str, *parts: str) -> Path:
        clean_source = source_id.replace("/", "_").replace("..", "_")
        path = self.root / clean_source
        for part in parts:
            path = path / part.replace("/", "_").replace("..", "_")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
