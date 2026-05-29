from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Source:
    source_id: str
    name: str
    type: str
    whether_forecast_as_issued: bool
    whether_allowed_for_training_features: bool
    whether_allowed_for_truth_target: bool
    metadata: dict[str, Any]


class SourceRegistry:
    def __init__(self, path: str | Path = "config/data_sources.yaml"):
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        self.registry_version = raw.get("registry_version", "unknown")
        self.sources = {
            item["source_id"]: Source(
                source_id=item["source_id"],
                name=item["name"],
                type=item["type"],
                whether_forecast_as_issued=bool(item["whether_forecast_as_issued"]),
                whether_allowed_for_training_features=bool(item["whether_allowed_for_training_features"]),
                whether_allowed_for_truth_target=bool(item["whether_allowed_for_truth_target"]),
                metadata=item,
            )
            for item in raw["sources"]
        }

    def get(self, source_id: str) -> Source:
        return self.sources[source_id]

    def require_feature_source(self, source_id: str) -> None:
        source = self.get(source_id)
        if not source.whether_allowed_for_training_features:
            raise ValueError(f"{source_id} is not allowed for training features")

    def require_truth_source(self, source_id: str) -> None:
        source = self.get(source_id)
        if not source.whether_allowed_for_truth_target:
            raise ValueError(f"{source_id} is not allowed for truth targets")
