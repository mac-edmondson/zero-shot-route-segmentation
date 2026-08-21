"""Public contracts for reproducible preprocessing datasets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from ..interfaces.augmentation import AugmentationPlan
from ..interfaces.data_models import ImageRecord


class MaterializationMode(str, Enum):
    STATIC = "static"
    ON_THE_FLY = "on_the_fly"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    splits: tuple[str, ...] = ("test",)
    materialization: MaterializationMode = MaterializationMode.ON_THE_FLY
    augmentation_plan: AugmentationPlan | None = None
    seed: int = 0
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string.")
        splits = tuple(self.splits)
        if (
            not splits
            or len(set(splits)) != len(splits)
            or any(split not in {"train", "val", "test"} for split in splits)
        ):
            raise ValueError(
                "splits must be a non-empty unique subset of train, val, test."
            )
        if not isinstance(self.materialization, MaterializationMode):
            raise TypeError("materialization must be MaterializationMode.")
        if self.augmentation_plan is not None and not isinstance(
            self.augmentation_plan, AugmentationPlan
        ):
            raise TypeError("augmentation_plan must be AugmentationPlan or None.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DatasetProvider(Protocol):
    def iter_split(self, split: str) -> Iterable[ImageRecord]: ...
    def describe(self) -> DatasetSpec: ...


class DataPreprocessingPipelineProtocol(Protocol):
    def build(
        self, sources: Sequence[ImageRecord], spec: DatasetSpec
    ) -> DatasetProvider: ...
