"""Reproducible clean and augmented in-memory dataset providers."""
from __future__ import annotations
from collections.abc import Iterable, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import Callable
from ..interfaces.data_models import ImageRecord
from ..interfaces.data_preprocessing import DatasetProvider, DatasetSpec, MaterializationMode
from .augmentation_suite import AugmentationSuite

class _Provider:
    def __init__(self, spec: DatasetSpec, records: Sequence[ImageRecord], derive: Callable[[ImageRecord], ImageRecord | None]) -> None:
        self._spec, self._records, self._derive = spec, tuple(records), derive
    def describe(self) -> DatasetSpec: return self._spec
    def iter_split(self, split: str) -> Iterable[ImageRecord]:
        if split not in self._spec.splits: raise ValueError(f"Unknown split '{split}'.")
        for record in self._records:
            if record.split == split:
                yield replace(record, image=record.image.copy())
                derived = self._derive(record)
                if derived is not None: yield replace(derived, image=derived.image.copy())

class DataPreprocessingPipeline:
    """Create split-stable clean and augmented records without mutating inputs."""
    _WEIGHTS = {"train": 0.8, "val": 0.1, "test": 0.1}

    def __init__(self, augmentation_suite: AugmentationSuite | None = None) -> None:
        self.augmentation_suite = augmentation_suite or AugmentationSuite()

    def _split(self, record: ImageRecord, spec: DatasetSpec) -> str:
        if record.split is not None:
            if record.split not in spec.splits: raise ValueError(f"Record '{record.image_id}' has split '{record.split}', absent from DatasetSpec.")
            return record.split
        weights = [self._WEIGHTS[split] for split in spec.splits]; total = sum(weights)
        value = int.from_bytes(sha256(f"{spec.seed}:{record.source_id}".encode()).digest()[:8], "big") / 2**64
        boundary = 0.0
        for split, weight in zip(spec.splits, weights, strict=True):
            boundary += weight / total
            if value < boundary: return split
        return spec.splits[-1]

    @staticmethod
    def _recipe_metadata(spec: DatasetSpec) -> dict[str, object]:
        plan = spec.augmentation_plan
        return {"dataset_id": spec.dataset_id, "seed": plan.seed if plan and plan.seed is not None else spec.seed, "augmentations": tuple(recipe.name for recipe in plan.augmentations) if plan else ()}

    def _derived(self, record: ImageRecord, spec: DatasetSpec) -> ImageRecord | None:
        plan = spec.augmentation_plan
        if plan is None or not plan.augmentations: return None
        if plan.seed is None: plan = replace(plan, seed=spec.seed)
        image = self.augmentation_suite.apply_plan(record.image, plan)
        fingerprint = sha256(repr((record.source_id, spec.dataset_id, plan)).encode()).hexdigest()[:12]
        return ImageRecord(image_id=f"{record.image_id}:aug:{fingerprint}", image=image, annotations=record.annotations, split=record.split, source_id=record.source_id, parent_image_id=record.image_id, condition="augmented", augmentation_metadata=self._recipe_metadata(spec))

    def build(self, sources: Sequence[ImageRecord], spec: DatasetSpec) -> DatasetProvider:
        if not isinstance(spec, DatasetSpec): raise TypeError("spec must be DatasetSpec.")
        records = tuple(sources)
        if any(not isinstance(record, ImageRecord) for record in records): raise TypeError("sources must contain ImageRecord values.")
        clean = tuple(replace(record, image=record.image.copy(), split=self._split(record, spec), source_id=record.source_id, condition="clean", augmentation_metadata={}) for record in records)
        if spec.materialization is MaterializationMode.STATIC:
            derived = {record.image_id: self._derived(record, spec) for record in clean}
            return _Provider(spec, clean, lambda record: derived[record.image_id])
        return _Provider(spec, clean, lambda record: self._derived(record, spec))
