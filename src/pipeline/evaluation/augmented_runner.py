"""Build reproducible augmented records for evaluation arms."""

from __future__ import annotations

from collections.abc import Sequence

from ..interfaces.data_models import ImageRecord
from ..preprocessing.augmentation_suite import (
    AugmentationPlan,
    LightingAugmentation,
)
from ..preprocessing.data_preprocessing import DatasetSpec, MaterializationMode
from ..preprocessing.data_preprocessing_pipeline import DataPreprocessingPipeline

LIGHTING_ARMS = {
    "lighting_dark_025": -0.25,
    "lighting_dark_050": -0.50,
    "lighting_dark_075": -0.75,
}

def build_lighting_records(
    records: Sequence[ImageRecord],
    strength: float,
    *,
    seed: int = 0,
    dataset_id: str = "roboflow-evaluation",
    split: str = "test",
) -> tuple[ImageRecord, ...]:
    """Return clean records paired with one deterministic lighting arm."""
    plan = AugmentationPlan((LightingAugmentation(strength),), seed=seed)
    provider = DataPreprocessingPipeline().build(
        records,
        DatasetSpec(
            dataset_id,
            splits=(split,),
            materialization=MaterializationMode.STATIC,
            augmentation_plan=plan,
            seed=seed,
        ),
    )
    return tuple(provider.iter_split(split))

def lighting_manifest(
    records: Sequence[ImageRecord],
    *,
    arm: str,
    strength: float,
    seed: int,
    split: str,
) -> dict[str, object]:
    """Describe one clean/augmented benchmark arm."""
    clean_count = sum(record.condition == "clean" for record in records)
    distorted_count = sum(record.condition == "augmented" for record in records)
    return {
        "schema_version": 1,
        "arm": arm,
        "augmentation": "lighting",
        "strength": strength,
        "severity": arm.removeprefix("lighting_dark_"),
        "seed": seed,
        "split": split,
        "clean_record_count": clean_count,
        "distorted_record_count": distorted_count,
        "record_count": len(records),
        "condition_mapping": {"clean": "clean", "augmented": "distorted"},
        "detectors": ["Mask-RCNN", "YOLO", "SAM 3"],
        "route_discriminators": 12,
    }
