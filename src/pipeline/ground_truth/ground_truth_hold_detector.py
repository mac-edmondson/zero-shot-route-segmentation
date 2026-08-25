"""Hold detector that replays ground-truth annotations instead of predicting.

Matches an incoming image to one of the annotated evaluation images by an
exact pixel-content hash, then returns that image's ground-truth holds.
Only exact, unmodified copies of the reference images match -- anything
that touches pixels (resize, recompression, crop) won't hash-match.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image as PILImage

from ..interfaces.data_models import Hold, Image
from .ground_truth_loader import load_coco_restructured, load_via_csv, pixel_hash
from ..hold_detector.hold_detector import HoldDetector

_DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "ground_truth_labels"
    / "bouldering_holds_restructured.json"
)


class GroundTruthHoldDetector(HoldDetector):
    @property
    def implementation_id(self) -> str:
        return "ground_truth"

    def __init__(
        self, annotations_path: str | Path = _DEFAULT_ANNOTATIONS_PATH, **config
    ):
        annotations_path = Path(annotations_path)

        if annotations_path.suffix == ".json":
            holds_by_image = load_coco_restructured(annotations_path)
            with annotations_path.open(encoding="utf-8") as file:
                file_names = [image["file_name"] for image in json.load(file)["images"]]
        elif annotations_path.suffix == ".csv":
            # TODO: This is need to be revisited once more. 
            holds_by_image = load_via_csv(annotations_path)
            with annotations_path.open(newline="", encoding="utf-8") as file:
                file_names = list(
                    dict.fromkeys(row["filename"] for row in csv.DictReader(file))
                )
        else:
            raise ValueError(f"Unsupported annotations format: {annotations_path.suffix}")

        self._holds_by_hash: dict[str, Sequence[Hold]] = {}
        for file_name, holds in zip(file_names, holds_by_image, strict=True):
            reference_path = annotations_path.parent / file_name
            if not reference_path.exists():
                continue
            reference_image = PILImage.open(reference_path)
            self._holds_by_hash[pixel_hash(reference_image)] = holds

    def get_holds(self, images: Sequence[Image]) -> Sequence[Sequence[Hold]]:
        return [self._holds_by_hash.get(pixel_hash(image), []) for image in images]
