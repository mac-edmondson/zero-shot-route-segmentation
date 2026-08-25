"""Route discriminator that replays ground-truth route groupings."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from PIL import Image as PILImage

from ..interfaces.data_models import Hold, Image, Route
from ..route_discriminator.route_discriminator import RouteDiscriminator
from .ground_truth_loader import build_routes, load_coco_restructured, pixel_hash

_DEFAULT_ANNOTATIONS_PATH = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "evaluation"
    / "ground_truth_labels"
    / "bouldering_holds_restructured.json"
)


class GroundTruthRouteDiscriminator(RouteDiscriminator):
    @property
    def implementation_id(self) -> str:
        return "ground_truth"

    def __init__(
        self, annotations_path: str | Path = _DEFAULT_ANNOTATIONS_PATH, **config
    ):
        annotations_path = Path(annotations_path)
        holds_by_image = load_coco_restructured(annotations_path)

        with annotations_path.open(encoding="utf-8") as file:
            file_names = [image["file_name"] for image in json.load(file)["images"]]

        self._routes_by_hash: dict[str, Sequence[Route]] = {}
        for file_name, holds in zip(file_names, holds_by_image, strict=True):
            reference_image = PILImage.open(annotations_path.parent / file_name)
            self._routes_by_hash[pixel_hash(reference_image)] = build_routes(holds)

    def get_routes(
        self, images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> Sequence[Sequence[Route]]:
        return [self._routes_by_hash.get(pixel_hash(image), []) for image in images]
