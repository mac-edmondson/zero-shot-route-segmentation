from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..interfaces.data_models import Coordinate, Hold, Image, Polygon
from .hold_detector import HoldDetector


MIN_NUM_HOLDS_PER_IMAGE = 0
MAX_NUM_HOLDS_PER_IMAGE = 100


class MockHoldDetector(HoldDetector):
    @property
    def implementation_id(self) -> str:
        return "mock_hold_detector"

    def __init__(self, **config):
        seed = 12345
        if "seed" in config:
            seed = config["seed"]
        self.rng = np.random.default_rng(seed)

    def get_holds(self, images: Sequence[Image]) -> Sequence[Sequence[Hold]]:
        """Return one detected-hold list for every input image."""
        holds = [[] for _ in images]
        for i, image in enumerate(images):
            if image.width == 0 or image.height == 0:
                continue

            if image.width < 2 or image.height < 2:
                continue

            num_holds = self.rng.integers(
                MIN_NUM_HOLDS_PER_IMAGE,
                MAX_NUM_HOLDS_PER_IMAGE + 1,
            )
            holds[i] = [self._random_hold(image) for _ in range(num_holds)]

        return holds

    def _random_hold(self, image: Image) -> Hold:
        x0, x1 = sorted(self.rng.integers(0, image.width, size=2))
        y0, y1 = sorted(self.rng.integers(0, image.height, size=2))
        if x0 == x1:
            x1 = min(x1 + 1, image.width - 1)
        if y0 == y1:
            y1 = min(y1 + 1, image.height - 1)
        return Hold(
            Polygon(
                (
                    Coordinate(int(x0), int(y0)),
                    Coordinate(int(x1), int(y0)),
                    Coordinate(int(x1), int(y1)),
                    Coordinate(int(x0), int(y1)),
                )
            )
        )
