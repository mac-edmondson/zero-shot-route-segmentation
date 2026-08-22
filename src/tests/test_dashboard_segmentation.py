from __future__ import annotations

import numpy as np
from PIL import Image

from backend.schemas import Coordinate
from backend.services import dashboard


class _FakeSAMWrapper:
    model = object()
    processor = object()

    def __init__(self) -> None:
        self.clicks: list[dict[str, float]] | None = None

    def generate_mask(
        self, image: Image.Image, clicks: list[dict[str, float]]
    ) -> np.ndarray:
        self.clicks = clicks
        return np.zeros((1, image.height, image.width), dtype=bool)

    @staticmethod
    def to_polygons(_masks: np.ndarray) -> list[list[list[int]]]:
        return [[[0, 0], [9, 0], [0, 19]]]


def test_detect_segments_converts_sam_click_mask_to_normalized_segment(
    monkeypatch,
) -> None:
    sam = _FakeSAMWrapper()
    monkeypatch.setattr(dashboard, "_sam_wrapper", sam)

    result = dashboard.detect_segments(
        Image.new("RGB", (10, 20)), [Coordinate(x=0.5, y=0.25)]
    )

    assert sam.clicks == [{"x": 0.5, "y": 0.25}]
    assert len(result) == 1
    assert result[0].segment_id.startswith("seg_")
    assert result[0].polygon.points == [
        Coordinate(x=0.0, y=0.0),
        Coordinate(x=1.0, y=0.0),
        Coordinate(x=0.0, y=1.0),
    ]
