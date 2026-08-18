"""Public contract for automatic climbing-hold detectors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from ..interfaces.data_models import Hold, Image


class HoldDetectionError(RuntimeError):
    """Base class for automatic hold-detection failures."""


class InvalidImageError(HoldDetectionError):
    """Raised when a batch has an invalid image."""


class BatchAlignmentError(HoldDetectionError):
    """Raised when image and detection batches differ in length."""


class InvalidHoldDetectorConfigError(ValueError):
    """Raised when there is an invalid configuration within a hold detector"""


class HoldDetector(Protocol):
    @property
    def implementation_id(self) -> str: ...

    def get_holds(self, images: Sequence[Image]) -> Sequence[Sequence[Hold]]:
        """Return one detected-hold list for every input image."""
        ...

    @staticmethod
    def mark_holds(
        images: Sequence[Image.Image], holds: Sequence[Sequence[Hold]]
    ) -> Sequence[Image.Image]:
        """Render hold polygons over copies of their source images."""
        if len(images) != len(holds):
            raise BatchAlignmentError("images and holds must align.")
        result = []
        for image, detected in zip(images, holds, strict=True):
            overlay = image.convert("RGB").copy()
            draw = ImageDraw.Draw(overlay)
            for hold in detected:
                points = [
                    (point.x, point.y)
                    for point in (*hold.polygon.points, hold.polygon.points[0])
                ]
                draw.line(points, fill=(255, 80, 0), width=3)
            result.append(overlay)
        return result
