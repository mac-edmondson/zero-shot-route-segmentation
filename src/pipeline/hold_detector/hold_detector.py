"""Public contract for automatic climbing-hold detectors."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..interfaces.data_models import Hold, Image


class HoldDetectionError(RuntimeError):
    """Base class for automatic hold-detection failures."""


class InvalidImageError(HoldDetectionError):
    """Raised when a batch has an invalid image."""


class BatchAlignmentError(HoldDetectionError):
    """Raised when image and detection batches differ in length."""


class HoldDetector(Protocol):
    @property
    def implementation_id(self) -> str: ...

    def get_holds(self, images: Sequence[Image]) -> list[list[Hold]]:
        """Return one detected-hold list for every input image."""

    @staticmethod
    def mark_holds(
        images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> list[Image]:
        """Render hold overlays without changing detection behavior."""
