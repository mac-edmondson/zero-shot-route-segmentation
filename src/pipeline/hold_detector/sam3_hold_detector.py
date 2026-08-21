"""SAM3 text and exemplar hold detector."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from ..interfaces.data_models import Coordinate, Hold, Image, Polygon
from ..interfaces.errors import InvalidHoldDetectorConfigError, InvalidImageError
from ..utility.sam3_ui import Exemplar, MaskPrediction, SAM3HoldWrapper
from .hold_detector import HoldDetector


class SAMHoldDetector(HoldDetector, SAM3HoldWrapper):
    """Detect holds from text and an optional ordered list of exemplars.

    Use ``SAMHoldDetector(text_prompt="climbing hold")`` for text-only
    detection. Pass ``exemplars=[(image_a, mask_a), (image_b, mask_b)]`` to
    run text-plus-exemplar detection. A one-item list is the single-exemplar
    form. Predictions from all supplied exemplars are merged with
    confidence-ordered mask-IoU NMS; only masks overlapping at or above
    ``nms_iou`` are removed.
    """

    @property
    def implementation_id(self) -> str:
        return "sam_hold_detector"

    def __init__(
        self,
        text_prompt: str = "climbing hold",
        exemplars: Sequence[Exemplar] = (),
        nms_iou: float = 0.85,
        **config: object,
    ) -> None:
        if not text_prompt.strip():
            raise InvalidHoldDetectorConfigError("text_prompt must not be empty.")
        if not 0.0 < nms_iou <= 1.0:
            raise InvalidHoldDetectorConfigError("nms_iou must be in (0, 1].")
        if isinstance(exemplars, (str, bytes)) or not isinstance(exemplars, Sequence):
            raise InvalidHoldDetectorConfigError(
                "exemplars must be a sequence of (image, mask) pairs."
            )

        super().__init__(**config)
        self.text_prompt = text_prompt
        self.exemplars = tuple(
            self._validate_exemplar(exemplar) for exemplar in exemplars
        )
        self.nms_iou = nms_iou

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "mode": "text_exemplar" if self.exemplars else "text",
            "model_dir": str(self.model_dir),
            "device": str(self.device),
            "text_prompt": self.text_prompt,
            "exemplar_count": len(self.exemplars),
            "nms_iou": self.nms_iou,
        }

    def get_holds(self, images: Sequence[Image.Image]) -> list[list[Hold]]:
        """Return deduplicated holds for each image using configured exemplars."""
        self._validate_images(images)
        self._ensure_model_loaded()
        predictions = self._prediction_batches(images)
        mode = "text_exemplar" if self.exemplars else "text"
        return [
            self._to_holds([prediction.mask for prediction in batch], mode)
            for batch in predictions
        ]

    def _prediction_batches(
        self, images: Sequence[Image.Image]
    ) -> list[list[MaskPrediction]]:
        if not self.exemplars:
            return [
                self._generate_mask_predictions(image, self.text_prompt, None)
                for image in images
            ]

        merged: list[list[MaskPrediction]] = [[] for _ in images]
        for exemplar in self.exemplars:
            for index, image in enumerate(images):
                merged[index].extend(
                    self._generate_mask_predictions(image, self.text_prompt, exemplar)
                )
        return [self._deduplicate_predictions(batch) for batch in merged]

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if (
            isinstance(images, (str, bytes))
            or not isinstance(images, Sequence)
            or any(not isinstance(image, Image.Image) for image in images)
        ):
            raise InvalidImageError("images must be a sequence of PIL images.")

    def _deduplicate_predictions(
        self, predictions: Sequence[MaskPrediction]
    ) -> list[MaskPrediction]:
        retained: list[MaskPrediction] = []
        for prediction in sorted(
            predictions, key=lambda item: item.confidence, reverse=True
        ):
            if all(
                self._mask_iou(prediction.mask, kept.mask) < self.nms_iou
                for kept in retained
            ):
                retained.append(prediction)
        return retained

    @staticmethod
    def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
        left, right = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
        if left.shape != right.shape:
            raise ValueError("Masks must share a shape for NMS.")
        union = np.count_nonzero(left | right)
        return float(np.count_nonzero(left & right) / union) if union else 0.0

    @staticmethod
    def _to_holds(masks: Sequence[np.ndarray], mode: str) -> list[Hold]:
        holds = []
        for mask in masks:
            contours, _ = cv2.findContours(
                np.asarray(mask, dtype=np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            try:
                polygon = Polygon(
                    tuple(Coordinate(int(x), int(y)) for x, y in contour.reshape(-1, 2))
                )
            except ValueError:
                continue
            holds.append(
                Hold(
                    polygon,
                    {"mask": np.asarray(mask, dtype=bool).copy(), "mode": mode},
                )
            )
        return holds
