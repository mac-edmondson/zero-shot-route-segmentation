"""Inference-only YOLOv8 climbing-hold detector."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ..interfaces.data_models import Coordinate, Hold, Polygon
from ..interfaces.errors import InvalidImageError


class YOLOv8HoldDetector:
    implementation_id = "yolov8_hold_detector"
    _WEIGHTS_PATH = (
        Path(__file__).resolve().parents[3]
        / "models"
        / "yolov8_hold_detector"
        / "best.pt" 
    )
    _CLASS_NAMES = {0: "hold"}

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        score_threshold: float | None = 0.4,
        nms_iou_threshold: float | None = 0.3,
        **config: object,
    ) -> None:
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1].")
        if nms_iou_threshold is not None and not 0 <= nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be in [0, 1].")
        self.weights_path = Path(weights_path or self._WEIGHTS_PATH)
        self.device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
        )
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("YOLOv8HoldDetector supports CPU and CUDA devices.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        self.score_threshold, self.nms_iou_threshold, self.extra_config, self.model = (
            score_threshold,
            nms_iou_threshold,
            dict(config),
            None,
        )

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "weights_path": str(self.weights_path),
            "device": str(self.device),
            "score_threshold": self.score_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            **self.extra_config,
        }

    def load_model(self) -> None:
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"YOLOv8 weights file was not found at '{self.weights_path}'."
            )
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Ultralytics is unavailable. Install it to use YOLOv8."
            ) from error
        self.model = YOLO(str(self.weights_path))

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if (
            isinstance(images, (str, bytes))
            or not isinstance(images, Sequence)
            or any(not isinstance(image, Image.Image) for image in images)
        ):
            raise InvalidImageError("images must be a sequence of PIL images.")

    def get_holds(self, images: Sequence[Image.Image]) -> list[list[Hold]]:
        self._validate_images(images)
        if self.model is None:
            self.load_model()
        kwargs: dict[str, Any] = {"device": str(self.device), "verbose": False}
        if self.score_threshold is not None:
            kwargs["conf"] = self.score_threshold
        if self.nms_iou_threshold is not None:
            kwargs["iou"] = self.nms_iou_threshold
        results = self.model.predict(
            [image.convert("RGB") for image in images], **kwargs
        )
        return [
            self._result_to_holds(result, image.size)
            for result, image in zip(results, images, strict=True)
        ]

    @classmethod
    def _result_to_holds(cls, result: Any, image_size: tuple[int, int]) -> list[Hold]:
        holds = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if class_id != 0:
                continue
            hold = cls._box_to_hold(
                box.xyxy[0].tolist(), float(box.conf[0].item()), class_id, image_size
            )
            if hold:
                holds.append(hold)
        return holds

    @classmethod
    def _box_to_hold(
        cls,
        box: Sequence[float],
        confidence: float,
        class_id: int,
        image_size: tuple[int, int],
    ) -> Hold | None:
        width, height = image_size
        left, top, right, bottom = (
            max(0, min(width - 1, round(value)))
            if index % 2 == 0
            else max(0, min(height - 1, round(value)))
            for index, value in enumerate(box)
        )
        if right <= left or bottom <= top:
            return None
        return Hold(
            Polygon(
                (
                    Coordinate(left, top),
                    Coordinate(right, top),
                    Coordinate(right, bottom),
                    Coordinate(left, bottom),
                )
            ),
            {
                "confidence": confidence,
                "class_id": class_id,
                "class_name": cls._CLASS_NAMES[class_id],
                "bbox": (left, top, right, bottom),
                "implementation_id": cls.implementation_id,
            },
        )
