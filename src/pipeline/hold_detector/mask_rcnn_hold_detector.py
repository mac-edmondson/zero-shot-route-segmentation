"""Inference-only Detectron2 Mask R-CNN climbing-hold detector."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from ..interfaces.data_models import Coordinate, Hold, Polygon
from ..interfaces.errors import BatchAlignmentError, InvalidImageError


class MaskRCNNHoldDetector:
    implementation_id = "mask_rcnn_hold_detector"
    _ARTIFACT_DIR = (
        Path(__file__).resolve().parents[3] / "models" / "mask_rcnn_hold_detector"
    )
    _DETECTRON2_SOURCE = Path("/home/vault/v123be/v123be56/LIT/models/detectron2")
    _CLASS_NAMES = {0: "hold", 1: "volume"}

    def __init__(
        self,
        config_path: str | Path | None = None,
        weights_path: str | Path | None = None,
        device: str | torch.device | None = None,
        score_threshold: float | None = None,
        include_volumes: bool = False,
        detectron2_source: str | Path | None = _DETECTRON2_SOURCE,
        **config: object,
    ) -> None:
        if score_threshold is not None and not 0 <= score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1].")
        self.config_path, self.weights_path = (
            Path(config_path or self._ARTIFACT_DIR / "experiment_config.yml"),
            Path(weights_path or self._ARTIFACT_DIR / "model_final.pth"),
        )
        self.device = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
        )
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("MaskRCNNHoldDetector supports CPU and CUDA devices.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        self.score_threshold, self.include_volumes = score_threshold, include_volumes
        self.detectron2_source = Path(detectron2_source) if Path(detectron2_source).exists() else None
        self.extra_config, self.model, self.augmentation = dict(config), None, None

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "config_path": str(self.config_path),
            "weights_path": str(self.weights_path),
            "device": str(self.device),
            "score_threshold": self.score_threshold,
            "include_volumes": self.include_volumes,
            "detectron2_source": str(self.detectron2_source)
            if self.detectron2_source
            else None,
            **self.extra_config,
        }

    def _detectron2_components(self) -> tuple[Any, Any, Any, Any]:
        try:
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.config import get_cfg
            from detectron2.data import transforms as T
            from detectron2.modeling import build_model
        except ModuleNotFoundError as error:
            if (
                self.detectron2_source is None
                or not (self.detectron2_source / "detectron2").is_dir()
            ):
                raise RuntimeError(
                    "Detectron2 is unavailable. Install it or set detectron2_source."
                ) from error
            if str(self.detectron2_source) not in sys.path:
                sys.path.insert(0, str(self.detectron2_source))
            try:
                from detectron2.checkpoint import DetectionCheckpointer
                from detectron2.config import get_cfg
                from detectron2.data import transforms as T
                from detectron2.modeling import build_model
            except (ImportError, OSError) as fallback:
                raise RuntimeError(
                    f"Detectron2 at '{self.detectron2_source}' could not be imported; build it for this environment."
                ) from fallback
        return get_cfg, build_model, DetectionCheckpointer, T

    def load_model(self) -> None:
        for path, name in (
            (self.config_path, "configuration"),
            (self.weights_path, "weights"),
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Mask R-CNN {name} file was not found at '{path}'."
                )
        get_cfg, build_model, checkpointer, transforms = self._detectron2_components()
        cfg = get_cfg()
        cfg.merge_from_file(str(self.config_path))
        cfg.MODEL.WEIGHTS = str(self.weights_path)
        cfg.MODEL.DEVICE = str(self.device)
        if self.score_threshold is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        cfg.freeze()
        self.model = build_model(cfg)
        checkpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.model.eval()
        self.augmentation = transforms.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if (
            isinstance(images, (str, bytes))
            or not isinstance(images, Sequence)
            or any(not isinstance(i, Image.Image) for i in images)
        ):
            raise InvalidImageError("images must be a sequence of PIL images.")

    def get_holds(self, images: Sequence[Image.Image]) -> list[list[Hold]]:
        self._validate_images(images)
        if self.model is None:
            self.load_model()
        inputs = []
        for image in images:
            bgr = np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
            height, width = bgr.shape[:2]
            resized = (
                self.augmentation.get_transform(bgr).apply_image(bgr)
                if self.augmentation
                else bgr
            )
            inputs.append(
                {
                    "image": torch.as_tensor(
                        resized.astype("float32").transpose(2, 0, 1)
                    ).to(self.device),
                    "height": height,
                    "width": width,
                }
            )
        with torch.no_grad():
            outputs = self.model(inputs)
        return [self._instances_to_holds(output["instances"]) for output in outputs]

    def _instances_to_holds(self, instances: Any) -> list[Hold]:
        local, result = instances.to("cpu"), []
        for mask, score, class_id in zip(
            local.pred_masks.detach().numpy().astype(bool),
            local.scores.detach().numpy(),
            local.pred_classes.detach().numpy(),
            strict=True,
        ):
            class_id = int(class_id)
            if class_id in self._CLASS_NAMES and (
                class_id == 0 or self.include_volumes
            ):
                hold = self._mask_to_hold(mask, float(score), class_id)
                if hold:
                    result.append(hold)
        return result

    @classmethod
    def _mask_to_hold(
        cls, mask: np.ndarray, confidence: float, class_id: int
    ) -> Hold | None:
        contours, _ = cv2.findContours(
            np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        try:
            polygon = Polygon(
                tuple(Coordinate(int(x), int(y)) for x, y in contour.reshape(-1, 2))
            )
        except ValueError:
            return None
        return Hold(
            polygon,
            {
                "mask": np.asarray(mask, dtype=bool).copy(),
                "confidence": confidence,
                "class_id": class_id,
                "class_name": cls._CLASS_NAMES[class_id],
                "implementation_id": cls.implementation_id,
            },
        )
