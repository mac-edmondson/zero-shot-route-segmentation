"""SAM3 text and exemplar hold detector."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import Sam3VideoModel, Sam3VideoProcessor

from .data_models import Coordinate, Hold, Polygon
from .hold_detector import BatchAlignmentError, InvalidImageError
from .hold_detector_factory import InvalidHoldDetectorConfigError

Exemplar: TypeAlias = tuple[Image.Image, np.ndarray]


@dataclass(frozen=True)
class MaskPrediction:
    """A binary SAM3 mask paired with its model confidence."""

    mask: np.ndarray
    confidence: float


class SAM3HoldWrapper:
    """Load SAM3 and run one text or text-plus-exemplar inference session."""

    def __init__(
        self,
        model_dir: str | Path = "/home/vault/v123be/v123be56/LIT/models/sam3",
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device("cuda" if device is None and torch.cuda.is_available() else device or "cpu")
        if resolved.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("SAM3HoldWrapper supports CPU, CUDA, and MPS devices.")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return resolved

    def load_model(self) -> None:
        """Load the configured local SAM3 video model and processor."""
        if not (self.model_dir / "config.json").is_file() or not (self.model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Local SAM3 model files were not found in '{self.model_dir}'.")
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Sam3VideoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        self.model.eval()

    def _generate_mask_predictions(
        self,
        image: Image.Image,
        text_prompt: str,
        exemplar: Exemplar | None,
    ) -> list[MaskPrediction]:
        """Run one SAM3 session; multi-exemplar orchestration belongs to the detector."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() before inference.")
        if not isinstance(image, Image.Image) or not text_prompt.strip():
            raise ValueError("image must be a PIL image and text_prompt must not be empty.")

        target = image.convert("RGB")
        frames = [target]
        if exemplar is not None:
            exemplar_image, exemplar_mask = self._validate_exemplar(exemplar)
            frames.insert(0, exemplar_image.convert("RGB"))

        session = self.processor.init_video_session(
            video=frames,
            inference_device=self.device,
            processing_device=self.device,
        )
        self.processor.add_text_prompt(session, text_prompt)
        if exemplar is not None:
            object_index = session.obj_id_to_idx(0)
            session.obj_id_to_prompt_id[0] = 0
            session.add_mask_inputs(
                object_index,
                0,
                torch.from_numpy(exemplar_mask).to(self.device).unsqueeze(0).unsqueeze(0),
            )
            session.obj_with_new_inputs = [0]
            session.max_obj_id = 0
            session.obj_id_to_score[0] = 1.0
            session.obj_id_to_tracker_score_frame_wise[0][0] = 1.0
            session.obj_first_frame_idx[0] = 0
            session.trk_keep_alive[0] = self.model.init_trk_keep_alive

        output: Any | None = None
        for frame_index in range(len(frames)):
            output = self.model(inference_session=session, frame_idx=frame_index)
        processed = self.processor.postprocess_outputs(session, output, original_sizes=[list(target.size[::-1])])
        masks = [mask.detach().cpu().numpy().astype(bool) for mask in processed["masks"]]
        scores = self._extract_confidences(processed, output, expected_count=len(masks))
        return [MaskPrediction(mask, confidence) for mask, confidence in zip(masks, scores, strict=True)]

    @staticmethod
    def _extract_confidences(processed: Any, output: Any, expected_count: int) -> list[float]:
        """Return one normalized confidence per mask or fail explicitly."""
        if expected_count == 0:
            return []
        candidate_names = ("scores", "confidences", "mask_scores", "iou_scores", "pred_scores", "object_score_logits")
        for source in (processed, output):
            for name in candidate_names:
                value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
                if value is None:
                    continue
                values = (
                    value.detach().float().cpu().reshape(-1).tolist()
                    if isinstance(value, torch.Tensor)
                    else np.asarray(value, dtype=float).reshape(-1).tolist()
                )
                if len(values) == expected_count and np.all(np.isfinite(values)):
                    return [float(score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + np.exp(-score))) for score in values]
        raise RuntimeError("SAM3 did not expose one finite confidence per predicted mask.")

    @staticmethod
    def _validate_exemplar(exemplar: Exemplar) -> Exemplar:
        if isinstance(exemplar, (str, bytes)) or not isinstance(exemplar, Sequence) or len(exemplar) != 2:
            raise InvalidHoldDetectorConfigError("Each exemplar must be an (image, binary_mask) pair.")
        image, mask = exemplar
        if not isinstance(image, Image.Image):
            raise InvalidHoldDetectorConfigError("Each exemplar image must be a PIL image.")
        return image, SAM3HoldWrapper._validate_mask(mask, image.size)

    @staticmethod
    def _validate_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        array = np.asarray(mask)
        expected_shape = (image_size[1], image_size[0])
        if array.shape != expected_shape:
            raise InvalidHoldDetectorConfigError(f"exemplar mask must have shape {expected_shape}, got {array.shape}.")
        if array.dtype != np.bool_ and not np.all(np.isin(array, (0, 1))):
            raise InvalidHoldDetectorConfigError("exemplar mask must be binary.")
        result = array.astype(bool)
        if not result.any():
            raise InvalidHoldDetectorConfigError("exemplar mask must not be empty.")
        return result


class SAMHoldDetector(SAM3HoldWrapper):
    """Detect holds from text and an optional ordered list of exemplars.

    Use ``SAMHoldDetector(text_prompt="climbing hold")`` for text-only
    detection. Pass ``exemplars=[(image_a, mask_a), (image_b, mask_b)]`` to
    run text-plus-exemplar detection. A one-item list is the single-exemplar
    form. Predictions from all supplied exemplars are merged with
    confidence-ordered mask-IoU NMS; only masks overlapping at or above
    ``nms_iou`` are removed.
    """

    implementation_id = "sam_hold_detector"

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
            raise InvalidHoldDetectorConfigError("exemplars must be a sequence of (image, mask) pairs.")

        super().__init__(**config)
        self.text_prompt = text_prompt
        self.exemplars = tuple(self._validate_exemplar(exemplar) for exemplar in exemplars)
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
        return [self._to_holds([prediction.mask for prediction in batch], mode) for batch in predictions]

    def _ensure_model_loaded(self) -> None:
        if self.model is None:
            self.load_model()

    def _prediction_batches(self, images: Sequence[Image.Image]) -> list[list[MaskPrediction]]:
        if not self.exemplars:
            return [self._generate_mask_predictions(image, self.text_prompt, None) for image in images]

        merged: list[list[MaskPrediction]] = [[] for _ in images]
        for exemplar in self.exemplars:
            for index, image in enumerate(images):
                merged[index].extend(self._generate_mask_predictions(image, self.text_prompt, exemplar))
        return [self._deduplicate_predictions(batch) for batch in merged]

    @staticmethod
    def _validate_images(images: Sequence[Image.Image]) -> None:
        if isinstance(images, (str, bytes)) or not isinstance(images, Sequence) or any(not isinstance(image, Image.Image) for image in images):
            raise InvalidImageError("images must be a sequence of PIL images.")

    def _deduplicate_predictions(self, predictions: Sequence[MaskPrediction]) -> list[MaskPrediction]:
        retained: list[MaskPrediction] = []
        for prediction in sorted(predictions, key=lambda item: item.confidence, reverse=True):
            if all(self._mask_iou(prediction.mask, kept.mask) < self.nms_iou for kept in retained):
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
    def mark_holds(images: Sequence[Image.Image], holds: Sequence[Sequence[Hold]]) -> list[Image.Image]:
        """Render hold polygons over copies of their source images."""
        if len(images) != len(holds):
            raise BatchAlignmentError("images and holds must align.")
        result = []
        for image, detected in zip(images, holds, strict=True):
            overlay = image.convert("RGB").copy()
            draw = ImageDraw.Draw(overlay)
            for hold in detected:
                points = [(point.x, point.y) for point in (*hold.polygon.points, hold.polygon.points[0])]
                draw.line(points, fill=(255, 80, 0), width=3)
            result.append(overlay)
        return result

    @staticmethod
    def _to_holds(masks: Sequence[np.ndarray], mode: str) -> list[Hold]:
        holds = []
        for mask in masks:
            contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            try:
                polygon = Polygon(tuple(Coordinate(float(x), float(y)) for x, y in contour.reshape(-1, 2)))
            except ValueError:
                continue
            holds.append(Hold(
                Coordinate(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                polygon,
                {"mask": np.asarray(mask, dtype=bool).copy(), "mode": mode},
            ))
        return holds
