"""Local shared-weight SAM 3 wrapper for hold-detection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from collections.abc import Sequence
from PIL import Image, ImageDraw

from .data_models import Coordinate, Hold, Polygon
from .hold_detector import BatchAlignmentError, InvalidImageError
from .hold_detector_factory import InvalidHoldDetectorConfigError

import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor


@dataclass(frozen=True)
class MaskPrediction:
    """A binary SAM3 mask paired with its model confidence."""

    mask: np.ndarray
    confidence: float


class SAM3HoldWrapper:
    """Provide low-level SAM3 text and exemplar mask generation."""
    def __init__(self, model_dir: str | Path = "/home/vault/v123be/v123be56/LIT/models/sam3", device: str | torch.device | None = None) -> None:
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
            raise FileNotFoundError(f"Local SAM 3 model files were not found in '{self.model_dir}'.")
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Sam3VideoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        self.model.eval()

    def generate_mask_predictions(self, image: Image.Image, text_prompt: str = "climbing hold", exemplar_image: Image.Image | None = None, exemplar_mask: np.ndarray | None = None) -> list[MaskPrediction]:
        """Generate binary masks and their SAM3 confidences."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() before generate_mask_predictions().")
        if not isinstance(image, Image.Image) or not text_prompt.strip():
            raise ValueError("image must be a PIL image and text_prompt must not be empty.")
        target = image.convert("RGB")
        frames = [target] if exemplar_image is None else [exemplar_image.convert("RGB"), target]
        if (exemplar_image is None) != (exemplar_mask is None):
            raise ValueError("exemplar_image and exemplar_mask must be supplied together.")
        session = self.processor.init_video_session(video=frames, inference_device=self.device, processing_device=self.device)
        self.processor.add_text_prompt(session, text_prompt)
        if exemplar_mask is not None:
            mask = self._validate_mask(exemplar_mask, frames[0].size)
            object_index = session.obj_id_to_idx(0)
            session.obj_id_to_prompt_id[0] = 0
            session.add_mask_inputs(object_index, 0, torch.from_numpy(mask).to(self.device).unsqueeze(0).unsqueeze(0))
            session.obj_with_new_inputs = [0]
            session.max_obj_id = 0
            session.obj_id_to_score[0] = 1.0
            session.obj_id_to_tracker_score_frame_wise[0][0] = 1.0
            session.obj_first_frame_idx[0] = 0
            session.trk_keep_alive[0] = self.model.init_trk_keep_alive
        output = None
        for frame_index in range(len(frames)):
            output = self.model(inference_session=session, frame_idx=frame_index)
        processed = self.processor.postprocess_outputs(session, output, original_sizes=[list(target.size[::-1])])
        masks = [mask.detach().cpu().numpy().astype(bool) for mask in processed["masks"]]
        scores = self._extract_confidences(processed, output, expected_count=len(masks))
        return [MaskPrediction(mask, confidence) for mask, confidence in zip(masks, scores, strict=True)]

    def generate_masks(self, image: Image.Image, text_prompt: str = "climbing hold", exemplar_image: Image.Image | None = None, exemplar_mask: np.ndarray | None = None) -> list[np.ndarray]:
        """Generate masks from text, optionally conditioned on an exemplar."""
        return [prediction.mask for prediction in self.generate_mask_predictions(image, text_prompt, exemplar_image, exemplar_mask)]

    @staticmethod
    def _extract_confidences(processed: Any, output: Any, expected_count: int) -> list[float]:
        """Return one normalized confidence per mask or fail rather than disabling filtering."""
        if expected_count == 0:
            return []
        candidate_names = ("scores", "confidences", "mask_scores", "iou_scores", "pred_scores", "object_score_logits")
        for source in (processed, output):
            for name in candidate_names:
                value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
                if value is None:
                    continue
                values = value.detach().float().cpu().reshape(-1).tolist() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=float).reshape(-1).tolist()
                if len(values) != expected_count or not np.all(np.isfinite(values)):
                    continue
                return [float(score if 0.0 <= score <= 1.0 else 1.0 / (1.0 + np.exp(-score))) for score in values]
        raise RuntimeError("SAM3 did not expose one finite confidence per predicted mask; refusing confidence-filtered evaluation.")

    @staticmethod
    def _validate_mask(mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
        array = np.asarray(mask)
        expected_shape = (image_size[1], image_size[0])
        if array.shape != expected_shape:
            raise ValueError(f"exemplar_mask must have shape {expected_shape}, got {array.shape}.")
        if array.dtype != np.bool_ and not np.all(np.isin(array, (0, 1))):
            raise ValueError("exemplar_mask must be binary.")
        return array.astype(bool)

class SAMHoldDetector(SAM3HoldWrapper):
    """Detect holds with text-only or text-plus-exemplar SAM3 prompting.

    Use ``mode="text"`` with no exemplar inputs for text-only detection.
    Use ``mode="text_exemplar"`` with ``exemplar_image`` and
    ``exemplar_mask`` for the backwards-compatible single-exemplar form, or
    with ``exemplars=[(image_a, mask_a), (image_b, mask_b)]`` for multiple
    exemplars. Multi-exemplar predictions are merged and mask-IoU NMS removes
    only duplicate masks; nearby distinct holds are retained.
    """

    implementation_id = "sam_hold_detector"

    def __init__(
        self,
        mode: str = "text",
        text_prompt: str = "climbing hold",
        exemplar_image: Image.Image | None = None,
        exemplar_mask: np.ndarray | None = None,
        exemplars: Sequence[tuple[Image.Image, np.ndarray]] | None = None,
        nms_iou: float = 0.85,
        **config: object,
    ) -> None:
        if mode not in {"text", "text_exemplar"}:
            raise InvalidHoldDetectorConfigError("mode must be text or text_exemplar.")
        if not text_prompt.strip():
            raise InvalidHoldDetectorConfigError("text_prompt must not be empty.")
        if not 0.0 < nms_iou <= 1.0:
            raise InvalidHoldDetectorConfigError("nms_iou must be in (0, 1].")

        singular_configured = exemplar_image is not None or exemplar_mask is not None
        if singular_configured and (exemplar_image is None or exemplar_mask is None):
            raise InvalidHoldDetectorConfigError("exemplar_image and exemplar_mask must be supplied together.")
        if singular_configured and exemplars is not None:
            raise InvalidHoldDetectorConfigError("Use either exemplar_image/exemplar_mask or exemplars, not both.")
        normalized = [(exemplar_image, exemplar_mask)] if singular_configured else list(exemplars or ())
        if mode == "text" and normalized:
            raise InvalidHoldDetectorConfigError("text mode does not accept exemplars.")
        if mode == "text_exemplar" and not normalized:
            raise InvalidHoldDetectorConfigError("exemplar mode requires one or more exemplar image/mask pairs.")
        for exemplar, mask in normalized:
            if not isinstance(exemplar, Image.Image):
                raise InvalidHoldDetectorConfigError("Each exemplar image must be a PIL image.")
            self._validate_mask(mask, exemplar.size)

        super().__init__(**config)
        self.mode = mode
        self.text_prompt = text_prompt
        self.exemplars = tuple(normalized)
        self.nms_iou = nms_iou
        self.exemplar_image, self.exemplar_mask = self.exemplars[0] if len(self.exemplars) == 1 else (None, None)

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "model_dir": str(self.model_dir),
            "device": str(self.device),
            "text_prompt": self.text_prompt,
            "exemplar_count": len(self.exemplars),
            "nms_iou": self.nms_iou,
        }

    def get_holds(self, images: Sequence[Image.Image]) -> list[list[Hold]]:
        """Return holds for each image using this detector's configured mode."""
        if not isinstance(images, Sequence) or any(not isinstance(image, Image.Image) for image in images):
            raise InvalidImageError("images must be a sequence of PIL images.")
        if self.model is None:
            self.load_model()
        if self.mode == "text":
            return [self._to_holds(self.generate_masks(image, self.text_prompt), "text") for image in images]
        return self.get_holds_text_exemplars(images, self.exemplars)

    def get_mask_predictions_text_exemplar(
        self,
        images: Sequence[Image.Image],
        exemplar_image: Image.Image,
        exemplar_mask: np.ndarray,
    ) -> list[list[MaskPrediction]]:
        """Return raw scored masks for one exemplar across a target batch."""
        return self.get_mask_predictions_text_exemplars(
            images, ((exemplar_image, exemplar_mask),), deduplicate=False
        )

    def get_mask_predictions_text_exemplars(
        self,
        images: Sequence[Image.Image],
        exemplars: Sequence[tuple[Image.Image, np.ndarray]],
        *,
        deduplicate: bool = True,
    ) -> list[list[MaskPrediction]]:
        """Merge predictions from exemplar pairs and optionally deduplicate by mask IoU."""
        if not exemplars:
            raise ValueError("At least one exemplar image/mask pair is required.")
        if self.model is None:
            self.load_model()
        merged: list[list[MaskPrediction]] = [[] for _ in images]
        for exemplar_image, exemplar_mask in exemplars:
            if not isinstance(exemplar_image, Image.Image):
                raise TypeError("Each exemplar image must be a PIL image.")
            self._validate_mask(exemplar_mask, exemplar_image.size)
            for index, image in enumerate(images):
                merged[index].extend(
                    self.generate_mask_predictions(image, self.text_prompt, exemplar_image, exemplar_mask)
                )
        return [self._deduplicate_predictions(predictions) if deduplicate else predictions for predictions in merged]

    def get_holds_text(self, images: Sequence[Image.Image]) -> list[list[Hold]]:
        """Detect holds using text only, independent of configured exemplars."""
        if self.model is None:
            self.load_model()
        return [self._to_holds(self.generate_masks(image, self.text_prompt), "text") for image in images]

    def get_holds_text_exemplar(
        self,
        images: Sequence[Image.Image],
        exemplar_image: Image.Image,
        exemplar_mask: np.ndarray,
    ) -> list[list[Hold]]:
        """Detect holds from text plus one labelled exemplar image and mask."""
        predictions = self.get_mask_predictions_text_exemplar(images, exemplar_image, exemplar_mask)
        return [self._to_holds([prediction.mask for prediction in batch], "text_exemplar") for batch in predictions]

    def get_holds_text_exemplars(
        self,
        images: Sequence[Image.Image],
        exemplars: Sequence[tuple[Image.Image, np.ndarray]],
    ) -> list[list[Hold]]:
        """Detect holds from text plus ordered exemplar image/mask pairs with NMS."""
        predictions = self.get_mask_predictions_text_exemplars(images, exemplars)
        return [self._to_holds([prediction.mask for prediction in batch], "text_exemplar") for batch in predictions]

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
        """Render hold overlays without changing detection behavior."""
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
