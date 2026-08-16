"""Local shared-weight SAM 3 wrapper for hold-detection experiments."""

from __future__ import annotations

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

    def generate_masks(self, image: Image.Image, text_prompt: str = "climbing hold", exemplar_image: Image.Image | None = None, exemplar_mask: np.ndarray | None = None) -> list[np.ndarray]:
        """Generate masks from text, optionally conditioned on an exemplar."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() before generate_masks().")
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
        return [mask.detach().cpu().numpy().astype(bool) for mask in processed["masks"]]

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
    """Detect all holds with text-only or text-plus-exemplar SAM prompts."""
    implementation_id = "sam_hold_detector"
    def __init__(self, mode: str = "text", text_prompt: str = "climbing hold", exemplar_image: Image | None = None, exemplar_mask: np.ndarray | None = None, **config: object) -> None:
        if mode not in {"text", "text_exemplar"}:
            raise InvalidHoldDetectorConfigError("mode must be text or text_exemplar.")

        if not text_prompt.strip():
            raise InvalidHoldDetectorConfigError("text_prompt must not be empty.")

        if mode == "text_exemplar" and (exemplar_image is None or exemplar_mask is None):
            raise InvalidHoldDetectorConfigError("exemplar mode requires exemplar_image and exemplar_mask.")
        super().__init__(**config); self.mode, self.text_prompt, self.exemplar_image, self.exemplar_mask = mode, text_prompt, exemplar_image, exemplar_mask

    @property
    def configuration(self) -> dict[str, object]:
        return {"mode": self.mode, "model_dir": str(self.model_dir), "device": str(self.device), "text_prompt": self.text_prompt, "exemplar_configured": self.exemplar_image is not None}

    def get_holds(self, images: Sequence[Image]) -> list[list[Hold]]:
        """Detect holds using the configured prompting mode."""
        if not isinstance(images, Sequence) or any(not isinstance(image, Image.Image) for image in images): raise InvalidImageError("images must be PIL image batch.")
        if self.model is None: self.load_model()
        exemplar = (self.exemplar_image, self.exemplar_mask) if self.mode == "text_exemplar" else (None, None)
        return [self._to_holds(self.generate_masks(image, self.text_prompt, *exemplar), self.mode) for image in images]

    def get_holds_text(self, images: Sequence[Image]) -> list[list[Hold]]:
        """Detect holds using only the configured text prompt."""
        if self.model is None: self.load_model()
        return [self._to_holds(self.generate_masks(image, self.text_prompt), "text") for image in images]

    def get_holds_text_exemplar(self, images: Sequence[Image], exemplar_image: Image, exemplar_mask: np.ndarray) -> list[list[Hold]]:
        """Detect holds using text plus a labelled exemplar image."""
        if self.model is None: self.load_model()
        return [self._to_holds(self.generate_masks(image, self.text_prompt, exemplar_image, exemplar_mask), "text_exemplar") for image in images]

    @staticmethod
    def mark_holds(images: Sequence[Image], holds: Sequence[Sequence[Hold]]) -> list[Image]:
        """Draw detected hold polygons over their source images."""
        if len(images) != len(holds): raise BatchAlignmentError("images and holds must align.")
        result = []
        for image, detected in zip(images, holds):
            overlay = image.convert("RGB").copy(); draw = ImageDraw.Draw(overlay)
            for hold in detected: draw.line([(p.x, p.y) for p in (*hold.polygon.points, hold.polygon.points[0])], fill=(255, 80, 0), width=3)
            result.append(overlay)
        return result

    @staticmethod
    def _to_holds(masks: Sequence[np.ndarray], mode: str) -> list[Hold]:
        holds = []
        for mask in masks:
            contours, _ = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            contour = max(contours, key=cv2.contourArea); moments = cv2.moments(contour)
            if moments["m00"] == 0: continue
            try: polygon = Polygon(tuple(Coordinate(float(x), float(y)) for x, y in contour.reshape(-1, 2)))
            except ValueError: continue
            holds.append(Hold(Coordinate(moments["m10"]/moments["m00"], moments["m01"]/moments["m00"]), polygon, {"mask": np.asarray(mask, dtype=bool).copy(), "mode": mode}))
        return holds
