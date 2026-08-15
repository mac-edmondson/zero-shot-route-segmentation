"""Local shared-weight SAM 3 wrapper for hold-detection experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3VideoModel, Sam3VideoProcessor


class SAM3HoldWrapper:
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
        if not (self.model_dir / "config.json").is_file() or not (self.model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Local SAM 3 model files were not found in '{self.model_dir}'.")
        self.processor = Sam3VideoProcessor.from_pretrained(self.model_dir, local_files_only=True)
        self.model = Sam3VideoModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        self.model.eval()

    def generate_masks(self, image: Image.Image, text_prompt: str = "climbing hold", exemplar_image: Image.Image | None = None, exemplar_mask: np.ndarray | None = None) -> list[np.ndarray]:
        """Return text-prompted masks, optionally conditioned by a masked exemplar frame."""
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
