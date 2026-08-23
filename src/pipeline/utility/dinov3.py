"""Local DINOv3 feature-extractor plumbing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModel


class DINOv3:
    """Load a frozen local DINOv3 ViT-S/16 and extract patch tokens."""

    MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
    IMAGE_SIZE = 1280
    PATCH_SIZE = 16
    HIDDEN_SIZE = 384
    NUM_REGISTER_TOKENS = 4
    NUM_PATCH_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2
    _MODEL_FILES = ("config.json", "model.safetensors", "preprocessor_config.json")

    def __init__(
        self,
        model_dir: str | Path = "models/dinov3",
        device: str | torch.device | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device(
            "cuda" if device is None and torch.cuda.is_available() else device or "cpu"
        )
        if resolved.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("DINOv3 supports CPU, CUDA, and MPS devices.")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but it is not available.")
            if (
                resolved.index is not None
                and resolved.index >= torch.cuda.device_count()
            ):
                raise RuntimeError(
                    f"CUDA device index {resolved.index} is not available."
                )
        if resolved.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but it is not available.")
        return resolved

    def load_model(self) -> None:
        """Load the configured checkpoint from local files in FP32."""
        if not self.model_dir.is_dir() or any(
            not (self.model_dir / name).is_file() for name in self._MODEL_FILES
        ):
            raise FileNotFoundError(
                f"DINOv3 model files were not found in '{self.model_dir}'. Download "
                f"'{self.MODEL_ID}' with `hf download {self.MODEL_ID} --local-dir "
                f"{self.model_dir}`."
            )

        processor = AutoImageProcessor.from_pretrained(
            self.model_dir, local_files_only=True
        )
        model = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        self._validate_config(model.config)
        model.to(device=self.device, dtype=torch.float32)
        model.requires_grad_(False)
        model.eval()
        self.processor, self.model = processor, model

    @classmethod
    def _validate_config(cls, config: Any) -> None:
        expected = {
            "patch_size": cls.PATCH_SIZE,
            "hidden_size": cls.HIDDEN_SIZE,
            "num_register_tokens": cls.NUM_REGISTER_TOKENS,
        }
        mismatches = [
            f"{name}={getattr(config, name, None)!r} (expected {value})"
            for name, value in expected.items()
            if getattr(config, name, None) != value
        ]
        if mismatches:
            raise RuntimeError(
                "The local checkpoint is not the expected DINOv3 ViT-S/16 model: "
                + ", ".join(mismatches)
            )

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.processor is None:
            self.load_model()

    def extract_patch_tokens(self, image: PILImage.Image) -> torch.Tensor:
        """Return the image's 6,400 local patch embeddings on the model device."""
        if not isinstance(image, PILImage.Image):
            raise TypeError("image must be a PIL.Image.Image.")
        self._ensure_model_loaded()

        inputs = self.processor(
            images=image.convert("RGB"),
            size={"height": self.IMAGE_SIZE, "width": self.IMAGE_SIZE},
            do_center_crop=False,
            return_tensors="pt",
        )
        inputs = {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            output = self.model(**inputs)

        hidden_state = output.last_hidden_state
        patch_tokens = hidden_state[:, 1 + self.NUM_REGISTER_TOKENS :, :]
        expected_shape = (1, self.NUM_PATCH_TOKENS, self.HIDDEN_SIZE)
        if patch_tokens.shape != expected_shape:
            raise RuntimeError(
                "DINOv3 returned unexpected patch-token shape "
                f"{tuple(patch_tokens.shape)}; expected {expected_shape}."
            )
        return patch_tokens[0].detach()
