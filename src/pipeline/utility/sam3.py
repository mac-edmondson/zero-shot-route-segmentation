"""Shared SAM3 lifecycle and mask-normalization helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import torch

class SAM3Base:
    """Common local-model state shared by SAM3 Tracker and Video adapters."""
    def __init__(self, model_dir: str | Path = "/home/vault/v123be/v123be56/LIT/models/sam3", device: str | torch.device | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device("cuda" if device is None and torch.cuda.is_available() else device or "cpu")
        if resolved.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("SAM3 supports CPU, CUDA, and MPS devices.")
        if resolved.type == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA was requested, but it is not available.")
            if resolved.index is not None and resolved.index >= torch.cuda.device_count(): raise RuntimeError(f"CUDA device index {resolved.index} is not available.")
        if resolved.type == "mps" and not torch.backends.mps.is_available(): raise RuntimeError("MPS was requested, but it is not available.")
        return resolved

    def _require_model_files(self, names: tuple[str, ...], *, model_id: str = "SAM3") -> None:
        if not self.model_dir.is_dir() or any(not (self.model_dir / name).is_file() for name in names):
            raise FileNotFoundError(f"Model files for '{model_id}' were not found in '{self.model_dir}'.")

    def _require_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() before inference.")

    def _ensure_model_loaded(self) -> None:
        if self.model is None or self.processor is None:
            self.load_model()

    @staticmethod
    def _binary_masks(masks: Any, expected_count: int | None = None, threshold: bool = False) -> np.ndarray:
        if isinstance(masks, torch.Tensor): masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1: masks = masks[:, 0]
        if expected_count is not None and (masks.ndim != 3 or masks.shape[0] != expected_count):
            raise RuntimeError(f"SAM post-processing returned an unexpected shape: {masks.shape}.")
        if threshold: masks = np.greater(masks, 0)
        if masks.ndim != 3 or masks.shape[1] == 0 or masks.shape[2] == 0:
            raise ValueError("Masks must have shape (N, H, W) with non-empty spatial dimensions.")
        if masks.dtype == np.bool_: return masks
        if not np.issubdtype(masks.dtype, np.number) or not np.all(np.isin(masks, (0, 1))):
            raise ValueError("Masks must be binary boolean or numeric 0/1 arrays.")
        return np.equal(masks, 1)
