""" SAM 3 Wrapper Class"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Sam3TrackerModel, Sam3TrackerProcessor

ClickCoordinates = Sequence[Sequence[int | float]]


class SAMWrapper:
    def __init__(self, model_id: str = "facebook/sam3", model_dir: str | Path = "/home/vault/v123be/v123be56/LIT/models/sam3", device: str | torch.device | None = None) -> None:
        self.model_id = model_id
        self.model_dir = Path(model_dir)
        self.device = self._resolve_device(device)
        self.model: Any | None = None
        self.processor: Any | None = None

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device("cuda" if device is None and torch.cuda.is_available() else device or "cpu")
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("SAMWrapper supports only CPU and CUDA devices.")
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available.")
        if resolved.type == "cuda" and resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {resolved.index} is not available.")
        return resolved

    def load_model(self) -> None:
        if not self.model_dir.is_dir() or not (self.model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Model files for '{self.model_id}' were not found in '{self.model_dir}'.")
        try:
            self.processor = Sam3TrackerProcessor.from_pretrained(self.model_dir, local_files_only=True)
            self.model = Sam3TrackerModel.from_pretrained(self.model_dir, local_files_only=True).to(self.device)
        except OSError as error:
            raise FileNotFoundError(f"Unable to load local model files from '{self.model_dir}'.") from error
        self.model.eval()

    def generate_mask(self, input_img: Image.Image, click_coordinates: ClickCoordinates) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model is not loaded. Call load_model() before generate_mask().")
        if not isinstance(input_img, Image.Image):
            raise TypeError("input_img must be a PIL.Image.Image.")
        image = input_img.convert("RGB")
        clicks = self._validate_clicks(click_coordinates, image.size)
        inputs = self.processor(images=image, input_points=[[[[x, y]] for x, y in clicks]], input_labels=[[[1] for _ in clicks]], return_tensors="pt")
        model_inputs = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = self.model(**model_inputs, multimask_output=False)
        processed = self.processor.post_process_masks(outputs.pred_masks.detach().cpu(), inputs["original_sizes"].cpu(), binarize=True)
        return self._binary_masks(processed[0], expected_count=len(clicks), threshold=True)

    @staticmethod
    def _validate_clicks(clicks: ClickCoordinates, image_size: tuple[int, int]) -> list[tuple[float, float]]:
        if isinstance(clicks, (str, bytes)) or not isinstance(clicks, Sequence):
            raise TypeError("click_coordinates must be a sequence of [x, y] pairs.")
        if not clicks:
            raise ValueError("click_coordinates must contain at least one positive click.")
        width, height = image_size
        result: list[tuple[float, float]] = []
        for index, point in enumerate(clicks):
            if isinstance(point, (str, bytes)) or not isinstance(point, Sequence) or len(point) != 2:
                raise ValueError(f"Click at index {index} must be a [x, y] pair.")
            x, y = point
            if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, Real) or not isinstance(y, Real):
                raise ValueError(f"Click at index {index} must contain numeric x and y values.")
            x, y = float(x), float(y)
            if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Click at index {index} must be finite and inside image bounds.")
            result.append((x, y))
        return result

    @staticmethod
    def _binary_masks(masks: Any, expected_count: int | None = None, threshold: bool = False) -> np.ndarray:
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks = np.asarray(masks)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        if expected_count is not None and (masks.ndim != 3 or masks.shape[0] != expected_count):
            raise RuntimeError(f"SAM post-processing returned an unexpected shape: {masks.shape}.")
        if threshold:
            masks = np.greater(masks, 0)
        if masks.ndim != 3 or masks.shape[1] == 0 or masks.shape[2] == 0:
            raise ValueError("Masks must have shape (N, H, W) with non-empty spatial dimensions.")
        if masks.dtype == np.bool_:
            return masks
        if not np.issubdtype(masks.dtype, np.number) or not np.all(np.isin(masks, (0, 1))):
            raise ValueError("Masks must be binary boolean or numeric 0/1 arrays.")
        return np.equal(masks, 1)

    @staticmethod
    def to_polygons(original_resized_masks: np.ndarray, simplify_tolerance: float = 0.01) -> list[list[list[int]]]:
        if not isinstance(simplify_tolerance, Real) or simplify_tolerance < 0:
            raise ValueError("simplify_tolerance must be a non-negative number.")
        masks = SAMWrapper._binary_masks(original_resized_masks)
        polygons: list[list[list[int]]] = []
        for mask in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                polygons.append([])
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) <= 0:
                polygons.append([])
                continue
            simplified = cv2.approxPolyDP(contour, float(simplify_tolerance) * cv2.arcLength(contour, True), True).reshape(-1, 2)
            polygons.append([[int(x), int(y)] for x, y in simplified])
        return polygons

    @staticmethod
    def to_rle(original_resized_masks: np.ndarray) -> list[dict[str, Any]]:
        masks = SAMWrapper._binary_masks(original_resized_masks)
        try:
            from pycocotools import mask as mask_utils
        except ImportError as error:
            raise ImportError("to_rle() requires the 'pycocotools' package.") from error
        rles = []
        for mask in masks:
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            counts = encoded["counts"]
            rles.append({"size": [int(value) for value in encoded["size"]], "counts": counts.decode("ascii") if isinstance(counts, bytes) else counts})
        return rles
