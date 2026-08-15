"""Dataset loading and VIA polygon rasterization for climbing holds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def load_image(image_path: str | Path) -> Image.Image:
    """Load a prepared dataset image as an RGB PIL image."""
    with Image.open(image_path) as image:
        return image.convert("RGB")


def load_hold_masks(annotation_path: str | Path, filename: str, image_size: tuple[int, int]) -> list[np.ndarray]:
    """Rasterize VIA polygons labelled as climbing holds for one image."""
    width, height = image_size
    masks: list[np.ndarray] = []
    with Path(annotation_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["filename"] != filename:
                continue
            attributes = json.loads(row["region_attributes"])
            shape = json.loads(row["region_shape_attributes"])
            if attributes.get("hold_type") != "hold" or shape.get("name") != "polygon":
                continue
            points = np.asarray(list(zip(shape["all_points_x"], shape["all_points_y"])), dtype=np.int32)
            if len(points) < 3:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 1)
            masks.append(mask.astype(bool))
    return masks


def largest_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Return the largest non-empty instance mask."""
    if not masks:
        raise ValueError("At least one annotated hold mask is required for the exemplar.")
    return max(masks, key=np.count_nonzero)
