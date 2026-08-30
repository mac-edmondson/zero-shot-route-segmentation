"""Deterministic image augmentations for evaluation datasets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image as PILImage

from ..interfaces.data_models import Hold, Image, Polygon, RGBColor

type AugmentationTarget = Polygon | Hold


def _number(value: float, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number.")
    value = float(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}].")
    return value


def _image(image: Image) -> PILImage.Image:
    if not isinstance(image, PILImage.Image):
        raise TypeError("image must be PIL.Image.Image.")
    return image.convert("RGB")


def _polygons(targets: Sequence[AugmentationTarget]) -> tuple[Polygon, ...]:
    polygons = tuple(
        target.polygon if isinstance(target, Hold) else target for target in targets
    )
    if any(not isinstance(polygon, Polygon) for polygon in polygons):
        raise TypeError("targets must contain Polygon or Hold values.")
    return polygons


def _mask(size: tuple[int, int], polygon: Polygon) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.asarray(
        [(point.x, point.y) for point in polygon.points], dtype=np.int32
    )
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def _from_array(values: np.ndarray) -> Image:
    return PILImage.fromarray(np.clip(values, 0, 255).astype(np.uint8), mode="RGB")


def _aligned(
    targets: Sequence[AugmentationTarget], values: Sequence[object], name: str
) -> tuple[tuple[Polygon, ...], tuple[object, ...]]:
    polygons = _polygons(targets)
    values = tuple(values)
    if len(polygons) != len(values):
        raise ValueError(f"targets and {name} must have the same length.")
    return polygons, values


def add_chalk(
    image: Image,
    targets: Sequence[AugmentationTarget],
    strengths: Sequence[float],
    *,
    seed: int | None = None,
) -> Image:
    """Whiten bright target surfaces while preserving dark hardware."""
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("seed must be an integer or None.")
    polygons, strengths = _aligned(targets, strengths, "strengths")
    strengths = tuple(_number(strength, "strength", 0, 1) for strength in strengths)
    source = _image(image)
    values = np.asarray(source, dtype=np.float32).copy()
    rng = np.random.default_rng(seed)
    for polygon, strength in zip(polygons, strengths, strict=True):
        if strength == 0:
            continue
        mask = _mask(source.size, polygon)
        if not mask.any():
            continue
        # Coarse noise produces hold-scale smudges instead of pixel-level static.
        height, width = mask.shape
        texture = rng.uniform(
            0,
            1,
            size=(max(1, height // 24), max(1, width // 24)),
        ).astype(np.float32)
        # Bicubic upscaling makes density transitions continuous; squaring keeps
        # most chalk faint while retaining a few visibly dense patches.
        alpha = cv2.resize(texture, (width, height), interpolation=cv2.INTER_CUBIC)
        alpha = np.clip(alpha, 0, 1) ** 2
        # Chalk should not fill bolt holes or deep recesses. Fade from no chalk
        # below 10% luminance to full chalk above 40% luminance.
        selected_values = values[mask]
        luminance = np.dot(selected_values, (0.2126, 0.7152, 0.0722)) / 255
        visibility = np.clip((luminance - 0.1) / 0.3, 0, 1)
        alpha = alpha[mask] * visibility
        # Blend only the selected bright surface pixels toward white.
        selected_values += (255 - selected_values) * alpha[:, None] * strength
        values[mask] = selected_values
    return _from_array(values)


def change_color(
    image: Image,
    targets: Sequence[AugmentationTarget],
    colors: Sequence[RGBColor],
) -> Image:
    """Recolour targets while preserving local shading around target lightness."""
    polygons, colors = _aligned(targets, colors, "colors")
    if any(not isinstance(color, RGBColor) for color in colors):
        raise TypeError("colors must contain RGBColor values.")
    source = _image(image)
    hls = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2HLS)
    for polygon, color in zip(polygons, colors, strict=True):
        mask = _mask(source.size, polygon)
        if not mask.any():
            continue
        target = cv2.cvtColor(
            np.asarray([[[color.r, color.g, color.b]]], dtype=np.uint8),
            cv2.COLOR_RGB2HLS,
        )[0, 0]
        selected_hls = hls[mask]
        selected_hls[:, 1] = np.clip(
            selected_hls[:, 1].astype(np.float32)
            - np.median(selected_hls[:, 1])
            + target[1],
            0,
            255,
        )
        selected_hls[:, 0] = target[0]
        selected_hls[:, 2] = target[2]
        hls[mask] = selected_hls
    return PILImage.fromarray(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB), mode="RGB")


def change_lighting(image: Image, strength: float) -> Image:
    """Scale every channel in an ``H × W × 3`` RGB image by one brightness factor.

    ``strength`` is dimensionless in ``[-1, 1]`` and maps to the frontend's
    brightness factor ``1 + 0.9 * strength`` in ``[0.1, 1.9]``.
    """
    strength = _number(strength, "strength", -1, 1)
    values = np.asarray(_image(image), dtype=np.float32).copy()
    values *= 1 + strength
    return _from_array(values)


@dataclass(frozen=True)
class ChalkAugmentation:
    targets: tuple[AugmentationTarget, ...]
    strengths: tuple[float, ...]

    def __post_init__(self) -> None:
        polygons, strengths = _aligned(self.targets, self.strengths, "strengths")
        object.__setattr__(self, "targets", polygons)
        object.__setattr__(
            self,
            "strengths",
            tuple(_number(strength, "strength", 0, 1) for strength in strengths),
        )


@dataclass(frozen=True)
class ColorAugmentation:
    targets: tuple[AugmentationTarget, ...]
    colors: tuple[RGBColor, ...]

    def __post_init__(self) -> None:
        polygons, colors = _aligned(self.targets, self.colors, "colors")
        if any(not isinstance(color, RGBColor) for color in colors):
            raise TypeError("colors must contain RGBColor values.")
        object.__setattr__(self, "targets", polygons)
        object.__setattr__(self, "colors", colors)


@dataclass(frozen=True)
class LightingAugmentation:
    strength: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "strength", _number(self.strength, "strength", -1, 1))


type AugmentationRecipe = ChalkAugmentation | ColorAugmentation | LightingAugmentation


@dataclass(frozen=True)
class AugmentationPlan:
    augmentations: tuple[AugmentationRecipe, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        augmentations = tuple(self.augmentations)
        if any(
            not isinstance(
                augmentation,
                (ChalkAugmentation, ColorAugmentation, LightingAugmentation),
            )
            for augmentation in augmentations
        ):
            raise TypeError(
                "augmentations must contain supported augmentation recipes."
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None.")
        object.__setattr__(self, "augmentations", augmentations)

    def apply(self, image: Image) -> Image:
        """Apply recipes in declaration order without mutating image."""
        result = _image(image)
        for index, augmentation in enumerate(self.augmentations):
            if isinstance(augmentation, ChalkAugmentation):
                seed = (
                    None
                    if self.seed is None
                    else int(
                        np.random.SeedSequence((self.seed, index)).generate_state(1)[0]
                    )
                )
                result = add_chalk(
                    result,
                    augmentation.targets,
                    augmentation.strengths,
                    seed=seed,
                )
            elif isinstance(augmentation, ColorAugmentation):
                result = change_color(result, augmentation.targets, augmentation.colors)
            else:
                result = change_lighting(result, augmentation.strength)
        return result
