"""Public contracts for reproducible evaluation augmentations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .data_models import Polygon, RGBColor


def _strength(value: float, name: str, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a number.")
    if not low <= float(value) <= high:
        raise ValueError(f"{name} must be in [{low}, {high}].")


@dataclass(frozen=True)
class ChalkAugmentationParams:
    intensity: float = 0.5

    def __post_init__(self) -> None:
        _strength(self.intensity, "intensity", 0, 1)


@dataclass(frozen=True)
class ColorAugmentationParams:
    color: RGBColor
    intensity: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.color, RGBColor):
            raise TypeError("color must be RGBColor.")
        _strength(self.intensity, "intensity", 0, 1)


@dataclass(frozen=True)
class LightingAugmentationParams:
    intensity: float = 0.0

    def __post_init__(self) -> None:
        _strength(self.intensity, "intensity", -1, 1)


class Augmentation(Protocol):
    @property
    def name(self) -> str: ...


def _polygons(values: Sequence[Polygon]) -> tuple[Polygon, ...]:
    polygons = tuple(values)
    if any(not isinstance(polygon, Polygon) for polygon in polygons):
        raise TypeError("polygons must contain Polygon values.")
    return polygons


@dataclass(frozen=True)
class ChalkAugmentation:
    polygons: tuple[Polygon, ...]
    params: ChalkAugmentationParams
    name: str = "chalk"

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygons", _polygons(self.polygons))
        if not isinstance(self.params, ChalkAugmentationParams):
            raise TypeError("params must be ChalkAugmentationParams.")


@dataclass(frozen=True)
class ColorAugmentation:
    polygons: tuple[Polygon, ...]
    params: ColorAugmentationParams
    name: str = "color"

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygons", _polygons(self.polygons))
        if not isinstance(self.params, ColorAugmentationParams):
            raise TypeError("params must be ColorAugmentationParams.")


@dataclass(frozen=True)
class LightingAugmentation:
    params: LightingAugmentationParams
    name: str = "lighting"

    def __post_init__(self) -> None:
        if not isinstance(self.params, LightingAugmentationParams):
            raise TypeError("params must be LightingAugmentationParams.")


AugmentationRecipe = ChalkAugmentation | ColorAugmentation | LightingAugmentation


@dataclass(frozen=True)
class AugmentationPlan:
    augmentations: tuple[AugmentationRecipe, ...]
    seed: int | None = None

    def __post_init__(self) -> None:
        recipes = tuple(self.augmentations)
        if any(
            not isinstance(
                recipe, (ChalkAugmentation, ColorAugmentation, LightingAugmentation)
            )
            for recipe in recipes
        ):
            raise TypeError(
                "augmentations must contain supported augmentation recipes."
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None.")
        object.__setattr__(self, "augmentations", recipes)
