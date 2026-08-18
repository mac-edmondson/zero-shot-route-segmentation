"""Simple deterministic image augmentations for evaluation datasets."""
from __future__ import annotations
from collections.abc import Sequence
import cv2
import numpy as np
from PIL import Image as PILImage
from ..interfaces.augmentation import (
    AugmentationPlan, AugmentationRecipe, ChalkAugmentation, ChalkAugmentationParams,
    ColorAugmentation, ColorAugmentationParams, LightingAugmentation, LightingAugmentationParams,
)
from ..interfaces.data_models import Image, Polygon, RGBColor

class AugmentationSuite:
    """Apply polygon-local chalk/color changes and global lighting changes."""

    @staticmethod
    def _image(image: Image) -> PILImage.Image:
        if not isinstance(image, PILImage.Image): raise TypeError("image must be PIL.Image.Image.")
        return image.convert("RGB")

    @staticmethod
    def _mask(size: tuple[int, int], polygons: Sequence[Polygon]) -> np.ndarray:
        width, height = size; mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in polygons:
            if not isinstance(polygon, Polygon): raise TypeError("polygons must contain Polygon values.")
            points = np.asarray([(point.x, point.y) for point in polygon.points], dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)

    @staticmethod
    def _from_array(values: np.ndarray) -> Image:
        return PILImage.fromarray(np.clip(values, 0, 255).astype(np.uint8), mode="RGB")

    def add_chalk(self, image: Image, polygons: Sequence[Polygon], params: ChalkAugmentationParams, *, seed: int | None = None) -> Image:
        if not isinstance(params, ChalkAugmentationParams): raise TypeError("params must be ChalkAugmentationParams.")
        source = self._image(image); mask = self._mask(source.size, polygons)
        if not mask.any() or params.intensity == 0: return source.copy()
        values = np.asarray(source, dtype=np.float32).copy(); rng = np.random.default_rng(seed)
        texture = rng.uniform(0.35, 1.0, size=mask.shape).astype(np.float32) * float(params.intensity)
        alpha = texture[mask, None]; values[mask] += (255 - values[mask]) * alpha
        return self._from_array(values)

    def change_color(self, image: Image, polygons: Sequence[Polygon], params: ColorAugmentationParams) -> Image:
        if not isinstance(params, ColorAugmentationParams): raise TypeError("params must be ColorAugmentationParams.")
        source = self._image(image); mask = self._mask(source.size, polygons)
        if not mask.any() or params.intensity == 0: return source.copy()
        values = np.asarray(source, dtype=np.float32).copy(); target = np.asarray((params.color.r, params.color.g, params.color.b), dtype=np.float32)
        values[mask] = values[mask] * (1 - float(params.intensity)) + target * float(params.intensity)
        return self._from_array(values)

    def change_lighting(self, image: Image, params: LightingAugmentationParams) -> Image:
        if not isinstance(params, LightingAugmentationParams): raise TypeError("params must be LightingAugmentationParams.")
        values = np.asarray(self._image(image), dtype=np.float32).copy(); intensity = float(params.intensity)
        if intensity >= 0: values += (255 - values) * intensity
        else: values *= 1 + intensity
        return self._from_array(values)

    def apply(self, image: Image, *, chalk_polygons: Sequence[Polygon] | None = None, chalk_params: ChalkAugmentationParams | None = None, color_polygons: Sequence[Polygon] | None = None, color_params: ColorAugmentationParams | None = None, lighting_params: LightingAugmentationParams | None = None, seed: int | None = None) -> Image:
        result = self._image(image).copy()
        if chalk_params is not None: result = self.add_chalk(result, chalk_polygons or (), chalk_params, seed=seed)
        if color_params is not None: result = self.change_color(result, color_polygons or (), color_params)
        if lighting_params is not None: result = self.change_lighting(result, lighting_params)
        return result

    @staticmethod
    def plan(*augmentations: AugmentationRecipe, seed: int | None = None) -> AugmentationPlan:
        return AugmentationPlan(tuple(augmentations), seed)

    def apply_plan(self, image: Image, plan: AugmentationPlan) -> Image:
        if not isinstance(plan, AugmentationPlan): raise TypeError("plan must be AugmentationPlan.")
        result = self._image(image).copy()
        order = (ChalkAugmentation, ColorAugmentation, LightingAugmentation)
        for kind in order:
            for augmentation in plan.augmentations:
                if isinstance(augmentation, kind):
                    if isinstance(augmentation, ChalkAugmentation): result = self.add_chalk(result, augmentation.polygons, augmentation.params, seed=plan.seed)
                    elif isinstance(augmentation, ColorAugmentation): result = self.change_color(result, augmentation.polygons, augmentation.params)
                    else: result = self.change_lighting(result, augmentation.params)
        return result

    def materialize(self, images: Sequence[Image], plan: AugmentationPlan) -> list[Image]:
        return [self.apply_plan(image, plan) for image in images]
