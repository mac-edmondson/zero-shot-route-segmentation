"""Behavioral tests for image augmentations."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from pipeline.interfaces.data_models import (
    Coordinate,
    Hold,
    ImageRecord,
    Polygon,
    RGBColor,
)
from pipeline.preprocessing.augmentation_suite import (
    AugmentationPlan,
    ChalkAugmentation,
    ColorAugmentation,
    LightingAugmentation,
    add_chalk,
    change_color,
    change_lighting,
)
from pipeline.preprocessing.data_preprocessing import DatasetSpec
from pipeline.preprocessing.data_preprocessing_pipeline import DataPreprocessingPipeline


def _polygon() -> Polygon:
    return Polygon(
        (
            Coordinate(1, 1),
            Coordinate(5, 1),
            Coordinate(5, 5),
            Coordinate(1, 5),
        )
    )


def _image() -> PILImage.Image:
    values = np.full((8, 8, 3), 50, dtype=np.uint8)
    values[1:6, 1:3] = 80
    values[1:6, 3:6] = 160
    return PILImage.fromarray(values, mode="RGB")


def test_add_chalk_validates_alignment_and_accepts_holds() -> None:
    image = _image()
    with pytest.raises(ValueError, match="same length"):
        add_chalk(image, [_polygon()], [])

    result = add_chalk(image, [Hold(_polygon())], [1], seed=1)
    assert result.getpixel((2, 2))[0] > image.getpixel((2, 2))[0]
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))


def test_add_chalk_uses_smooth_patchy_texture() -> None:
    image = PILImage.new("RGB", (96, 96), (100, 100, 100))
    polygon = Polygon(
        (
            Coordinate(1, 1),
            Coordinate(94, 1),
            Coordinate(94, 94),
            Coordinate(1, 94),
        )
    )

    result = np.asarray(add_chalk(image, [polygon], [0.5], seed=0))[:, :, 0]
    chalk = result[2:94, 2:94]
    adjacent_difference = np.abs(np.diff(chalk.astype(int), axis=0)).mean()

    assert chalk.min() < chalk.max()
    assert adjacent_difference < 4


def test_change_color_preserves_target_lightness() -> None:
    image = _image()
    result = change_color(image, [_polygon()], [RGBColor(220, 20, 20)])

    dark, light = result.getpixel((2, 2)), result.getpixel((4, 2))
    assert dark[0] > dark[1] and dark[0] > dark[2]
    assert light[0] > light[1] and light[0] > light[2]
    assert sum(light) > sum(dark)
    assert result.getpixel((0, 0)) == image.getpixel((0, 0))


def test_lighting_uses_the_frontend_brightness_scale() -> None:
    image = PILImage.new("RGB", (1, 1), (10, 20, 30))
    assert change_lighting(image, -1).getpixel((0, 0)) == (0, 0, 0)
    assert change_lighting(image, 0).getpixel((0, 0)) == (10, 20, 30)
    assert change_lighting(image, 1).getpixel((0, 0)) == (20, 40, 60)


def test_plan_is_ordered_and_seeded() -> None:
    image = _image()
    color = ColorAugmentation((_polygon(),), (RGBColor(220, 20, 20),))
    chalk = ChalkAugmentation((_polygon(),), (0.5,))
    lighting = LightingAugmentation(-0.5)
    color_chalk_lighting = AugmentationPlan((color, chalk, lighting), seed=4)
    chalk_color_lighting = AugmentationPlan((chalk, color, lighting), seed=4)

    assert list(color_chalk_lighting.apply(image).get_flattened_data()) == list(
        color_chalk_lighting.apply(image).get_flattened_data()
    )
    assert list(color_chalk_lighting.apply(image).get_flattened_data()) != list(
        chalk_color_lighting.apply(image).get_flattened_data()
    )


def test_plan_applies_color_recipe() -> None:
    image = _image()
    result = AugmentationPlan(
        (ColorAugmentation((_polygon(),), (RGBColor(20, 220, 20),)),)
    ).apply(image)

    pixel = result.getpixel((2, 2))
    assert pixel[1] > pixel[0] and pixel[1] > pixel[2]


def test_preprocessing_records_recipe_metadata() -> None:
    image = _image()
    provider = DataPreprocessingPipeline().build(
        [ImageRecord("wall", image, split="test")],
        DatasetSpec(
            "evaluation",
            augmentation_plan=AugmentationPlan((LightingAugmentation(0.5),)),
        ),
    )

    records = list(provider.iter_split("test"))
    assert records[1].augmentation_metadata["augmentations"] == ("lighting",)
