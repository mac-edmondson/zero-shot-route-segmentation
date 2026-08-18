import numpy as np
from PIL import Image
from pipeline.interfaces.augmentation import (
    ChalkAugmentation, ChalkAugmentationParams, ColorAugmentation, ColorAugmentationParams,
    LightingAugmentation, LightingAugmentationParams,
)
from pipeline.interfaces.data_models import Coordinate, Polygon, RGBColor
from pipeline.preprocessing.augmentation_suite import AugmentationSuite

def polygon(): return Polygon((Coordinate(2, 2), Coordinate(7, 2), Coordinate(2, 7)))
def image(): return Image.new("RGB", (10, 10), (40, 80, 120))

def test_polygon_augmentations_are_seeded_local_and_non_mutating():
    suite, source = AugmentationSuite(), image(); before = np.asarray(source).copy()
    first = suite.add_chalk(source, [polygon()], ChalkAugmentationParams(0.8), seed=3)
    second = suite.add_chalk(source, [polygon()], ChalkAugmentationParams(0.8), seed=3)
    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert np.array_equal(np.asarray(source), before)
    assert np.array_equal(np.asarray(first)[0, 0], before[0, 0])
    assert not np.array_equal(np.asarray(first)[3, 3], before[3, 3])
    color = suite.change_color(source, [polygon()], ColorAugmentationParams(RGBColor(255, 0, 0), 0.5))
    assert np.array_equal(np.asarray(color)[0, 0], before[0, 0])
    assert not np.array_equal(np.asarray(color)[3, 3], before[3, 3])

def test_lighting_and_plan_order():
    suite, source = AugmentationSuite(), image()
    assert np.asarray(suite.change_lighting(source, LightingAugmentationParams(0.5))).mean() > np.asarray(source).mean()
    assert np.asarray(suite.change_lighting(source, LightingAugmentationParams(-0.5))).mean() < np.asarray(source).mean()
    plan = suite.plan(LightingAugmentation(LightingAugmentationParams(-0.2)), ColorAugmentation((polygon(),), ColorAugmentationParams(RGBColor(255, 0, 0), 0.5)), ChalkAugmentation((polygon(),), ChalkAugmentationParams(0.5)), seed=2)
    expected = suite.change_lighting(suite.change_color(suite.add_chalk(source, [polygon()], ChalkAugmentationParams(0.5), seed=2), [polygon()], ColorAugmentationParams(RGBColor(255, 0, 0), 0.5)), LightingAugmentationParams(-0.2))
    assert np.array_equal(np.asarray(suite.apply_plan(source, plan)), np.asarray(expected))
