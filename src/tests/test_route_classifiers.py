import pytest
import cv2
import numpy as np
from PIL import Image, ImageDraw

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.interfaces.errors import BatchAlignmentError
from pipeline.route_discriminator.color_only_route_discriminator import (
    ColorOnlyRouteDiscriminator,
)


def holds() -> list[Hold]:
    return [
        Hold(Polygon((Coordinate(1, 1), Coordinate(3, 1), Coordinate(1, 3)))),
        Hold(Polygon((Coordinate(6, 6), Coordinate(8, 6), Coordinate(6, 8)))),
    ]


def test_color_discriminator_groups_holds_and_marks_routes():
    image = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(image)
    draw.polygon([(1, 1), (3, 1), (1, 3)], fill=(255, 0, 0))
    draw.polygon([(6, 6), (8, 6), (6, 8)], fill=(0, 0, 255))
    discriminator = ColorOnlyRouteDiscriminator(n_clusters=2)
    routes = discriminator.get_routes([image], [holds()])
    assert len(routes) == 1
    assert sorted(len(route.holds) for route in routes[0]) == [1, 1]
    assert len(discriminator.mark_routes([image], routes)) == 1


def test_color_discriminator_validates_batch_alignment():
    discriminator = ColorOnlyRouteDiscriminator(n_clusters=1)
    with pytest.raises(BatchAlignmentError):
        discriminator.get_routes([Image.new("RGB", (1, 1))], [])
    with pytest.raises(BatchAlignmentError):
        discriminator.mark_routes([Image.new("RGB", (1, 1))], [])


def test_color_discriminator_supports_lab_color_space():
    image = Image.new("RGB", (10, 10), (255, 0, 0))
    discriminator = ColorOnlyRouteDiscriminator(n_clusters=1, color_space="lab")

    features = discriminator._color(image, holds()[0])

    expected = cv2.cvtColor(np.uint8([[[255, 0, 0]]]), cv2.COLOR_RGB2LAB)[0, 0]
    assert np.array_equal(features, expected)
    assert discriminator.configuration["color_space"] == "lab"


def test_color_discriminator_rejects_unknown_color_space():
    with pytest.raises(ValueError, match="color_space"):
        ColorOnlyRouteDiscriminator(n_clusters=1, color_space="hsv")
