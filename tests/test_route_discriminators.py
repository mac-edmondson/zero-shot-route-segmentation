import pytest
from PIL import Image, ImageDraw

from pipeline.interfaces.data_models import Coordinate, Hold, Polygon
from pipeline.route_discriminator.color_only_route_discriminator import ColorOnlyRouteDiscriminator
from pipeline.interfaces.route_discriminator import BatchAlignmentError
from pipeline.interfaces.route_discriminator_factory import (
    InvalidRouteDiscriminatorConfigError,
    RouteDiscriminatorFactory,
    UnknownRouteDiscriminatorError,
)


def holds() -> list[Hold]:
    return [
        Hold(Polygon((Coordinate(1, 1), Coordinate(3, 1), Coordinate(1, 3)))),
        Hold(Polygon((Coordinate(6, 6), Coordinate(8, 6), Coordinate(6, 8)))),
    ]


def test_color_classifier_groups_holds_and_marks_routes():
    image = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(image)
    draw.polygon([(1, 1), (3, 1), (1, 3)], fill=(255, 0, 0))
    draw.polygon([(6, 6), (8, 6), (6, 8)], fill=(0, 0, 255))
    classifier = ColorOnlyRouteDiscriminator(n_clusters=2)
    routes = classifier.get_routes([image], [holds()])
    assert len(routes) == 1
    assert sorted(len(route.holds) for route in routes[0]) == [1, 1]
    assert len(classifier.mark_routes([image], routes)) == 1


def test_color_classifier_validates_batch_alignment():
    classifier = ColorOnlyRouteDiscriminator(n_clusters=1)
    with pytest.raises(BatchAlignmentError):
        classifier.get_routes([Image.new("RGB", (1, 1))], [])
    with pytest.raises(BatchAlignmentError):
        classifier.mark_routes([Image.new("RGB", (1, 1))], [])


def test_factory_exposes_supported_methods_and_errors():
    factory = RouteDiscriminatorFactory()
    assert factory.available_methods() == ("color_only_discriminator", "triplet_route_discriminator")
