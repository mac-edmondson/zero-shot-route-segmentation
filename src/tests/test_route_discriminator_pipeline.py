from PIL import Image

from pipeline.hold_detector.mock_hold_detector import MockHoldDetector
from pipeline.route_discriminator.mock_route_discriminator import (
    MockRouteDiscriminator,
)
from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline


def test_mock_pipeline_returns_routes():
    images = [Image.new("RGB", (100, 100)) for _ in range(2)]
    pipeline = RouteDiscriminatorPipeline(
        hold_detector=MockHoldDetector(seed=12345),
        route_discriminator=MockRouteDiscriminator(seed=12345),
    )

    routes = pipeline.get_routes(images)

    assert len(routes) == len(images)
    assert any(routes)
    assert all(route.holds for image_routes in routes for route in image_routes)
