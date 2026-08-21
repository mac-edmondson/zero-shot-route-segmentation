from .hold_detector import HoldDetector
from .route_discriminator import RouteDiscriminator
from dataclasses import dataclass
from collections.abc import Sequence
from .interfaces.data_models import Hold, Route, Image


@dataclass(frozen=True)
class PipelineDescriptor:
    detector_id: str
    detector_config: dict
    classifier_id: str
    classifier_config: dict


class RouteDiscriminatorPipeline:
    def __init__(
        self,
        hold_detector: HoldDetector,
        route_discriminator: RouteDiscriminator,
    ):
        self.hold_detector = hold_detector
        self.route_discriminator = route_discriminator

    def get_routes(self, images: Sequence) -> Sequence[Sequence[Route]]:
        """Return one detected-route list for every input image."""
        holds = self.hold_detector.get_holds(images=images)
        return self.route_discriminator.get_routes(images=images, holds=holds)

    @staticmethod
    def mark_routes(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> Sequence[Image]:
        """Render hold overlays without changing detxection behavior."""
        return RouteDiscriminator.mark_routes(images, routes)
