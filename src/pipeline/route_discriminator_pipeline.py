from .hold_detector import HoldDetector
from .route_discriminator import RouteDiscriminator
from dataclasses import dataclass
from collections.abc import Sequence
from .interfaces.data_models import Route, Image


@dataclass(frozen=True)
class PipelineDescriptor:
    detector_id: str
    detector_config: dict
    discriminator_id: str
    discriminator_config: dict


class RouteDiscriminatorPipeline:
    def __init__(
        self,
        hold_detector: HoldDetector,
        route_discriminator: RouteDiscriminator,
    ):
        self.hold_detector = hold_detector
        self.route_discriminator = route_discriminator
        self._last_holds = None
        self._last_routes = None

    def get_routes(self, images: Sequence) -> Sequence[Sequence[Route]]:
        """Return one detected-route list for every input image."""
        self._last_holds = self.hold_detector.get_holds(images=images)
        self._last_routes = self.route_discriminator.get_routes(
            images=images, holds=self._last_holds
        )
        return self._last_routes

    @staticmethod
    def mark_routes(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> Sequence[Image]:
        """Render route overlays on an image."""
        return RouteDiscriminator.mark_routes(images, routes)

    @staticmethod
    def mark_holds(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> Sequence[Image]:
        """Render hold overlays on an image."""
        return HoldDetector.mark_holds(images, routes)
