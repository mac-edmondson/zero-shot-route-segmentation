"""Public contract for automatic climbing-hold route discriminator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from PIL import ImageDraw

from ..interfaces.data_models import Hold, Image, Route
from ..interfaces.errors import BatchAlignmentError


class UnknownRouteDiscriminatorError(ValueError): ...


class InvalidRouteDiscriminatorConfigError(ValueError): ...


class RouteDiscriminator(Protocol):
    @property
    def implementation_id(self) -> str: ...

    def get_routes(
        self, images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> Sequence[Sequence[Route]]:
        """Return one detected-route list for every input image."""
        ...

    @staticmethod
    def mark_routes(
        images: Sequence[Image], routes: Sequence[Sequence[Route]]
    ) -> list[Image]:
        """Render hold overlays without changing detection behavior."""
        if len(images) != len(routes):
            raise BatchAlignmentError("images and routes must align.")

        palette = (
            (255, 80, 0),
            (0, 180, 255),
            (180, 80, 255),
            (80, 220, 80),
            (255, 200, 0),
            (255, 80, 180),
        )
        result = []
        for image, detected_routes in zip(images, routes, strict=True):
            overlay = image.convert("RGB").copy()
            draw = ImageDraw.Draw(overlay)
            for route_index, route in enumerate(detected_routes):
                color = palette[route_index % len(palette)]
                for hold in route.holds:
                    points = [
                        (point.x, point.y)
                        for point in (*hold.polygon.points, hold.polygon.points[0])
                    ]
                    draw.line(points, fill=color, width=3)
            result.append(overlay)
        return result
