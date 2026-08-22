"""Public contract for automatic climbing-hold route discriminator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image as PILImage, ImageDraw

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
            fill_overlay = PILImage.new("RGBA", overlay.size, (0, 0, 0, 0))
            fill_draw = ImageDraw.Draw(fill_overlay)
            for route_index, route in enumerate(detected_routes):
                color = palette[route_index % len(palette)]
                for hold in route.holds:
                    points = [
                        (point.x, point.y)
                        for point in (*hold.polygon.points, hold.polygon.points[0])
                    ]
                    fill_draw.polygon(points, fill=(*color, 75))
            overlay = PILImage.alpha_composite(
                overlay.convert("RGBA"), fill_overlay
            ).convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for route_index, route in enumerate(detected_routes):
                color = palette[route_index % len(palette)]
                for hold in route.holds:
                    points = [
                        (point.x, point.y)
                        for point in (*hold.polygon.points, hold.polygon.points[0])
                    ]
                    draw.line(points, fill=color, width=5)
            result.append(overlay)
        return result


def _kmeans(
    features: np.ndarray, n_clusters: int, random_state: int | None
) -> np.ndarray:
    """Return deterministic k-means labels without an additional dependency."""
    rng = np.random.default_rng(random_state)
    centers = features[rng.choice(len(features), n_clusters, replace=False)]
    for _ in range(100):
        labels = ((features[:, None] - centers) ** 2).sum(2).argmin(1)
        updated = np.array(
            [
                features[labels == index].mean(0)
                if np.any(labels == index)
                else centers[index]
                for index in range(n_clusters)
            ]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels
