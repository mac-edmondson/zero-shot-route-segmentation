from __future__ import annotations

from collections.abc import Sequence

from ..interfaces.data_models import Hold, Image, Route
from .route_discriminator import RouteDiscriminator

import numpy as np

MIN_NUM_ROUTES_PER_IMAGE = 0
MAX_NUM_ROUTES_PER_IMAGE = 20


class MockRouteDiscriminator(RouteDiscriminator):
    @property
    def implementation_id(self) -> str:
        return "mock_route_discriminator"

    def __init__(self, **config):
        seed = 12345
        if "seed" in config:
            seed = config["seed"]
        self.rng = np.random.default_rng(seed)

    def get_routes(
        self, images: Sequence[Image], holds: Sequence[Sequence[Hold]]
    ) -> Sequence[Sequence[Route]]:
        """Return one detected-route list for every input image."""
        assert len(images) == len(holds)

        routes = [[] for _ in range(len(images))]
        for i, (_, img_holds) in enumerate(zip(images, holds)):
            if len(img_holds) == 0:
                continue  # There are no holds to make routes out of

            num_routes = self.rng.integers(
                MIN_NUM_ROUTES_PER_IMAGE,
                min(len(img_holds), MAX_NUM_ROUTES_PER_IMAGE)
                + 1,  # High is exclusive, thus the plus 1
            )
            assert num_routes <= len(img_holds)

            if num_routes == 0:
                continue

            # Returns approximately equal sized groups
            img_routes = np.array_split(
                self.rng.permutation(np.array(img_holds)), num_routes
            )
            routes[i] = [Route(set(img_routes[j]), j) for j in range(num_routes)]

        return routes
