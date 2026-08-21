from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..interfaces.data_models import Coordinate, Hold, Image, Polygon
from .hold_detector import HoldDetector

# Number of holds returned for each non-empty image.
MIN_NUM_HOLDS_PER_IMAGE = 0
MAX_NUM_HOLDS_PER_IMAGE = 100

# Number of vertices in each generated polygon.
MIN_NUM_POINTS_PER_HOLD = 10
MAX_NUM_POINTS_PER_HOLD = 200

# Distance from a hold's generated centroid to each vertex, in pixels.
# Note: if this isn't 1, tests fall into some weird
# infinite loop situation. Don't think about it, much. Just leave it as is though
# , lol.
MIN_POINT_DISTANCE = 1
MAX_POINT_DISTANCE = 100

# Keep generated geometry bounded and reasonably varied for mock data.
MAX_OVERLAP_ATTEMPTS = 10


class MockHoldDetector(HoldDetector):
    @property
    def implementation_id(self) -> str:
        return "mock_hold_detector"

    def __init__(self, **config):
        seed = 12345
        if "seed" in config:
            seed = config["seed"]
        self.rng = np.random.default_rng(seed)

    def get_holds(self, images: Sequence[Image]) -> Sequence[Sequence[Hold]]:
        """Return one detected-hold list for every input image."""
        holds = [[] for _ in images]
        for i, image in enumerate(images):
            if image.width == 0 or image.height == 0:
                continue

            if image.width < 2 or image.height < 2:
                continue

            num_holds = self.rng.integers(
                MIN_NUM_HOLDS_PER_IMAGE,
                MAX_NUM_HOLDS_PER_IMAGE + 1,
            )
            # Retry a few candidates to avoid crowded bounding boxes when possible.
            image_holds = []
            for _ in range(num_holds):
                hold = self._random_hold(image)
                for _ in range(MAX_OVERLAP_ATTEMPTS):
                    if not any(
                        self._bounding_boxes_overlap(hold, previous)
                        for previous in image_holds
                    ):
                        break
                    hold = self._random_hold(image)
                image_holds.append(hold)
            holds[i] = image_holds

        return holds

    def _random_hold(self, image: Image) -> Hold:
        point_count = int(
            self.rng.integers(MIN_NUM_POINTS_PER_HOLD, MAX_NUM_POINTS_PER_HOLD + 1)
        )
        centroid_x = int(self.rng.integers(0, image.width))
        centroid_y = int(self.rng.integers(0, image.height))
        # Build a radial polygon around a random pixel centroid.
        points = []
        while len(points) < point_count:
            angle = self.rng.uniform(0, 2 * np.pi)
            direction_x, direction_y = np.cos(angle), np.sin(angle)
            limits = []
            if direction_x > 0:
                limits.append((image.width - 1 - centroid_x) / direction_x)
            elif direction_x < 0:
                limits.append(-centroid_x / direction_x)
            if direction_y > 0:
                limits.append((image.height - 1 - centroid_y) / direction_y)
            elif direction_y < 0:
                limits.append(-centroid_y / direction_y)

            # Limit the radius to the image edge before rounding to pixel coordinates.
            max_distance = min(MAX_POINT_DISTANCE, *limits)
            if max_distance < MIN_POINT_DISTANCE:
                continue

            distance = self.rng.uniform(MIN_POINT_DISTANCE, max_distance)
            point = (
                int(round(centroid_x + distance * direction_x)),
                int(round(centroid_y + distance * direction_y)),
            )
            actual_distance = np.hypot(
                point[0] - centroid_x,
                point[1] - centroid_y,
            )
            if MIN_POINT_DISTANCE <= actual_distance <= MAX_POINT_DISTANCE:
                points.append(point)

        # Tiny images can quantize several samples to the same pixel.
        distinct_points = set(points)
        if len(distinct_points) < 3:
            replacements = []
            for x in range(image.width):
                for y in range(image.height):
                    point = (x, y)
                    if (
                        point != (centroid_x, centroid_y)
                        and point not in distinct_points
                    ):
                        replacements.append(point)
                        if len(replacements) == 3 - len(distinct_points):
                            break
                if len(replacements) == 3 - len(distinct_points):
                    break
            points[: len(replacements)] = replacements

        # Angular ordering turns the samples into a drawable polygon.
        points.sort(
            key=lambda point: np.arctan2(
                point[1] - centroid_y,
                point[0] - centroid_x,
            )
        )
        return Hold(Polygon(tuple(Coordinate(x, y) for x, y in points)))

    # Bounding boxes are cheap; overlap avoidance is intentionally best-effort.
    @staticmethod
    def _bounding_boxes_overlap(first: Hold, second: Hold) -> bool:
        first_points = first.polygon.points
        second_points = second.polygon.points
        first_xs = [point.x for point in first_points]
        first_ys = [point.y for point in first_points]
        second_xs = [point.x for point in second_points]
        second_ys = [point.y for point in second_points]
        return not (
            max(first_xs) < min(second_xs)
            or max(second_xs) < min(first_xs)
            or max(first_ys) < min(second_ys)
            or max(second_ys) < min(first_ys)
        )
