"""Implementation-neutral values shared by pipeline components."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, TypeAlias

from PIL import Image as PILImage

Image: TypeAlias = PILImage.Image


@dataclass(frozen=True)
class Coordinate:
    """Represents the coordinate of a pixel in an image."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0:
            raise ValueError(
                f"x or y was less than 0. This doesn't make any since for a coordinate. {self.x=} {self.y=}"
            )

        object.__setattr__(self, "x", int(self.x))
        object.__setattr__(self, "y", int(self.y))


@dataclass(frozen=True)
class Polygon:
    points: tuple[Coordinate, ...]

    def __post_init__(self) -> None:
        points = tuple(self.points)
        distinct_points = {(p.x, p.y) for p in points}
        if len(distinct_points) < 3:
            raise ValueError("A polygon needs at least three distinct points.")

        if any(not isinstance(p, Coordinate) for p in points):
            raise TypeError("A polygon should be made-up of Coordinates!")

        object.__setattr__(self, "points", points)


@dataclass(frozen=True)
class RGBColor:
    r: int
    g: int
    b: int

    def __post_init__(self) -> None:

        if not any(isinstance(v, int) for v in (self.r, self.g, self.b)):
            raise TypeError("RGB Values should be made up of integers!")
        if any(0 <= v <= 255 for v in (self.r, self.g, self.b)):
            raise ValueError("RGB values must be integers in [0, 255].")


@dataclass(frozen=True, eq=False)
class Hold:
    polygon: Polygon
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def centroid(self) -> Coordinate:
        points = self.polygon.points
        area_twice = 0  # Signed polygon area multiplied by two.
        centroid_x = 0
        centroid_y = 0

        previous = points[-1]
        for current in points:
            # Each edge contributes its cross product to the shoelace sums.
            cross = previous.x * current.y - current.x * previous.y
            area_twice += cross
            centroid_x += (previous.x + current.x) * cross
            centroid_y += (previous.y + current.y) * cross
            previous = current

        if area_twice == 0:
            # Collinear points have no area; use their average as a fallback.
            return Coordinate(
                round(sum(point.x for point in points) / len(points)),
                round(sum(point.y for point in points) / len(points)),
            )

        return Coordinate(
            round(centroid_x / (3 * area_twice)),
            round(centroid_y / (3 * area_twice)),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.centroid, Coordinate) or not isinstance(
            self.polygon, Polygon
        ):
            raise TypeError("Hold values must use shared data models.")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))

    def get_crop(self, image: Image) -> Image:
        if not isinstance(image, PILImage.Image):
            raise TypeError("image must be PIL.Image.Image.")
        xs, ys = [p.x for p in self.polygon.points], [p.y for p in self.polygon.points]
        return image.crop(
            (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)
        )

    def get_color(self, image: Image | None = None) -> RGBColor:
        if image is None and isinstance(self.attributes.get("color"), RGBColor):
            return self.attributes["color"]

        if not isinstance(image, Image):
            raise TypeError("An image is required to calculate color.")

        pixel = image.convert("RGB").getpixel(
            (round(self.centroid.x), round(self.centroid.y))
        )

        if (
            pixel is not None
            and not isinstance(pixel, float)
            and not isinstance(pixel, int)
        ):
            assert (l := len(pixel)) == 3, (
                f"For some reason the value returned for the pixel isn't RGB, it has len {l}."
            )
            return RGBColor(*pixel)
        else:
            raise TypeError(
                f"For some reason the pixel came back as an unexpected type: {type(pixel)}"
            )


@dataclass(frozen=True)
class Route:
    holds: set[Hold]
    route_id: int

    # TODO: Implement
    def mark_route(self, image: Image) -> Image: ...


# TODO: Define output objects for pipeline results.
