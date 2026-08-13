from dataclasses import dataclass


class Image:
    """Opaque image-like value accepted by pipeline components."""

    pass  # TODO: Define this


# TODO: Enforce lower-bound constraint
@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float


@dataclass(frozen=True)
class Polygon:
    points: tuple[Coordinate, ...]


# TODO: Define upper and lower bound constraints
@dataclass(frozen=True)
class RGBColor:
    r: int
    g: int
    b: int


# TODO: Define constraints that the centroid must be within the polygon
# TODO: Define constraints that the polygon must contain at least three points
@dataclass(frozen=True)
class Hold:
    centroid: Coordinate
    polygon: Polygon

    def get_crop(self, image: Image) -> Image: ...
    def get_color(self, image: Image | None = None) -> RGBColor: ...


@dataclass(frozen=True)
class Route:
    holds: set[Hold]
    route_id: int  # Unique route identifier

    def mark_route(self, image: Image) -> Image: ...


# TODO: Define some output objects for use in the pipeline, rather than
# passing lists all over the place.
