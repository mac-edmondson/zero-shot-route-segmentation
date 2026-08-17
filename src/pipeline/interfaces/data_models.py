"""Implementation-neutral values shared by pipeline components."""
from __future__ import annotations
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias
from PIL import Image as PILImage

Image: TypeAlias = PILImage.Image

@dataclass(frozen=True)
class Coordinate:
    x: float
    y: float
    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) for v in (self.x, self.y)): raise ValueError("Coordinates must be finite.")
        object.__setattr__(self, "x", float(self.x)); object.__setattr__(self, "y", float(self.y))

@dataclass(frozen=True)
class Polygon:
    points: tuple[Coordinate, ...]
    def __post_init__(self) -> None:
        points = tuple(self.points)
        if len(points) < 3 or any(not isinstance(p, Coordinate) for p in points): raise ValueError("Polygon needs at least three points.")
        if len({(p.x, p.y) for p in points}) < 3: raise ValueError("Polygon needs three distinct points.")
        object.__setattr__(self, "points", points)

@dataclass(frozen=True)
class RGBColor:
    r: int
    g: int
    b: int
    def __post_init__(self) -> None:
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255 for v in (self.r, self.g, self.b)): raise ValueError("RGB values must be integers in [0, 255].")

@dataclass(frozen=True, eq=False)
class Hold:
    centroid: Coordinate
    polygon: Polygon
    attributes: Mapping[str, Any] = field(default_factory=dict)
    def __post_init__(self) -> None:
        if not isinstance(self.centroid, Coordinate) or not isinstance(self.polygon, Polygon): raise TypeError("Hold values must use shared data models.")
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
    def get_crop(self, image: Image) -> Image:
        if not isinstance(image, PILImage.Image): raise TypeError("image must be PIL.Image.Image.")
        xs, ys = [p.x for p in self.polygon.points], [p.y for p in self.polygon.points]
        return image.crop((int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1))
    def get_color(self, image: Image | None = None) -> RGBColor:
        if image is None and isinstance(self.attributes.get("color"), RGBColor): return self.attributes["color"]
        if not isinstance(image, PILImage.Image): raise ValueError("An image is required to calculate color.")
        pixel = image.convert("RGB").getpixel((round(self.centroid.x), round(self.centroid.y)))
        return RGBColor(*pixel)

@dataclass(frozen=True)
class Route:
    holds: set[Hold]
    route_id: int
    def mark_route(self, image: Image) -> Image: ...


# TODO: Define output objects for pipeline results.
