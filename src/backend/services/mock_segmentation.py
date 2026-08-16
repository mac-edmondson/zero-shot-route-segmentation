"""
Mock stand-in for SAM3 (docs/spec/pipeline/interfaces/hold-detector.md).

Given one clicked point, synthesizes a small randomized polygon around it so
the frontend has real (if fake) polygon data to draw end to end before the
actual model is wired in -- randomized, rather than one fixed shape stamped
at every point, so each mock "hold" reads as its own detection instead of an
obvious copy-paste. Swap `mock_segment_point`'s body for a real SAM3
point-prompt inference call later -- the request/response contract in
routes.py is built to not need to change when that happens (it's already
stateless per point, already returns one polygon per point).

Kept in sync with the frontend's own mock
(src/frontend/lib/api/mocks/mockClient.ts's mockPolygonAround) so mock and
real (mock-backed) modes look the same.
"""

from __future__ import annotations

import math
import random

from ..schemas import Coordinate, Polygon

_MIN_SIDES = 6
_MAX_SIDES = 10
# Normalized units -- roughly a hold-sized blob at typical image scale.
_MIN_RADIUS = 0.015
_MAX_RADIUS = 0.035
# Per-vertex radius jitter, as a fraction of that polygon's base radius --
# keeps vertices irregular/organic rather than a perfect regular polygon.
_VERTEX_JITTER = 0.3


def mock_segment_point(point: Coordinate) -> Polygon:
    sides = random.randint(_MIN_SIDES, _MAX_SIDES)
    base_radius = random.uniform(_MIN_RADIUS, _MAX_RADIUS)
    points = [
        Coordinate(
            x=point.x
            + base_radius
            * random.uniform(1 - _VERTEX_JITTER, 1 + _VERTEX_JITTER)
            * math.cos(2 * math.pi * i / sides),
            y=point.y
            + base_radius
            * random.uniform(1 - _VERTEX_JITTER, 1 + _VERTEX_JITTER)
            * math.sin(2 * math.pi * i / sides),
        )
        for i in range(sides)
    ]
    return Polygon(points=points)
