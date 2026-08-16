"""
Mock stand-in for SAM3 (docs/spec/pipeline/interfaces/hold-detector.md).

Given one clicked point, synthesizes a small polygon around it so the
frontend has real polygon data to draw end to end before the actual model is
wired in. Swap `mock_segment_point`'s body for a real SAM3 point-prompt
inference call later -- the request/response contract in routes.py is built
to not need to change when that happens (it's already stateless per point,
already returns one polygon per point).
"""

from __future__ import annotations

import math

from ..schemas import Coordinate, Polygon

_SIDES = 8
_RADIUS = 0.025  # normalized units -- roughly a hold-sized blob at typical image scale


def mock_segment_point(point: Coordinate) -> Polygon:
    points = [
        Coordinate(
            x=point.x + _RADIUS * math.cos(2 * math.pi * i / _SIDES),
            y=point.y + _RADIUS * math.sin(2 * math.pi * i / _SIDES),
        )
        for i in range(_SIDES)
    ]
    return Polygon(points=points)
