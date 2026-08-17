"""
Pydantic models for the Dashboard Backend API
(docs/spec/pipeline/interfaces/dashboard-backend.md).

Coordinates are normalized to [0, 1] of the image, matching the frontend's
existing convention (see src/frontend/components/image-canvas/ImageCanvas.tsx)
-- this keeps the wire contract stable regardless of the actual image's pixel
dimensions. A real SAM3 integration would convert to/from pixel space
internally using the image bytes it already receives; callers on either side
never need to know about that conversion.
"""

from __future__ import annotations

from pydantic import BaseModel


class Coordinate(BaseModel):
    x: float
    y: float


class Polygon(BaseModel):
    points: list[Coordinate]


class SegmentResult(BaseModel):
    """One detected hold -- the polygon returned for one clicked point."""

    segment_id: str
    polygon: Polygon


class DetectSegmentsResponse(BaseModel):
    segments: list[SegmentResult]
