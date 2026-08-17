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


class GalleryCategory(BaseModel):
    """One subdirectory of the gallery root (e.g. "bh", "sm") -- the external
    server groups its images this way, so the picker lists categories first
    instead of loading every image across every category at once."""

    name: str


class GalleryCategoriesResponse(BaseModel):
    categories: list[GalleryCategory]


class GalleryImage(BaseModel):
    name: str
    category: str
    """Direct URL on the external gallery server -- safe to use as an <img
    src> on the frontend as-is (CORS only blocks JS from reading bytes, not
    the browser from rendering an <img>); only fetching raw bytes needs the
    /gallery/images/{category}/{name} proxy below."""
    url: str


class GalleryListResponse(BaseModel):
    images: list[GalleryImage]
