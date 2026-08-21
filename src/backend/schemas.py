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

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


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


class SegmentAugmentation(BaseModel):
    """One segment's chalk augmentation, as sent by `POST /image/working/augment`
    (src/frontend/lib/api/types.ts: SegmentAugmentation). JSON body, not a
    form -- unlike /image/working/segments, so camelCase from the frontend is
    accepted directly via the alias generator rather than needing a
    JSON-encoded form field."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    segment_id: str
    chalk_percent: float


class AugmentWorkingImageRequest(BaseModel):
    """Body of `POST /image/working/augment`
    (src/frontend/lib/api/types.ts: AugmentWorkingImageRequest). No image
    bytes -- the backend is stateless and this call doesn't carry the working
    image, so there's nothing here to bake chalk/lighting into yet."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    lighting_percent: float
    segments: list[SegmentAugmentation]


class HoldResult(BaseModel):
    """One hold within a detected route, returned by
    `POST /pipeline/infer/working` -- shape matches SegmentResult's polygon
    convention (normalized [0, 1] coordinates), plus the centroid the shared
    pipeline data model (`src/pipeline/interfaces/data_models.py::Hold`)
    already computes."""

    centroid: Coordinate
    polygon: Polygon


class RouteResult(BaseModel):
    route_id: int
    holds: list[HoldResult]


class InferWorkingResponse(BaseModel):
    """Response of `POST /pipeline/infer/working`
    (src/frontend/lib/api/types.ts: InferenceResult) -- one route list for
    the working image, each route a list of holds ("list of routes (which is
    lists of holds)" per docs/diagrams/spec_rest_api.drawio.svg)."""

    routes: list[RouteResult]
    inference_metrics: dict[str, float] = {}


class AvailableConfigsResponse(BaseModel):
    """Response of `GET /pipeline/available_configs` -- the hold-detector/
    route-classifier implementation names the model-select dropdowns
    (WallImageWorkspace) should offer, straight from the pipeline's own
    factory registries (src/pipeline/*/*_factory.py). No camelCase alias
    here: unlike AugmentWorkingImageRequest above, this is a response the
    backend produces, not a body it has to accept from the frontend, so it's
    free to just use the field names the REST spec already gives in
    snake_case."""

    hold_detector: list[str]
    route_classifier: list[str]
