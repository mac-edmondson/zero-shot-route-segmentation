"""Wire models for the dashboard backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel


class Coordinate(BaseModel):
    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def normalized(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("coordinates must be normalized to [0, 1]")
        return value


class Polygon(BaseModel):
    points: list[Coordinate] = Field(min_length=3)


class SegmentResult(BaseModel):
    segment_id: str
    polygon: Polygon


class DetectSegmentsResponse(BaseModel):
    segments: list[SegmentResult]


class SegmentRequest(BaseModel):
    coordinates: list[Coordinate] = Field(min_length=1)


class SegmentStatusResponse(BaseModel):
    status: Literal["processing", "completed", "failed"]
    segments: list[SegmentResult] = Field(default_factory=list)
    error: str | None = None


class RGBColor(BaseModel):
    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class SegmentAugmentation(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    segment_id: str
    chalk_percent: float = Field(ge=0, le=100)
    color: RGBColor | None = None


class AugmentWorkingImageRequest(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    # -1 to 1, 0 = neutral/no change -- the same scale and neutral point as
    # LightingAugmentationParams.intensity (src/pipeline/interfaces/
    # augmentation.py) itself, which augment_image passes this straight
    # into unconverted (see src/backend/services/dashboard.py). Named
    # lighting_percent (not e.g. lighting_intensity) for backward
    # compatibility with the existing wire field/alias -- only the range
    # changed, not the field itself.
    lighting_percent: float = Field(ge=-1, le=1, default=0)
    segments: list[SegmentAugmentation] = Field(default_factory=list)


class WorkingImageResponse(BaseModel):
    status: Literal["processing", "completed", "failed"]
    image: str | None = None
    image_id: str | None = None
    error: str | None = None


class HoldResult(BaseModel):
    centroid: Coordinate
    polygon: Polygon


class RouteResult(BaseModel):
    route_id: int
    holds: list[HoldResult]


class InferWorkingResponse(BaseModel):
    status: Literal["processing", "completed", "failed"] = "completed"
    routes: list[RouteResult] = Field(default_factory=list)
    inference_metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class PipelineConfig(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    hold_detector: str
    route_classifier: str


class AvailableConfigsResponse(BaseModel):
    hold_detector: list[str]
    route_classifier: list[str]
