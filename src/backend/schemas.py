"""Wire models for the dashboard backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

# Shared normalized geometry


class Coordinate(BaseModel):
    """A normalized browser-space point."""

    x: float
    y: float

    @field_validator("x", "y")
    @classmethod
    def normalized(cls, value: float) -> float:
        """Reject coordinates outside the normalized image bounds."""
        if not 0 <= value <= 1:
            raise ValueError("coordinates must be normalized to [0, 1]")
        return value


class Polygon(BaseModel):
    """A normalized polygon with at least three points."""

    points: list[Coordinate] = Field(min_length=3)


# Segmentation


class SegmentResult(BaseModel):
    """A persisted working-image segment."""

    segment_id: str
    polygon: Polygon


class SegmentRequest(BaseModel):
    """Points requested for asynchronous segmentation."""

    coordinates: list[Coordinate] = Field(min_length=1)


class SegmentStatusResponse(BaseModel):
    """Current segmentation job state and results."""

    status: Literal["processing", "completed", "failed"]
    segments: list[SegmentResult] = Field(default_factory=list)
    error: str | None = None


# Image augmentation


class RGBColor(BaseModel):
    """An RGB colour channel triplet."""

    r: int = Field(ge=0, le=255)
    g: int = Field(ge=0, le=255)
    b: int = Field(ge=0, le=255)


class SegmentAugmentation(BaseModel):
    """Chalk and optional colour changes for one segment."""

    model_config = ConfigDict(alias_generator=to_camel)

    segment_id: str
    chalk_percent: float = Field(ge=0, le=100)
    color: RGBColor | None = None


class AugmentWorkingImageRequest(BaseModel):
    """Image-wide lighting and per-segment augmentation request."""

    model_config = ConfigDict(alias_generator=to_camel)

    lighting_percent: float = Field(ge=-1, le=1, default=0)
    segments: list[SegmentAugmentation] = Field(default_factory=list)


# Working image


class WorkingImageResponse(BaseModel):
    """Working image state and display data."""

    status: Literal["processing", "completed", "failed"]
    image: str | None = None
    image_id: str | None = None
    error: str | None = None


# Pipeline inference


class HoldResult(BaseModel):
    """A detected hold represented in normalized browser coordinates."""

    centroid: Coordinate
    polygon: Polygon


class RouteResult(BaseModel):
    """A classified route and its holds."""

    route_id: int
    holds: list[HoldResult]


class InferWorkingResponse(BaseModel):
    """Current inference job state and result."""

    status: Literal["processing", "completed", "failed"] = "completed"
    routes: list[RouteResult] = Field(default_factory=list)
    inference_metrics: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


# Pipeline configuration


class PipelineConfig(BaseModel):
    """Selected hold detector and route classifier."""

    model_config = ConfigDict(alias_generator=to_camel)

    hold_detector: str
    route_classifier: str


class AvailableConfigsResponse(BaseModel):
    """Pipeline implementations available for selection."""

    hold_detector: list[str]
    route_classifier: list[str]
