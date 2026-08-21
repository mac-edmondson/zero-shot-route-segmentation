from __future__ import annotations

import base64
import binascii
import io
import uuid
from collections.abc import Iterable, Sequence

from PIL import Image as PILImage

# This fallback keeps the imports testable when pytest loads `backend` top-level.
try:
    from ...pipeline.hold_detector.hold_detector_factory import hold_detector_factory
    from ...pipeline.interfaces.augmentation import (
        ChalkAugmentationParams,
        ColorAugmentationParams,
        LightingAugmentationParams,
    )
    from ...pipeline.interfaces.data_models import (
        Coordinate as PixelCoordinate,
    )
    from ...pipeline.interfaces.data_models import (
        Polygon as PixelPolygon,
    )
    from ...pipeline.interfaces.data_models import (
        RGBColor as PipelineRGBColor,
    )
    from ...pipeline.route_discriminator.route_discriminator_factory import (
        route_discriminator_factory,
    )
    from ...pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline
except ImportError:  # Support `PYTHONPATH=src` development imports.
    from pipeline.hold_detector.hold_detector_factory import hold_detector_factory
    from pipeline.interfaces.augmentation import (
        ChalkAugmentationParams,
        ColorAugmentationParams,
        LightingAugmentationParams,
    )
    from pipeline.interfaces.data_models import (
        Coordinate as PixelCoordinate,
    )
    from pipeline.interfaces.data_models import (
        Polygon as PixelPolygon,
    )
    from pipeline.interfaces.data_models import (
        RGBColor as PipelineRGBColor,
    )
    from pipeline.route_discriminator.route_discriminator_factory import (
        route_discriminator_factory,
    )
    from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline
from ..schemas import (
    AugmentWorkingImageRequest,
    Coordinate,
    HoldResult,
    InferWorkingResponse,
    Polygon,
    RouteResult,
    SegmentResult,
)
from .mock_segmentation import mock_segment_point


def decode_image(raw: bytes) -> PILImage.Image:
    try:
        with PILImage.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Couldn't read image") from exc


def decode_data_url(value: str) -> PILImage.Image:
    if not value.startswith("data:") or "," not in value:
        raise ValueError("image must be an image data URL")
    encoded = value.split(",", 1)[1]
    try:
        return decode_image(base64.b64decode(encoded, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image must be a valid image data URL") from exc


def image_data_url(image: PILImage.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def new_image_id() -> str:
    return str(uuid.uuid4())


def detect_segments(coordinates: Sequence[Coordinate]) -> list[SegmentResult]:
    # TODO: This should be actually implemented and the mock_segmentation
    # service then removed. This can't be implemented until a proper SAM3
    # segmenter is in place though.
    return [
        SegmentResult(
            segment_id=f"seg_{uuid.uuid4().hex[:12]}",
            polygon=mock_segment_point(coordinate),
        )
        for coordinate in coordinates
    ]


def _pixel_polygon(polygon: Polygon, image: PILImage.Image) -> PixelPolygon:
    width, height = image.size
    points = tuple(
        PixelCoordinate(
            round(point.x * max(width - 1, 1)),
            round(point.y * max(height - 1, 1)),
        )
        for point in polygon.points
    )
    if len({(point.x, point.y) for point in points}) < 3:
        raise ValueError("segment polygon is too small for this image")
    return PixelPolygon(points)


def _api_polygon(polygon: PixelPolygon, image: PILImage.Image) -> Polygon:
    width, height = image.size
    return Polygon(
        points=[
            Coordinate(
                x=point.x / max(width - 1, 1),
                y=point.y / max(height - 1, 1),
            )
            for point in polygon.points
        ]
    )


def augment_image(
    image: PILImage.Image,
    request: AugmentWorkingImageRequest,
    segments: Iterable[SegmentResult],
) -> PILImage.Image:
    try:
        from ...pipeline.preprocessing.augmentation_suite import AugmentationSuite
    except ImportError:
        from pipeline.preprocessing.augmentation_suite import AugmentationSuite
    suite = AugmentationSuite()
    result = image.convert("RGB").copy()
    by_id = {segment.segment_id: segment for segment in segments}

    for augmentation in request.segments:
        segment = by_id.get(augmentation.segment_id)
        if segment is None:
            raise KeyError(augmentation.segment_id)
        polygon = _pixel_polygon(segment.polygon, result)
        result = suite.add_chalk(
            result,
            [polygon],
            ChalkAugmentationParams(augmentation.chalk_percent / 100),
            seed=0,
        )
        if augmentation.color is not None:
            color = PipelineRGBColor(
                augmentation.color.r,
                augmentation.color.g,
                augmentation.color.b,
            )
            result = suite.change_color(
                result,
                [polygon],
                ColorAugmentationParams(color, augmentation.chalk_percent / 100),
            )

    if request.lighting_percent:
        result = suite.change_lighting(
            result,
            LightingAugmentationParams(request.lighting_percent / 100),
        )
    return result


def infer(
    image: PILImage.Image,
    hold_detector: str,
    route_classifier: str,
) -> InferWorkingResponse:
    detector = hold_detector_factory(hold_detector)
    discriminator = route_discriminator_factory(route_classifier)
    routes = RouteDiscriminatorPipeline(detector, discriminator).get_routes([image])[0]
    result_routes: list[RouteResult] = []
    for route in routes:
        holds = sorted(route.holds, key=lambda hold: (hold.centroid.y, hold.centroid.x))
        result_routes.append(
            RouteResult(
                route_id=route.route_id,
                holds=[
                    HoldResult(
                        centroid=Coordinate(
                            x=hold.centroid.x / max(image.width - 1, 1),
                            y=hold.centroid.y / max(image.height - 1, 1),
                        ),
                        polygon=_api_polygon(hold.polygon, image),
                    )
                    for hold in holds
                ],
            )
        )
    return InferWorkingResponse(
        status="completed",
        routes=result_routes,
        inference_metrics={
            "hold_count": float(sum(len(route.holds) for route in routes)),
            "route_count": float(len(routes)),
        },
    )


def model_error(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__
