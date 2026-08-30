"""Image processing adapters for the dashboard API."""

from __future__ import annotations

import base64
import io
import uuid
from collections.abc import Iterable, Sequence

from PIL import Image as PILImage

# This fallback keeps the imports testable when pytest loads `backend` top-level.
try:
    from ...pipeline.hold_detector.hold_detector_factory import hold_detector_factory
    from ...pipeline.interfaces.data_models import (
        Coordinate as PixelCoordinate,
    )
    from ...pipeline.interfaces.data_models import (
        Polygon as PixelPolygon,
    )
    from ...pipeline.interfaces.data_models import (
        RGBColor as PipelineRGBColor,
    )
    from ...pipeline.preprocessing.augmentation_suite import (
        AugmentationPlan,
        ChalkAugmentation,
        ColorAugmentation,
        LightingAugmentation,
    )
    from ...pipeline.route_discriminator.route_discriminator_factory import (
        route_discriminator_factory,
    )
    from ...pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline
    from ...pipeline.utility.sam3_ui import SAMWrapper
except ImportError:  # Support `PYTHONPATH=src` development imports.
    from pipeline.hold_detector.hold_detector_factory import hold_detector_factory
    from pipeline.interfaces.data_models import (
        Coordinate as PixelCoordinate,
    )
    from pipeline.interfaces.data_models import (
        Polygon as PixelPolygon,
    )
    from pipeline.interfaces.data_models import (
        RGBColor as PipelineRGBColor,
    )
    from pipeline.preprocessing.augmentation_suite import (
        AugmentationPlan,
        ChalkAugmentation,
        ColorAugmentation,
        LightingAugmentation,
    )
    from pipeline.route_discriminator.route_discriminator_factory import (
        route_discriminator_factory,
    )
    from pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline
    from pipeline.utility.sam3_ui import SAMWrapper
from ..schemas import (
    AugmentWorkingImageRequest,
    Coordinate,
    HoldResult,
    InferWorkingResponse,
    Polygon,
    RouteResult,
    SegmentResult,
)

_sam_wrapper = SAMWrapper()


def decode_image(raw: bytes) -> PILImage.Image:
    """Decode upload bytes as an RGB image."""
    try:
        with PILImage.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("Couldn't read image") from exc


def image_data_url(image: PILImage.Image) -> str:
    """Encode an image as a PNG data URL."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def new_image_id() -> str:
    """Return a fresh identifier for a working image."""
    return str(uuid.uuid4())


def detect_segments(
    image: PILImage.Image, coordinates: Sequence[Coordinate]
) -> list[SegmentResult]:
    """Segment each normalized click with SAM3."""
    if _sam_wrapper.model is None or _sam_wrapper.processor is None:
        _sam_wrapper.load_model()
    masks = _sam_wrapper.generate_mask(
        image, [{"x": coordinate.x, "y": coordinate.y} for coordinate in coordinates]
    )
    polygons = _sam_wrapper.to_polygons(masks)
    if len(polygons) != len(coordinates):
        raise RuntimeError("SAM3 returned an unexpected number of masks.")

    width, height = image.size
    results = []
    for points in polygons:
        if len(points) < 3:
            raise ValueError("SAM3 returned an empty or degenerate mask.")
        results.append(
            SegmentResult(
                segment_id=f"seg_{uuid.uuid4().hex[:12]}",
                polygon=Polygon(
                    points=[
                        Coordinate(x=x / max(width - 1, 1), y=y / max(height - 1, 1))
                        for x, y in points
                    ]
                ),
            )
        )
    return results


def _pixel_polygon(polygon: Polygon, image: PILImage.Image) -> PixelPolygon:
    """Convert a normalized polygon to image pixel coordinates."""
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
    """Convert a pixel polygon to normalized API coordinates."""
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
    """Apply requested colour, chalk, then lighting augmentations."""
    by_id = {segment.segment_id: segment for segment in segments}
    chalk_targets = []
    chalk_strengths = []
    color_targets = []
    colors = []

    for augmentation in request.segments:
        segment = by_id.get(augmentation.segment_id)
        if segment is None:
            raise KeyError(augmentation.segment_id)
        polygon = _pixel_polygon(segment.polygon, image)
        chalk_targets.append(polygon)
        chalk_strengths.append(augmentation.chalk_percent / 100)
        if augmentation.color is not None:
            color_targets.append(polygon)
            colors.append(
                PipelineRGBColor(
                    augmentation.color.r,
                    augmentation.color.g,
                    augmentation.color.b,
                )
            )

    augmentations = []
    if color_targets:
        augmentations.append(ColorAugmentation(tuple(color_targets), tuple(colors)))
    if chalk_targets:
        augmentations.append(
            ChalkAugmentation(tuple(chalk_targets), tuple(chalk_strengths))
        )
    if request.lighting_percent:
        augmentations.append(LightingAugmentation(request.lighting_percent))
    return AugmentationPlan(tuple(augmentations), seed=0).apply(image)


def infer(
    image: PILImage.Image,
    hold_detector: str,
    route_classifier: str,
) -> InferWorkingResponse:
    """Run the selected pipeline and normalize its route result."""
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
    """Return a client-safe message for a background job failure."""
    return str(exc) or exc.__class__.__name__
