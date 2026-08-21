from __future__ import annotations

import io
import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image as PILImage

from ..pipeline.hold_detector.hold_detector_factory import hold_detector_factory
from ..pipeline.route_discriminator.route_discriminator_factory import (
    route_discriminator_factory,
)
from ..pipeline.route_discriminator_pipeline import RouteDiscriminatorPipeline
from .schemas import (
    AugmentWorkingImageRequest,
    Coordinate,
    DetectSegmentsResponse,
    HoldResult,
    InferWorkingResponse,
    Polygon,
    RouteResult,
    SegmentResult,
)
from .services.mock_segmentation import mock_segment_point

router = APIRouter()


@router.post("/image/working/segments", response_model=DetectSegmentsResponse)
async def detect_segments(
    image: UploadFile = File(...),
    all_points_x: str = Form(...),
    all_points_y: str = Form(...),
) -> DetectSegmentsResponse:
    """
    Batched hold-segmentation endpoint.

    Takes the working image plus every clicked point in one request
    (VIA-style `all_points_x`/`all_points_y` lists -- JSON-encoded as form
    fields since they travel alongside a file in multipart/form-data) and
    loops each point through the segmentation model (currently
    `mock_segmentation`, a stand-in for SAM3), returning one polygon per
    point.

    Stateless by design: the image is never persisted or looked up by id --
    it's sent whole with every request and only read into memory for the
    duration of this call (see progress.md / dashboard-backend discussion for
    why: no backend-side image storage was wanted for this pass).
    """
    try:
        xs: list[float] = json.loads(all_points_x)
        ys: list[float] = json.loads(all_points_y)
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail="all_points_x/all_points_y must be JSON number arrays",
        ) from exc

    if not isinstance(xs, list) or not isinstance(ys, list) or len(xs) != len(ys):
        raise HTTPException(
            status_code=422,
            detail="all_points_x and all_points_y must be equal-length arrays",
        )
    if not xs:
        raise HTTPException(status_code=422, detail="At least one point is required")

    # Read (and discard) so the upload completes cleanly. Not decoded/used by
    # the mock yet -- real SAM3 would decode these bytes per point below.
    await image.read()

    segments = [
        SegmentResult(
            segment_id=f"seg_{uuid.uuid4().hex[:12]}",
            polygon=mock_segment_point(Coordinate(x=x, y=y)),
        )
        for x, y in zip(xs, ys)
    ]
    return DetectSegmentsResponse(segments=segments)


@router.post("/image/working/augment", status_code=204)
async def augment_working_image(body: AugmentWorkingImageRequest) -> None:
    """
    No-op for now: accepts the frontend's existing lighting/chalk
    augmentation request (`Finish Augment`) so it stops 404ing, rather than
    silently blocking the frontend's post-augment model-select reveal.

    Doesn't actually bake anything into image bytes -- this call carries no
    image (stateless backend, see the segments endpoint's docstring above),
    and real chalk/lighting rendering is still unbuilt (progress.md's Next
    Steps: "Build POST /image/working/augment ... so Lighting/Chalk actually
    bake into the image").
    """
    return None


@router.post("/pipeline/infer/working", response_model=InferWorkingResponse)
async def infer_working_pipeline(
    image: UploadFile = File(...),
    hold_detector: str = Form("mock"),
    route_discriminator: str = Form("mock"),
    lighting_percent: float = Form(0.0),
    segments: str = Form("[]"),
) -> InferWorkingResponse:
    """
    Recognition: runs the full RouteDiscriminatorPipeline
    (src/pipeline/route_discriminator_pipeline.py, composing a HoldDetector +
    a RouteDiscriminator per docs/spec/pipeline/interfaces/route-discriminator-pipeline.md)
    on the working image and returns the detected routes -- one list of
    holds per route, per the REST spec
    (docs/diagrams/spec_rest_api.drawio.svg: "routes: list of routes (which
    is lists of holds)").

    Takes both the augmentation state gathered by `Finish Augment`
    (lighting_percent/segments) and the model selection made just before
    `Recognition` is pressed, in one request -- this backend is stateless
    (see /image/working/segments' docstring above) and never stores the
    working image between calls, so there's nowhere else to combine the two;
    the pipeline needs the image and both selections together to run at all.

    hold_detector/route_discriminator currently always resolve to the mock
    implementations (src/pipeline/*/mock_*.py) regardless of the string sent
    -- SAM3 (needs torch/transformers/cv2, deliberately not installed in
    this lightweight backend build) and the real per-method route
    discriminators from the proposal (color-only/DINO/combined) aren't wired
    in yet. lighting_percent/segments are accepted but not yet baked into
    pixels, same caveat as POST /image/working/augment above.
    """
    try:
        json.loads(segments)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail="segments must be a JSON array"
        ) from exc

    raw = await image.read()
    try:
        pil_image = PILImage.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Couldn't read image") from exc

    # TODO(@teammate): once hold_detector/route_discriminator support more
    # than "mock", map hold_detector/route_discriminator (received above)
    # into the factories' real keys instead of hardcoding "mock" here.
    detector = hold_detector_factory("mock")
    discriminator = route_discriminator_factory("mock")
    pipeline = RouteDiscriminatorPipeline(detector, discriminator)

    routes = pipeline.get_routes([pil_image])[0]
    width, height = pil_image.width, pil_image.height

    return InferWorkingResponse(
        routes=[
            RouteResult(
                route_id=route.route_id,
                holds=[
                    HoldResult(
                        centroid=Coordinate(
                            x=hold.centroid.x / width, y=hold.centroid.y / height
                        ),
                        polygon=Polygon(
                            points=[
                                Coordinate(x=p.x / width, y=p.y / height)
                                for p in hold.polygon.points
                            ]
                        ),
                    )
                    for hold in route.holds
                ],
            )
            for route in routes
        ],
        inference_metrics={
            "hold_count": float(sum(len(route.holds) for route in routes)),
            "route_count": float(len(routes)),
        },
    )


@router.delete("/image/working/segment/{segment_id}", status_code=204)
async def delete_segment(segment_id: str) -> None:
    """
    No-op: nothing is stored server-side to delete (see module docstring
    above). Kept as a real endpoint so the frontend's existing
    deleteWorkingSegment call has somewhere to land without special-casing,
    and so this matches the DELETE /image/working/segment/{id} shape already
    sketched in docs/diagrams/spec_rest_api.drawio.svg.
    """
    return None
