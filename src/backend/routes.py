from __future__ import annotations

import json
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from starlette.datastructures import UploadFile as StarletteUploadFile

# This fallback keeps the imports testable when pytest loads `backend` top-level.
try:
    from ..pipeline.hold_detector.hold_detector_factory import (
        AVAILABLE_HOLD_DETECTORS,
        UnknownHoldDetectorError,
    )
    from ..pipeline.route_discriminator.route_discriminator_factory import (
        AVAILABLE_ROUTE_DISCRIMINATORS,
        UnknownRouteDiscriminatorError,
    )
except ImportError:
    from pipeline.hold_detector.hold_detector_factory import (
        AVAILABLE_HOLD_DETECTORS,
        UnknownHoldDetectorError,
    )
    from pipeline.route_discriminator.route_discriminator_factory import (
        AVAILABLE_ROUTE_DISCRIMINATORS,
        UnknownRouteDiscriminatorError,
    )

from .dependencies import get_session
from .schemas import (
    AugmentWorkingImageRequest,
    AvailableConfigsResponse,
    Coordinate,
    DetectSegmentsResponse,
    InferWorkingResponse,
    PipelineConfig,
    SegmentAugmentation,
    SegmentRequest,
    SegmentStatusResponse,
    WorkingImageResponse,
)
from .services.dashboard import (
    augment_image,
    decode_data_url,
    decode_image,
    detect_segments,
    image_data_url,
    infer,
    model_error,
    new_image_id,
)
from .session_store import JobState, SessionState

router = APIRouter()


def _conflict(message: str) -> HTTPException:
    return HTTPException(status_code=409, detail=message)


def _set_working_image(session: SessionState, image: Any) -> None:
    with session.lock:
        session.working_image = image
        session.working_image_id = new_image_id()
        session.generation += 1
        session.segments.clear()
        session.segment_job = JobState()
        session.image_job = JobState()
        session.inference_job = JobState()
        session.inference_result = None


def _valid_config(hold_detector: str, route_classifier: str) -> bool:
    return (
        hold_detector in AVAILABLE_HOLD_DETECTORS
        and route_classifier in AVAILABLE_ROUTE_DISCRIMINATORS
    )


def _compat_config(value: str | None, available: list[str], fallback: str) -> str:
    return value if value in available else fallback


def _parse_legacy_points(all_points_x: str, all_points_y: str) -> list[Coordinate]:
    try:
        xs = json.loads(all_points_x)
        ys = json.loads(all_points_y)
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
    try:
        return [Coordinate(x=x, y=y) for x, y in zip(xs, ys, strict=True)]
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="coordinates must be normalized numbers"
        ) from exc


def _finish_segment_job(
    session: SessionState,
    generation: int,
    coordinates: list[Coordinate],
) -> None:
    try:
        result = detect_segments(coordinates)
    except Exception as exc:
        with session.lock:
            if generation == session.generation:
                session.segment_job = JobState("failed", model_error(exc))
        return

    with session.lock:
        if generation != session.generation:
            return
        session.segments.update({segment.segment_id: segment for segment in result})
        session.segment_job = JobState()


def _finish_augmentation_job(
    session: SessionState,
    generation: int,
    request: AugmentWorkingImageRequest,
) -> None:
    try:
        with session.lock:
            image = (
                session.working_image.copy()
                if session.working_image is not None
                else None
            )
            segments = list(session.segments.values())
        if image is not None:
            image = augment_image(image, request, segments)
    except Exception as exc:
        with session.lock:
            if generation == session.generation:
                session.image_job = JobState("failed", model_error(exc))
        return

    with session.lock:
        if generation != session.generation:
            return
        if image is not None:
            session.working_image = image
        session.image_job = JobState()


def _finish_inference_job(
    session: SessionState,
    generation: int,
    image: Any,
    hold_detector: str,
    route_classifier: str,
) -> None:
    try:
        result = infer(image, hold_detector, route_classifier)
    except Exception as exc:
        with session.lock:
            if generation == session.generation:
                session.inference_job = JobState("failed", model_error(exc))
                session.inference_result = None
        return

    with session.lock:
        if generation != session.generation:
            return
        session.inference_result = result
        session.inference_job = JobState()


async def _read_image_request(request: Request) -> Any:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("image")
        if not isinstance(upload, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=422, detail="image file is required")
        try:
            return decode_image(await upload.read())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        payload = await request.json()
        value = payload["image"]
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=422, detail="request must contain an image"
        ) from exc
    if not isinstance(value, str):
        raise HTTPException(status_code=422, detail="image must be a data URL")
    try:
        return decode_data_url(value)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/pipeline/available_configs", response_model=AvailableConfigsResponse)
def available_configs() -> AvailableConfigsResponse:
    return AvailableConfigsResponse(
        hold_detector=AVAILABLE_HOLD_DETECTORS,
        route_classifier=AVAILABLE_ROUTE_DISCRIMINATORS,
    )


@router.get("/pipeline", response_model=PipelineConfig)
def get_pipeline(session: SessionState = Depends(get_session)) -> PipelineConfig:
    with session.lock:
        return PipelineConfig(
            hold_detector=session.hold_detector,
            route_classifier=session.route_classifier,
        )


@router.put("/pipeline", response_model=PipelineConfig)
def set_pipeline(
    body: PipelineConfig,
    session: SessionState = Depends(get_session),
) -> PipelineConfig:
    if not _valid_config(body.hold_detector, body.route_classifier):
        raise HTTPException(
            status_code=422,
            detail={
                "hold_detector": AVAILABLE_HOLD_DETECTORS,
                "route_classifier": AVAILABLE_ROUTE_DISCRIMINATORS,
            },
        )
    with session.lock:
        session.hold_detector = body.hold_detector
        session.route_classifier = body.route_classifier
    return body


@router.put("/image/working", status_code=200)
async def set_working_image(
    request: Request,
    session: SessionState = Depends(get_session),
) -> None:
    _set_working_image(session, await _read_image_request(request))


@router.get("/image/working", response_model=WorkingImageResponse)
def get_working_image(
    session: SessionState = Depends(get_session),
) -> WorkingImageResponse:
    with session.lock:
        return WorkingImageResponse(
            status=session.image_job.status,
            image=image_data_url(session.working_image)
            if session.working_image is not None
            else None,
            image_id=session.working_image_id,
            error=session.image_job.error,
        )


@router.post("/image/working/segment", status_code=202)
def start_segmentation(
    body: SegmentRequest,
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
) -> Response:
    with session.lock:
        if session.segment_job.status == "processing":
            raise _conflict("segmentation is already processing")
        if session.working_image is None:
            raise _conflict("set a working image before segmenting")
        generation = session.generation
        session.segment_job = JobState("processing")
    background_tasks.add_task(
        _finish_segment_job, session, generation, body.coordinates
    )
    return Response(status_code=202)


@router.get("/image/working/segment", response_model=SegmentStatusResponse)
def get_segmentation(
    session: SessionState = Depends(get_session),
) -> SegmentStatusResponse:
    with session.lock:
        return SegmentStatusResponse(
            status=session.segment_job.status,
            segments=list(session.segments.values()),
            error=session.segment_job.error,
        )


@router.delete("/image/working/segment/{segment_id}", status_code=204)
def delete_segment(
    segment_id: str, session: SessionState = Depends(get_session)
) -> None:
    with session.lock:
        session.segments.pop(segment_id, None)


@router.post("/image/working/augment", status_code=202)
def start_augmentation(
    body: AugmentWorkingImageRequest | list[SegmentAugmentation],
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
) -> Response:
    if isinstance(body, list):
        body = AugmentWorkingImageRequest(segments=body)
    with session.lock:
        if session.image_job.status == "processing":
            raise _conflict("augmentation is already processing")
        generation = session.generation
        session.image_job = JobState("processing")
    background_tasks.add_task(_finish_augmentation_job, session, generation, body)
    return Response(status_code=202)


@router.get("/pipeline/infer/working", response_model=InferWorkingResponse)
def get_inference(session: SessionState = Depends(get_session)) -> InferWorkingResponse:
    with session.lock:
        if session.inference_job.status == "failed":
            return InferWorkingResponse(
                status="failed", error=session.inference_job.error
            )
        if session.inference_job.status == "processing":
            return InferWorkingResponse(status="processing")
        return session.inference_result or InferWorkingResponse()


@router.post("/pipeline/infer/working", response_model=None)
async def start_inference(
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
    image: UploadFile | None = File(None),
    hold_detector: str | None = Form(None),
    route_discriminator: str | None = Form(None),
    route_classifier: str | None = Form(None),
    lighting_percent: float | None = Form(None),
    segments: str | None = Form(None),
) -> Response | InferWorkingResponse:
    """Start spec-compliant inference, with the old multipart call kept working."""
    if image is not None:
        try:
            uploaded = decode_image(await image.read())
            with session.lock:
                stored_segments = list(session.segments.values())
                session.working_image = uploaded
                session.working_image_id = session.working_image_id or new_image_id()
                session.generation += 1
                session.image_job = JobState()
                session.inference_job = JobState()
                session.inference_result = None
            if lighting_percent and not stored_segments:
                uploaded = augment_image(
                    uploaded,
                    AugmentWorkingImageRequest(lighting_percent=lighting_percent),
                    (),
                )
            selected_hold = _compat_config(
                hold_detector, AVAILABLE_HOLD_DETECTORS, "mock"
            )
            selected_route = _compat_config(
                route_classifier or route_discriminator,
                AVAILABLE_ROUTE_DISCRIMINATORS,
                "mock",
            )
            return infer(uploaded, selected_hold, selected_route)
        except (UnknownHoldDetectorError, UnknownRouteDiscriminatorError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail=model_error(exc)) from exc

    with session.lock:
        if session.inference_job.status == "processing":
            raise _conflict("inference is already processing")
        if session.working_image is None:
            raise _conflict("set a working image before inference")
        generation = session.generation
        working_image = session.working_image.copy()
        selected_hold = session.hold_detector
        selected_route = session.route_classifier
        session.inference_job = JobState("processing")
    background_tasks.add_task(
        _finish_inference_job,
        session,
        generation,
        working_image,
        selected_hold,
        selected_route,
    )
    return Response(status_code=202)


# Compatibility route for the previous stateless frontend client.
@router.post("/image/working/segments", response_model=DetectSegmentsResponse)
async def detect_segments_legacy(
    image: UploadFile = File(...),
    all_points_x: str = Form(...),
    all_points_y: str = Form(...),
    session: SessionState = Depends(get_session),
) -> DetectSegmentsResponse:
    coordinates = _parse_legacy_points(all_points_x, all_points_y)
    try:
        uploaded = decode_image(await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _set_working_image(session, uploaded)
    result = detect_segments(coordinates)
    with session.lock:
        session.segments.update({segment.segment_id: segment for segment in result})
        session.segment_job = JobState()
    return DetectSegmentsResponse(segments=result)
