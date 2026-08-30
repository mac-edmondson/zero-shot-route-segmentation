"""Asynchronous working-image segmentation endpoints."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Response
from PIL import Image as PILImage

from ..dependencies import get_session
from ..schemas import Coordinate, SegmentRequest, SegmentStatusResponse
from ..services.dashboard import detect_segments, model_error
from ..session_store import JobState, SessionState
from ._common import _conflict

logger = logging.getLogger(__name__)


router = APIRouter()


def _finish_segment_job(
    session: SessionState,
    generation: int,
    image_id: str | None,
    image: PILImage.Image,
    coordinates: list[Coordinate],
) -> None:
    """Store detected segments unless the image generation changed."""
    try:
        result = detect_segments(image, coordinates)
    except Exception as exc:
        logger.exception(
            "segmentation failed image_id=%s generation=%d", image_id, generation
        )
        with session.lock:
            if generation == session.generation:
                session.segment_job = JobState("failed", model_error(exc))
        return

    with session.lock:
        if generation != session.generation:
            return
        session.segments.update({segment.segment_id: segment for segment in result})
        session.segment_job = JobState()


@router.post("/image/working/segment", status_code=202)
def start_segmentation(
    body: SegmentRequest,
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
) -> Response:
    """Queue segmentation for requested points on the working image."""
    with session.lock:
        if session.segment_job.status == "processing":
            raise _conflict("segmentation is already processing")
        if session.working_image is None:
            raise _conflict("set a working image before segmenting")
        generation = session.generation
        image_id = session.working_image_id
        image = session.working_image.copy()
        session.segment_job = JobState("processing")
    background_tasks.add_task(
        _finish_segment_job, session, generation, image_id, image, body.coordinates
    )
    logger.info(
        "segmentation queued image_id=%s generation=%d point_count=%d",
        image_id,
        generation,
        len(body.coordinates),
    )
    return Response(status_code=202)


@router.get("/image/working/segment", response_model=SegmentStatusResponse)
def get_segmentation(
    session: SessionState = Depends(get_session),
) -> SegmentStatusResponse:
    """Return the current segmentation state and saved segments."""
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
    """Remove a segment from the current session."""
    with session.lock:
        session.segments.pop(segment_id, None)
