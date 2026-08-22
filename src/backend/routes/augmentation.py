"""Asynchronous working-image augmentation endpoint."""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Response

from ..dependencies import get_session
from ..schemas import AugmentWorkingImageRequest
from ..services.dashboard import augment_image, model_error
from ..session_store import JobState, SessionState
from ._common import _conflict

logger = logging.getLogger(__name__)


router = APIRouter()


def _finish_augmentation_job(
    session: SessionState,
    generation: int,
    image_id: str | None,
    request: AugmentWorkingImageRequest,
) -> None:
    """Store augmented pixels unless the image generation changed."""
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
        logger.exception(
            "augmentation failed image_id=%s generation=%d", image_id, generation
        )
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


@router.post("/image/working/augment", status_code=202)
def start_augmentation(
    body: AugmentWorkingImageRequest,
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
) -> Response:
    """Queue augmentation for the current working image."""
    with session.lock:
        if session.image_job.status == "processing":
            raise _conflict("augmentation is already processing")
        generation = session.generation
        image_id = session.working_image_id
        session.image_job = JobState("processing")
    background_tasks.add_task(
        _finish_augmentation_job, session, generation, image_id, body
    )
    logger.info("augmentation queued image_id=%s generation=%d", image_id, generation)
    return Response(status_code=202)
