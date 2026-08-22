"""Asynchronous working-image inference endpoints."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, Response

from ..dependencies import get_session
from ..schemas import InferWorkingResponse
from ..services.dashboard import infer, model_error
from ..session_store import JobState, SessionState
from ._common import _conflict

logger = logging.getLogger(__name__)


router = APIRouter()


def _finish_inference_job(
    session: SessionState,
    generation: int,
    image_id: str | None,
    image: Any,
    hold_detector: str,
    route_classifier: str,
) -> None:
    """Store inference output unless the working image changed."""
    try:
        result = infer(image, hold_detector, route_classifier)
    except Exception as exc:
        logger.exception(
            "inference failed image_id=%s generation=%d hold_detector=%s route_classifier=%s",
            image_id,
            generation,
            hold_detector,
            route_classifier,
        )
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


@router.get("/pipeline/infer/working", response_model=InferWorkingResponse)
def get_inference(session: SessionState = Depends(get_session)) -> InferWorkingResponse:
    """Return the current inference status or completed result."""
    with session.lock:
        if session.inference_job.status == "failed":
            return InferWorkingResponse(
                status="failed", error=session.inference_job.error
            )
        if session.inference_job.status == "processing":
            return InferWorkingResponse(status="processing")
        return session.inference_result or InferWorkingResponse()


@router.post("/pipeline/infer/working", response_model=None)
def start_inference(
    background_tasks: BackgroundTasks,
    session: SessionState = Depends(get_session),
) -> Response:
    """Queue inference for the current session working image."""
    with session.lock:
        if session.inference_job.status == "processing":
            raise _conflict("inference is already processing")
        if session.working_image is None:
            raise _conflict("set a working image before inference")
        generation = session.generation
        image_id = session.working_image_id
        working_image = session.working_image.copy()
        selected_hold = session.hold_detector
        selected_route = session.route_classifier
        session.inference_job = JobState("processing")
    background_tasks.add_task(
        _finish_inference_job,
        session,
        generation,
        image_id,
        working_image,
        selected_hold,
        selected_route,
    )
    logger.info(
        "inference queued image_id=%s generation=%d hold_detector=%s route_classifier=%s",
        image_id,
        generation,
        selected_hold,
        selected_route,
    )
    return Response(status_code=202)
