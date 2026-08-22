"""Working-image upload and retrieval endpoints."""

from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from ..dependencies import get_session
from ..schemas import WorkingImageResponse
from ..services.dashboard import decode_image, image_data_url, new_image_id
from ..session_store import JobState, SessionState

router = APIRouter()


def _set_working_image(session: SessionState, image: Any) -> None:
    """Replace a session image and clear image-derived state."""
    with session.lock:
        session.working_image = image
        session.working_image_id = new_image_id()
        session.generation += 1
        session.segments.clear()
        session.segment_job = JobState()
        session.image_job = JobState()
        session.inference_job = JobState()
        session.inference_result = None


@router.put("/image/working", status_code=200)
async def set_working_image(
    image: UploadFile = File(...),
    session: SessionState = Depends(get_session),
) -> None:
    """Decode and store a multipart working-image upload."""
    try:
        uploaded = decode_image(await image.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _set_working_image(session, uploaded)


@router.get("/image/working", response_model=WorkingImageResponse)
def get_working_image(
    session: SessionState = Depends(get_session),
) -> WorkingImageResponse:
    """Return working-image status and display data."""
    with session.lock:
        return WorkingImageResponse(
            status=session.image_job.status,
            image=image_data_url(session.working_image)
            if session.working_image is not None
            else None,
            image_id=session.working_image_id,
            error=session.image_job.error,
        )
