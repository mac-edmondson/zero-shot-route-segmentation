from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from .schemas import Coordinate, DetectSegmentsResponse, SegmentResult
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
