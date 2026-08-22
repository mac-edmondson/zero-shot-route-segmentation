"""Pipeline configuration endpoints."""

from fastapi import APIRouter, Depends, HTTPException

# This fallback keeps the imports testable when pytest loads `backend` top-level.
try:
    from ...pipeline.hold_detector.hold_detector_factory import AVAILABLE_HOLD_DETECTORS
    from ...pipeline.route_discriminator.route_discriminator_factory import (
        AVAILABLE_ROUTE_DISCRIMINATORS,
    )
except ImportError:
    from pipeline.hold_detector.hold_detector_factory import AVAILABLE_HOLD_DETECTORS
    from pipeline.route_discriminator.route_discriminator_factory import (
        AVAILABLE_ROUTE_DISCRIMINATORS,
    )

from ..dependencies import get_session
from ..schemas import AvailableConfigsResponse, PipelineConfig
from ..session_store import SessionState

router = APIRouter()


def _valid_config(hold_detector: str, route_classifier: str) -> bool:
    """Report whether both requested implementations are available."""
    return (
        hold_detector in AVAILABLE_HOLD_DETECTORS
        and route_classifier in AVAILABLE_ROUTE_DISCRIMINATORS
    )


@router.get("/pipeline/available_configs", response_model=AvailableConfigsResponse)
def available_configs() -> AvailableConfigsResponse:
    """List selectable pipeline implementations."""
    return AvailableConfigsResponse(
        hold_detector=AVAILABLE_HOLD_DETECTORS,
        route_classifier=AVAILABLE_ROUTE_DISCRIMINATORS,
    )


@router.get("/pipeline", response_model=PipelineConfig)
def get_pipeline(session: SessionState = Depends(get_session)) -> PipelineConfig:
    """Return the session's selected pipeline."""
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
    """Validate and store the session's pipeline selection."""
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
