from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from typing import Literal

from PIL import Image

from .schemas import InferWorkingResponse, SegmentResult

JobStatus = Literal["processing", "completed", "failed"]


@dataclass
class JobState:
    status: JobStatus = "completed"
    error: str | None = None


@dataclass
class SessionState:
    """All mutable dashboard state belonging to one browser session."""

    working_image: Image.Image | None = None
    working_image_id: str | None = None
    segments: dict[str, SegmentResult] = field(default_factory=dict)
    segment_job: JobState = field(default_factory=JobState)
    image_job: JobState = field(default_factory=JobState)
    inference_job: JobState = field(default_factory=JobState)
    inference_result: InferWorkingResponse | None = None
    hold_detector: str = "mock"
    route_classifier: str = "mock"
    generation: int = 0
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)


sessions: dict[str, SessionState] = {}


class SessionStore:
    def __init__(self, values: dict[str, SessionState]) -> None:
        self._values = values
        self._lock = RLock()

    def get_or_create(self, session_id: str | None) -> tuple[str, SessionState, bool]:
        with self._lock:
            if session_id and session_id in self._values:
                return session_id, self._values[session_id], False

            from uuid import uuid4

            new_id = str(uuid4())
            state = SessionState()
            self._values[new_id] = state
            return new_id, state, True


session_store = SessionStore(sessions)
