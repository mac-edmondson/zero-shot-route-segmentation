from __future__ import annotations

import os

from fastapi import Request, Response

from .session_store import SessionState, session_store


def _secure_cookie() -> bool:
    return os.getenv("SESSION_COOKIE_SECURE", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def get_session(request: Request, response: Response) -> SessionState:
    session_id = request.cookies.get("session_id")
    session_id, session, created = session_store.get_or_create(session_id)
    if created:
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            samesite="lax",
            secure=_secure_cookie(),
            path="/",
        )
    return session
