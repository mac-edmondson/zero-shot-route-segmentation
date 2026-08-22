"""Shared route helpers."""

from fastapi import HTTPException


def _conflict(message: str) -> HTTPException:
    """Create a conflict response for an invalid session transition."""
    return HTTPException(status_code=409, detail=message)
