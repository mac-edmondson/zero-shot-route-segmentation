"""
Proxy for the external sample-image gallery server.

That server (base URL/path configured below via env vars -- never hardcoded)
sends no CORS headers, so a browser-side fetch() straight to it is blocked
outright, not just slow: the browser refuses to hand the response body to
our JS at all. Routing both the listing and the byte-fetch through our own
backend sidesteps that entirely, since server-to-server HTTP calls aren't
subject to CORS. Displaying a thumbnail is a different story -- a plain
<img src="https://...">  works cross-origin with no proxy needed, which is
why GalleryImage.url below points straight at the external server rather
than through us.
"""

from __future__ import annotations

import os
import re

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .schemas import GalleryImage, GalleryListResponse

router = APIRouter(prefix="/gallery")

# Read lazily (function, not module-level constants) so tests/local overrides
# via os.environ after import still take effect, and so a missing .env value
# fails at request time with a clear error rather than baking in a stale
# value at import time.
_DEFAULT_BASE_URL = "https://cvp.iamemacs.com"
_DEFAULT_PATH = "/api/images/bh"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _base_url() -> str:
    return os.environ.get("IMAGE_GALLERY_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _path() -> str:
    return "/" + os.environ.get("IMAGE_GALLERY_PATH", _DEFAULT_PATH).strip("/")


def _listing_url() -> str:
    return f"{_base_url()}{_path()}/"


def _image_url(name: str) -> str:
    return f"{_base_url()}{_path()}/{name}"


@router.get("/images", response_model=GalleryListResponse)
async def list_gallery_images() -> GalleryListResponse:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_listing_url(), timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Couldn't reach gallery server: {exc}"
            ) from exc

    entries = response.json()
    images = [
        GalleryImage(name=entry["name"], url=_image_url(entry["name"]))
        for entry in entries
        if entry.get("type") == "file"
    ]
    return GalleryListResponse(images=images)


@router.get("/images/{name}")
async def get_gallery_image(name: str) -> Response:
    if not _SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid image name")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_image_url(name), timeout=15.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Couldn't reach gallery server: {exc}"
            ) from exc

    content_type = response.headers.get("content-type", "application/octet-stream")
    return Response(content=response.content, media_type=content_type)
