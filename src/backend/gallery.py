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

The server's root path (IMAGE_GALLERY_PATH) is itself a directory of
*categories* ("bh", "bh-phone", "model", "sm", ...), each holding hundreds to
thousands of images -- listing every image across every category in one shot
is what made the old flat picker heavy. So this proxies two listing levels:
categories first (/gallery/categories), then images within one category
(/gallery/images?category=...), picked by the user before anything image-
sized gets fetched.
"""

from __future__ import annotations

import os
import re

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from .schemas import (
    GalleryCategoriesResponse,
    GalleryCategory,
    GalleryImage,
    GalleryListResponse,
)

router = APIRouter(prefix="/gallery")

# Read lazily (function, not module-level constants) so tests/local overrides
# via os.environ after import still take effect, and so a missing .env value
# fails at request time with a clear error rather than baking in a stale
# value at import time.
_DEFAULT_BASE_URL = "https://cvp.iamemacs.com"
# The categories root -- NOT a specific category. See module docstring.
_DEFAULT_PATH = "/api/images"

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _base_url() -> str:
    return os.environ.get("IMAGE_GALLERY_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def _path() -> str:
    return "/" + os.environ.get("IMAGE_GALLERY_PATH", _DEFAULT_PATH).strip("/")


def _categories_url() -> str:
    return f"{_base_url()}{_path()}/"


def _category_listing_url(category: str) -> str:
    return f"{_base_url()}{_path()}/{category}/"


def _image_url(category: str, name: str) -> str:
    return f"{_base_url()}{_path()}/{category}/{name}"


async def _fetch_json(url: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Couldn't reach gallery server: {exc}"
            ) from exc
    return response.json()


@router.get("/categories", response_model=GalleryCategoriesResponse)
async def list_gallery_categories() -> GalleryCategoriesResponse:
    entries = await _fetch_json(_categories_url())
    categories = [
        GalleryCategory(name=entry["name"])
        for entry in entries
        if entry.get("type") == "directory"
    ]
    return GalleryCategoriesResponse(categories=categories)


@router.get("/images", response_model=GalleryListResponse)
async def list_gallery_images(
    category: str = Query(..., description="Category name from /gallery/categories"),
) -> GalleryListResponse:
    if not _SAFE_NAME.match(category):
        raise HTTPException(status_code=400, detail="Invalid category name")

    entries = await _fetch_json(_category_listing_url(category))
    images = [
        GalleryImage(name=entry["name"], category=category, url=_image_url(category, entry["name"]))
        for entry in entries
        if entry.get("type") == "file"
        # Some categories (e.g. "model") hold non-image artifacts alongside
        # or instead of photos -- skip those rather than handing the picker
        # a tile that can never render as an <img>.
        and entry["name"].lower().endswith(_IMAGE_EXTENSIONS)
    ]
    return GalleryListResponse(images=images)


@router.get("/images/{category}/{name}")
async def get_gallery_image(category: str, name: str) -> Response:
    if not _SAFE_NAME.match(category) or not _SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail="Invalid category or image name")

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(_image_url(category, name), timeout=15.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502, detail=f"Couldn't reach gallery server: {exc}"
            ) from exc

    content_type = response.headers.get("content-type", "application/octet-stream")
    return Response(content=response.content, media_type=content_type)
