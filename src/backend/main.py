from __future__ import annotations

import os

from dotenv import load_dotenv

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .routes import router  # noqa: E402

app = FastAPI(title="ROUTNet Dashboard Backend")

# Frontend dev origin, overridable via env rather than hardcoded (same "put
# it in .env, not in code" preference as the frontend's own API_BASE_URL /
# image-server base URL).
_frontend_origin = os.environ.get("FRONTEND_ORIGIN", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[_frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
