/**
 * Base URL for the Dashboard Backend API. Override via
 * `NEXT_PUBLIC_API_BASE_URL` in `.env.local`.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * Configures whether to use the in-memory mock client.
 * Defaults to true for standalone UI development without backend services.
 * Set `NEXT_PUBLIC_USE_MOCK_API=false` to use the real backend.
 */
export const USE_MOCK_API =
  (process.env.NEXT_PUBLIC_USE_MOCK_API ?? "true") !== "false";

/**
 * Used only by the mock API client (lib/api/mocks/mockClient.ts) to render
 * gallery thumbnails without a real backend running. This is independent of
 * the real backend's own copy of this config (IMAGE_GALLERY_BASE_URL /
 * IMAGE_GALLERY_PATH in src/backend/.env) -- the browser never talks to the
 * gallery server directly in real mode either (see src/backend/gallery.py),
 * so this value only ever feeds an `<img src>` for the mock picker's tiles.
 */
export const MOCK_GALLERY_BASE_URL = process.env.NEXT_PUBLIC_MOCK_GALLERY_BASE_URL ?? "";
