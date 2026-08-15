/**
 * Base URL for the Dashboard Backend API
 * (docs/spec/pipeline/interfaces/dashboard-backend.md). Override via
 * `NEXT_PUBLIC_API_BASE_URL` in `.env.local` once a real backend exists.
 */
export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/**
 * The Python backend doesn't exist yet (see docs/spec/pipeline/README.md),
 * so default to an in-memory mock client so the UI is fully usable end to
 * end. Set `NEXT_PUBLIC_USE_MOCK_API=false` once `NEXT_PUBLIC_API_BASE_URL`
 * points at a real backend.
 */
export const USE_MOCK_API =
  (process.env.NEXT_PUBLIC_USE_MOCK_API ?? "true") !== "false";
