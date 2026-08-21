import { API_BASE_URL } from "./config";
import { ApiError } from "./types";

interface RequestOptions {
  method?: "GET" | "POST" | "PUT" | "DELETE";
  body?: unknown;
  signal?: AbortSignal;
}

/** Thin typed `fetch` wrapper shared by every real REST endpoint call. */
export async function request<T>(
  path: string,
  { method = "GET", body, signal }: RequestOptions = {},
): Promise<T> {
  const isFormData = typeof FormData !== "undefined" && body instanceof FormData;

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: body && !isFormData ? { "Content-Type": "application/json" } : undefined,
      body: isFormData ? (body as FormData) : body ? JSON.stringify(body) : undefined,
      credentials: "include",
      signal,
    });
  } catch {
    throw new ApiError(`Failed to reach backend at ${API_BASE_URL}${path}`);
  }

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new ApiError(
      `Request to ${path} failed with ${response.status}${detail ? `: ${detail}` : ""}`,
      response.status,
    );
  }

  if (response.status === 202 || response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

/** Like {@link request}, but for endpoints that return raw bytes (e.g. an
 * image) rather than JSON -- used for the gallery image proxy. */
export async function requestBlob(
  path: string,
  { signal }: { signal?: AbortSignal } = {},
): Promise<Blob> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      signal,
      credentials: "include",
    });
  } catch {
    throw new ApiError(`Failed to reach backend at ${API_BASE_URL}${path}`);
  }

  if (!response.ok) {
    throw new ApiError(`Request to ${path} failed with ${response.status}`, response.status);
  }

  return response.blob();
}
