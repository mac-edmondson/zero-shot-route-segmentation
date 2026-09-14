import { API_BASE_URL } from "./config";
import { request, requestBlob } from "./client";
import type { RouteDetectionApiClient } from "./contract";
import type { AvailableConfigs, Coordinate, InferenceResult, JobStatus, Segment, WorkingImage } from "./types";
import { ApiError } from "./types";

/** One entry returned by Nginx's JSON autoindex for the gallery directory. */
interface GalleryDirectoryEntry {
  name: string;
  type: "file" | "directory";
}

const GALLERY_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"];

function galleryImageUrl(category: string, name: string): string {
  return `${API_BASE_URL}/images/${encodeURIComponent(category)}/${encodeURIComponent(name)}`;
}

/** Wire shape returned by `GET /image/working` (src/backend/schemas.py: WorkingImageResponse). */
interface WorkingImageResponse {
  status: JobStatus;
  image: string | null;
  image_id: string | null;
  error: string | null;
}

function mapWorkingImage(raw: WorkingImageResponse): WorkingImage {
  return { status: raw.status, imageId: raw.image_id, image: raw.image, error: raw.error };
}

/** Wire shape returned by `GET /image/working/segment` (src/backend/schemas.py: SegmentStatusResponse). */
interface SegmentStatusResponse {
  status: JobStatus;
  segments: {
    segment_id: string;
    polygon: { points: Coordinate[] };
  }[];
  error: string | null;
}

/** Wire shape returned by `GET /pipeline/available_configs` (src/backend/schemas.py). */
interface AvailableConfigsResponse {
  hold_detector: string[];
  route_classifier: string[];
}

/** Wire shape returned by `GET /pipeline/infer/working` (src/backend/schemas.py: InferWorkingResponse). */
interface InferWorkingResponse {
  status: JobStatus;
  routes: {
    route_id: number;
    holds: {
      centroid: Coordinate;
      polygon: { points: Coordinate[] };
    }[];
  }[];
  inference_metrics: Record<string, number>;
  error: string | null;
}

const POLL_INTERVAL_MS = 500;
const POLL_TIMEOUT_MS = 60_000;
/** inferWorkingPipeline's own polling interval, below -- inference tends to
 * run noticeably longer than segmentation/augmentation, so checking every
 * 500ms like those two was mostly wasted requests. Segmentation/augmentation
 * keep the faster 500ms (still POLL_INTERVAL_MS's default below) since they
 * usually settle quickly and a snappier check there is worth it. */
const INFERENCE_POLL_INTERVAL_MS = 3_000;

/**
 * Polls an asynchronous backend job endpoint on an interval until the
 * status changes from "processing" to completed/failed or times out.
 */
async function pollUntilSettled<T extends { status: JobStatus }>(
  fetchStatus: () => Promise<T>,
  intervalMs: number = POLL_INTERVAL_MS,
): Promise<T> {
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  for (;;) {
    const result = await fetchStatus();
    if (result.status !== "processing") return result;
    if (Date.now() >= deadline) {
      throw new ApiError("Timed out waiting for the backend to finish");
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
}

/**
 * Real REST implementation of {@link RouteDetectionApiClient}, matching the
 * endpoint sketch in docs/diagrams/spec_rest_api.drawio.svg. Field names on
 * that diagram were legible but not fully specified (see
 * docs/spec/pipeline/interfaces/dashboard-backend.md); adjust request/response
 * shapes here once the real backend settles them.
 */
export const restApiClient: RouteDetectionApiClient = {
  listImages(signal) {
    return request("/images", { signal });
  },

  uploadImage(file) {
    const form = new FormData();
    form.append("image", file);
    return request("/images", { method: "POST", body: form });
  },

  getImage(id, signal) {
    return request(`/image/${id}`, { signal });
  },

  async getWorkingImage(signal) {
    const raw = await request<WorkingImageResponse>("/image/working", { signal });
    return mapWorkingImage(raw);
  },

  async setWorkingImage(file, options) {
    const form = new FormData();
    form.append("image", file);
    const query = options?.keepSegments ? "?keep_segments=true" : "";
    await request(`/image/working${query}`, { method: "PUT", body: form });
  },

  async detectWorkingSegments(coordinates) {
    await request("/image/working/segment", {
      method: "POST",
      body: { coordinates: coordinates.map(({ x, y }) => ({ x, y })) },
    });

    const status = await pollUntilSettled(() =>
      request<SegmentStatusResponse>("/image/working/segment"),
    );
    if (status.status === "failed") {
      throw new ApiError(status.error ?? "Segmentation failed");
    }

    // The backend never echoes back which point produced which segment, but
    // segments accumulate in submission order -- the last `coordinates.length`
    // entries are exactly the ones this call just created, still lined up
    // with the coordinates that produced them.
    const created = status.segments.slice(-coordinates.length);
    return created.map((segment, index) => {
      const point = coordinates[index];
      return {
        segmentId: segment.segment_id,
        coordinates: point ? [point] : [],
        polygon: segment.polygon.points,
      } satisfies Segment;
    });
  },

  deleteWorkingSegment(segmentId) {
    return request(`/image/working/segment/${segmentId}`, { method: "DELETE" });
  },

  async augmentWorkingImage(body) {
    await request("/image/working/augment", { method: "POST", body });

    const raw = await pollUntilSettled(() =>
      request<WorkingImageResponse>("/image/working"),
    );
    if (raw.status === "failed") {
      throw new ApiError(raw.error ?? "Augmentation failed");
    }
    return mapWorkingImage(raw);
  },

  async inferWorkingPipeline() {
    await request("/pipeline/infer/working", { method: "POST" });

    const raw = await pollUntilSettled(
      () => request<InferWorkingResponse>("/pipeline/infer/working"),
      INFERENCE_POLL_INTERVAL_MS,
    );
    if (raw.status === "failed") {
      throw new ApiError(raw.error ?? "Inference failed");
    }

    return {
      status: raw.status,
      routes: raw.routes.map((route) => ({
        routeId: route.route_id,
        holds: route.holds.map((hold) => ({
          centroid: hold.centroid,
          polygon: hold.polygon,
        })),
      })),
      inferenceMetrics: raw.inference_metrics,
      error: raw.error,
    } satisfies InferenceResult;
  },

  getPipeline(signal) {
    return request("/pipeline", { signal });
  },

  setPipeline(config) {
    return request("/pipeline", { method: "PUT", body: config });
  },

  async getAvailableConfigs(signal) {
    const raw = await request<AvailableConfigsResponse>("/pipeline/available_configs", {
      signal,
    });
    return {
      holdDetector: raw.hold_detector,
      routeClassifier: raw.route_classifier,
    } satisfies AvailableConfigs;
  },

  async listGalleryCategories(signal) {
    const entries = await request<GalleryDirectoryEntry[]>("/images/", { signal });
    return entries.filter((entry) => entry.type === "directory").map((entry) => entry.name);
  },

  async listGalleryImages(category, signal) {
    const entries = await request<GalleryDirectoryEntry[]>(
      `/images/${encodeURIComponent(category)}/`,
      { signal },
    );
    return entries
      .filter(
        (entry) =>
          entry.type === "file" &&
          GALLERY_IMAGE_EXTENSIONS.some((extension) => entry.name.toLowerCase().endsWith(extension)),
      )
      .map((entry) => ({
        name: entry.name,
        category,
        url: galleryImageUrl(category, entry.name),
      }));
  },

  fetchGalleryImage(category, name) {
    return requestBlob(
      `/images/${encodeURIComponent(category)}/${encodeURIComponent(name)}`,
    );
  },
};
