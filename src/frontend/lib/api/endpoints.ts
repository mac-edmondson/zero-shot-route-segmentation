import { API_BASE_URL } from "./config";
import { request, requestBlob } from "./client";
import type { RouteDetectionApiClient } from "./contract";
import type { AvailableConfigs, Coordinate, InferenceResult, Segment } from "./types";

/** One entry returned by Nginx's JSON autoindex for the gallery directory. */
interface GalleryDirectoryEntry {
  name: string;
  type: "file" | "directory";
}

const GALLERY_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"];

function galleryImageUrl(category: string, name: string): string {
  return `${API_BASE_URL}/images/${encodeURIComponent(category)}/${encodeURIComponent(name)}`;
}

/** Wire shape returned by `POST /image/working/segments` (src/backend/schemas.py). */
interface DetectSegmentsResponse {
  segments: {
    segment_id: string;
    polygon: { points: Coordinate[] };
  }[];
}

/** Wire shape returned by `GET /pipeline/available_configs` (src/backend/schemas.py). */
interface AvailableConfigsResponse {
  hold_detector: string[];
  route_classifier: string[];
}

/** Wire shape returned by `POST /pipeline/infer/working` (src/backend/schemas.py). */
interface InferWorkingResponse {
  routes: {
    route_id: number;
    holds: {
      centroid: Coordinate;
      polygon: { points: Coordinate[] };
    }[];
  }[];
  inference_metrics: Record<string, number>;
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

  getWorkingImage(signal) {
    return request("/image/working", { signal });
  },

  setWorkingImage(imageId) {
    return request("/image/working", { method: "PUT", body: { imageId } });
  },

  async detectWorkingSegments(image, coordinates) {
    const form = new FormData();
    form.append("image", image);
    form.append("all_points_x", JSON.stringify(coordinates.map((c) => c.x)));
    form.append("all_points_y", JSON.stringify(coordinates.map((c) => c.y)));

    const { segments } = await request<DetectSegmentsResponse>(
      "/image/working/segments",
      { method: "POST", body: form },
    );

    // Backend returns one polygon per input point, in the same order --
    // zip back up with the coordinate that produced each one.
    return segments.map((segment, index) => {
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

  augmentWorkingImage(body) {
    return request("/image/working/augment", { method: "POST", body });
  },

  async inferWorkingPipeline(image, augmentation, config) {
    const form = new FormData();
    form.append("image", image);
    form.append("lighting_percent", String(augmentation.lightingPercent));
    form.append(
      "segments",
      JSON.stringify(
        augmentation.segments.map((segment) => ({
          segment_id: segment.segmentId,
          chalk_percent: segment.chalkPercent,
        })),
      ),
    );
    form.append("hold_detector", config.holdDetector);
    form.append("route_discriminator", config.routeClassifier);

    const raw = await request<InferWorkingResponse>("/pipeline/infer/working", {
      method: "POST",
      body: form,
    });

    return {
      status: "completed",
      routes: raw.routes.map((route) => ({
        routeId: route.route_id,
        holds: route.holds.map((hold) => ({
          centroid: hold.centroid,
          polygon: hold.polygon,
        })),
      })),
      inferenceMetrics: raw.inference_metrics,
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
