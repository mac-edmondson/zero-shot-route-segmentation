import { request, requestBlob } from "./client";
import type { RouteDetectionApiClient } from "./contract";
import type { Coordinate, GalleryImage, Segment } from "./types";

/** Wire shape returned by `POST /image/working/segments` (src/backend/schemas.py). */
interface DetectSegmentsResponse {
  segments: {
    segment_id: string;
    polygon: { points: Coordinate[] };
  }[];
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

  inferWorkingPipeline(config) {
    return request("/pipeline/infer/working", { method: "POST", body: config });
  },

  getPipeline(signal) {
    return request("/pipeline", { signal });
  },

  setPipeline(config) {
    return request("/pipeline", { method: "PUT", body: config });
  },

  async listGalleryCategories(signal) {
    const { categories } = await request<{ categories: { name: string }[] }>(
      "/gallery/categories",
      { signal },
    );
    return categories.map((c) => c.name);
  },

  async listGalleryImages(category, signal) {
    const { images } = await request<{ images: GalleryImage[] }>(
      `/gallery/images?category=${encodeURIComponent(category)}`,
      { signal },
    );
    return images;
  },

  fetchGalleryImage(category, name) {
    return requestBlob(
      `/gallery/images/${encodeURIComponent(category)}/${encodeURIComponent(name)}`,
    );
  },
};
