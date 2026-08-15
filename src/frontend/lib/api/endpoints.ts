import { request } from "./client";
import type { RouteDetectionApiClient } from "./contract";

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

  addWorkingSegment(coordinates) {
    return request("/image/working/segment", {
      method: "POST",
      body: { coordinates },
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
};
