import { MOCK_GALLERY_BASE_URL } from "../config";
import type { RouteDetectionApiClient } from "../contract";
import type {
  AvailableConfigs,
  Coordinate,
  GalleryImage,
  ImageSummary,
  InferenceResult,
  PipelineConfig,
  Segment,
  WorkingImage,
} from "../types";
import { ApiError } from "../types";

/**
 * Same randomized mock-polygon approach as the real backend's mock
 * (src/backend/services/mock_segmentation.py) -- a small irregular blob
 * around the clicked point, randomized per point rather than one fixed
 * shape stamped everywhere -- so mock and real (mock-backed) modes look the
 * same.
 */
const MOCK_POLYGON_MIN_SIDES = 6;
const MOCK_POLYGON_MAX_SIDES = 10;
// Normalized units -- roughly a hold-sized blob at typical image scale.
const MOCK_POLYGON_MIN_RADIUS = 0.015;
const MOCK_POLYGON_MAX_RADIUS = 0.035;
// Per-vertex radius jitter, as a fraction of that polygon's base radius --
// keeps vertices irregular/organic rather than a perfect regular polygon.
const MOCK_POLYGON_VERTEX_JITTER = 0.3;

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function mockPolygonAround(point: Coordinate): Coordinate[] {
  const sides = Math.round(randomBetween(MOCK_POLYGON_MIN_SIDES, MOCK_POLYGON_MAX_SIDES));
  const baseRadius = randomBetween(MOCK_POLYGON_MIN_RADIUS, MOCK_POLYGON_MAX_RADIUS);
  return Array.from({ length: sides }, (_, i) => {
    const angle = (2 * Math.PI * i) / sides;
    const radius =
      baseRadius * randomBetween(1 - MOCK_POLYGON_VERTEX_JITTER, 1 + MOCK_POLYGON_VERTEX_JITTER);
    return {
      x: point.x + radius * Math.cos(angle),
      y: point.y + radius * Math.sin(angle),
    };
  });
}

/**
 * In-memory stand-in for the Dashboard Backend
 * (docs/spec/pipeline/interfaces/dashboard-backend.md) so the frontend is
 * fully clickable before the Python backend exists. Swap this out for
 * {@link restApiClient} (see ../endpoints.ts) by setting
 * `NEXT_PUBLIC_USE_MOCK_API=false`.
 */

const MOCK_LATENCY_MS = 220;

function delay<T>(value: T): Promise<T> {
  return new Promise((resolve) => setTimeout(() => resolve(value), MOCK_LATENCY_MS));
}

function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function createId(prefix: string): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `${prefix}_${random}`;
}

interface StoredImage extends ImageSummary {
  dataUrl: string;
}

// A small, fixed slice of the real gallery's known category/naming scheme
// (bh/0000.jpg, bh/0001.jpg, ...) purely so the mock picker's tiles show
// real photos via <img src> (safe -- no CORS involved in display). The real
// backend (src/backend/gallery.py) lists the *actual* directories/files
// instead of hardcoding names like this.
const MOCK_GALLERY_CATEGORIES = ["bh", "bh-phone", "model", "sm"];

function mockGalleryImages(category: string): GalleryImage[] {
  if (!MOCK_GALLERY_BASE_URL) return [];
  return Array.from({ length: 12 }, (_, i) => {
    const name = `${String(i).padStart(4, "0")}.jpg`;
    return { name, category, url: `${MOCK_GALLERY_BASE_URL}/${category}/${name}` };
  });
}

/**
 * The mock client can't actually fetch the gallery server's bytes
 * client-side -- it sends no CORS headers, so a browser fetch() is blocked
 * regardless of mock/real API mode. Real mode instead proxies through the
 * backend (src/backend/gallery.py); the mock substitutes a synthesized
 * placeholder image so "select a gallery image" still works end to end
 * without a backend running.
 */
function placeholderGalleryBlob(name: string): Promise<Blob> {
  const canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 480;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.fillStyle = "#20242c";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#ff7a45";
    ctx.font = "28px sans-serif";
    ctx.fillText("Mock gallery image", 32, 56);
    ctx.fillStyle = "#a9adb6";
    ctx.font = "18px sans-serif";
    ctx.fillText(name, 32, 88);
  }
  return new Promise((resolve) => {
    canvas.toBlob((blob) => resolve(blob ?? new Blob()), "image/png");
  });
}

const images = new Map<string, StoredImage>();
const segments = new Map<string, Segment>();
let workingImageId: string | null = null;

let pipelineConfig: PipelineConfig = {
  holdDetector: "sam3",
  routeClassifier: "color_spatial_dino",
};

// Same labels the real backend's hold_detector_factory/
// route_discriminator_factory registries resolve method names through
// (src/pipeline/*/*_factory.py) -- kept as a fixed mock list here rather
// than trying to fabricate a registry, same "small stand-in" treatment as
// MOCK_GALLERY_CATEGORIES above.
const MOCK_AVAILABLE_CONFIGS: AvailableConfigs = {
  holdDetector: ["Color-only", "DINO-only", "Combined"],
  routeClassifier: ["Color-only", "Color + Spatial", "Combined"],
};

export const mockApiClient: RouteDetectionApiClient = {
  listImages() {
    return delay(Array.from(images.values()).map(({ id, title, category }) => ({ id, title, category })));
  },

  async uploadImage(file) {
    const dataUrl = await fileToDataUrl(file);
    const summary: StoredImage = {
      id: createId("img"),
      title: file instanceof File ? file.name : "captured-frame",
      category: "wall",
      dataUrl,
    };
    images.set(summary.id, summary);
    return delay({ id: summary.id, title: summary.title, category: summary.category });
  },

  getImage(id) {
    const image = images.get(id);
    if (!image) {
      return Promise.reject(new ApiError(`No image with id "${id}"`, 404));
    }
    return delay({ id: image.id, title: image.title, category: image.category });
  },

  getWorkingImage() {
    if (!workingImageId) {
      return Promise.reject(new ApiError("No working image set", 404));
    }
    const image = images.get(workingImageId);
    if (!image) {
      return Promise.reject(new ApiError("Working image no longer exists", 404));
    }
    const result: WorkingImage = {
      status: "completed",
      imageId: image.id,
      image: image.dataUrl,
      error: null,
    };
    return delay(result);
  },

  async setWorkingImage(file, options) {
    const dataUrl = await fileToDataUrl(file);
    const summary: StoredImage = {
      id: createId("img"),
      title: file instanceof File ? file.name : "captured-frame",
      category: "wall",
      dataUrl,
    };
    images.set(summary.id, summary);
    workingImageId = summary.id;
    // Mirrors the real backend's keep_segments query param (see
    // src/backend/routes/working_image.py) -- a fresh upload still clears
    // segments, a Back to Augment reset doesn't.
    if (!options?.keepSegments) {
      segments.clear();
    }
    await delay(undefined);
  },

  detectWorkingSegments(coordinates) {
    if (!workingImageId) {
      return Promise.reject(new ApiError("No working image set", 409));
    }
    const created = coordinates.map((point): Segment => {
      const segment: Segment = {
        segmentId: createId("seg"),
        coordinates: [point],
        polygon: mockPolygonAround(point),
      };
      segments.set(segment.segmentId, segment);
      return segment;
    });
    return delay(created);
  },

  deleteWorkingSegment(segmentId) {
    segments.delete(segmentId);
    return delay(undefined);
  },

  augmentWorkingImage() {
    if (!workingImageId) {
      return Promise.reject(new ApiError("No working image set", 409));
    }
    const image = images.get(workingImageId)!;
    // The mock has no real augmentation pipeline (chalk/lighting/color
    // rendering), so it just echoes the source image back. A real backend
    // would return the materialized, augmented image bytes here.
    const result: WorkingImage = {
      status: "completed",
      imageId: image.id,
      image: image.dataUrl,
      error: null,
    };
    return delay(result);
  },

  inferWorkingPipeline() {
    // The mock has no real pipeline to run (like augmentWorkingImage
    // above); this just fabricates routes out of whatever's already in
    // `segments`, using whatever config `setPipeline` last recorded (the
    // real backend resolves the same way -- via `PUT /pipeline`, not an
    // argument to this call).
    const result: InferenceResult = {
      status: "completed",
      routes: Array.from(segments.values()).map((segment, index) => ({
        routeId: index,
        holds: [
          {
            centroid: segment.coordinates[0] ?? { x: 0, y: 0 },
            polygon: { points: segment.polygon },
          },
        ],
      })),
      inferenceMetrics: {
        holdDetectorLatencyMs: MOCK_LATENCY_MS,
        routeClassifierLatencyMs: MOCK_LATENCY_MS,
        routeCount: segments.size,
      },
      error: null,
    };
    return delay(result);
  },

  getPipeline() {
    return delay(pipelineConfig);
  },

  setPipeline(config) {
    pipelineConfig = config;
    return delay(pipelineConfig);
  },

  getAvailableConfigs() {
    return delay(MOCK_AVAILABLE_CONFIGS);
  },

  listGalleryCategories() {
    return delay(MOCK_GALLERY_BASE_URL ? MOCK_GALLERY_CATEGORIES : []);
  },

  listGalleryImages(category) {
    return delay(mockGalleryImages(category));
  },

  fetchGalleryImage(category, name) {
    return placeholderGalleryBlob(`${category}/${name}`);
  },
};
