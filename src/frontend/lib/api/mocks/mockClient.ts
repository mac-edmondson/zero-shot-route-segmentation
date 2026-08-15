import type { RouteDetectionApiClient } from "../contract";
import type {
  ImageSummary,
  InferenceResult,
  PipelineConfig,
  Segment,
  WorkingImage,
} from "../types";
import { ApiError } from "../types";

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

const images = new Map<string, StoredImage>();
const segments = new Map<string, Segment>();
let workingImageId: string | null = null;

let pipelineConfig: PipelineConfig = {
  holdDetector: "sam3",
  routeClassifier: "color_spatial_dino",
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
    return delay({ imageId: image.id, image: image.dataUrl });
  },

  setWorkingImage(imageId) {
    const image = images.get(imageId);
    if (!image) {
      return Promise.reject(new ApiError(`No image with id "${imageId}"`, 404));
    }
    workingImageId = imageId;
    segments.clear();
    const result: WorkingImage = { imageId: image.id, image: image.dataUrl };
    return delay(result);
  },

  addWorkingSegment(coordinates) {
    if (!workingImageId) {
      return Promise.reject(new ApiError("No working image set", 409));
    }
    const segment: Segment = { segmentId: createId("seg"), coordinates };
    segments.set(segment.segmentId, segment);
    return delay(segment);
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
    // The mock has no real augmentation pipeline (chalk/lighting rendering),
    // so it just echoes the source image back. A real backend would return
    // the materialized, augmented image bytes here.
    return delay({ imageId: image.id, image: image.dataUrl });
  },

  inferWorkingPipeline(config) {
    if (!workingImageId) {
      return Promise.reject(new ApiError("No working image set", 409));
    }
    // Record the effective config so a later `getPipeline()` reflects any
    // one-off override passed to this call, mirroring how a real backend
    // would persist "last used" pipeline settings.
    pipelineConfig = { ...pipelineConfig, ...config };
    const result: InferenceResult = {
      status: "completed",
      routes: Array.from(segments.values()).map((segment, index) => ({
        routeId: index,
        holds: [
          {
            centroid: segment.coordinates[0] ?? { x: 0, y: 0 },
            polygon: { points: segment.coordinates },
          },
        ],
      })),
      inferenceMetrics: {
        holdDetectorLatencyMs: MOCK_LATENCY_MS,
        routeClassifierLatencyMs: MOCK_LATENCY_MS,
        routeCount: segments.size,
      },
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
};
