import type {
  AugmentWorkingImageRequest,
  AvailableConfigs,
  Coordinate,
  GalleryImage,
  ImageSummary,
  InferenceResult,
  PipelineConfig,
  Segment,
  WorkingImage,
} from "./types";

/**
 * Everything the UI needs from the Dashboard Backend API
 * (docs/spec/pipeline/interfaces/dashboard-backend.md). Implemented once for
 * real HTTP calls (`./endpoints.ts`) and once as an in-memory mock
 * (`./mocks/mockClient.ts`) so components can be built and demoed before the
 * Python backend exists, then switched over with a single env var.
 */
export interface RouteDetectionApiClient {
  listImages(signal?: AbortSignal): Promise<ImageSummary[]>;
  uploadImage(file: File | Blob): Promise<ImageSummary>;
  getImage(id: string, signal?: AbortSignal): Promise<ImageSummary>;

  getWorkingImage(signal?: AbortSignal): Promise<WorkingImage>;
  /**
   * Uploads `file`'s bytes as the working image via `PUT /image/working`.
   * The backend session then holds onto it -- later steps (segmentation,
   * augmentation, inference) reference it implicitly instead of re-sending
   * the file each time.
   */
  setWorkingImage(file: File | Blob): Promise<void>;

  /**
   * Submits every clicked point to `POST /image/working/segment`, then
   * polls `GET /image/working/segment` until the backend (a mock stand-in
   * for SAM3) finishes segmenting -- or fails -- and returns one polygon
   * per point, in the same order as `coordinates`. Only the points are
   * sent; the working image itself was already uploaded via
   * `setWorkingImage`.
   */
  detectWorkingSegments(coordinates: Coordinate[]): Promise<Segment[]>;
  deleteWorkingSegment(segmentId: string): Promise<void>;

  /**
   * Starts augmentation via `POST /image/working/augment`, then polls
   * `GET /image/working` until the backend finishes re-rendering the
   * working image with the requested lighting/chalk/color changes -- or
   * fails.
   */
  augmentWorkingImage(request: AugmentWorkingImageRequest): Promise<WorkingImage>;

  /**
   * Recognition: starts the full hold-detection + route-discrimination
   * pipeline via `POST /pipeline/infer/working` -- no body, since the
   * backend session already holds the working image (as left by
   * `Finish Augment`) and the model selection set by `setPipeline` just
   * before this is called -- then polls `GET /pipeline/infer/working`
   * until it completes or fails.
   */
  inferWorkingPipeline(): Promise<InferenceResult>;

  getPipeline(signal?: AbortSignal): Promise<PipelineConfig>;
  setPipeline(config: PipelineConfig): Promise<PipelineConfig>;

  /**
   * Lists the hold-detector/route-classifier implementations the backend
   * actually supports (its own pipeline factory registries) -- the
   * model-select dropdowns in the Augment step (WallImageWorkspace) call
   * this instead of carrying a hardcoded option list, so a newly
   * registered method shows up there without a frontend deploy.
   */
  getAvailableConfigs(signal?: AbortSignal): Promise<AvailableConfigs>;

  /**
   * Lists the top-level categories in the external sample-image gallery
   * (e.g. "bh", "sm") -- shown first so the picker never has to load every
   * image across every category at once (some categories run to 1000+
   * images). Proxied through the backend (src/backend/gallery.py) since
   * that server sends no CORS headers and a direct browser fetch() to it
   * would be blocked.
   */
  listGalleryCategories(signal?: AbortSignal): Promise<string[]>;
  /** Lists images within one gallery category, chosen from {@link listGalleryCategories}. */
  listGalleryImages(category: string, signal?: AbortSignal): Promise<GalleryImage[]>;
  /** Fetches one gallery image's actual bytes (same CORS reason as above). */
  fetchGalleryImage(category: string, name: string): Promise<Blob>;
}
