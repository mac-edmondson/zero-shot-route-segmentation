import type {
  AugmentWorkingImageRequest,
  AugmentWorkingImageResult,
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
  setWorkingImage(imageId: string): Promise<WorkingImage>;

  /**
   * Batched hold segmentation: sends the working image plus every clicked
   * point in one request; the backend loops each point through a
   * segmentation model (currently a mock stand-in for SAM3) and returns one
   * polygon per point, in the same order as `coordinates`.
   */
  detectWorkingSegments(
    image: File | Blob,
    coordinates: Coordinate[],
  ): Promise<Segment[]>;
  deleteWorkingSegment(segmentId: string): Promise<void>;

  augmentWorkingImage(
    request: AugmentWorkingImageRequest,
  ): Promise<AugmentWorkingImageResult>;

  /**
   * Recognition: runs the full hold-detection + route-discrimination
   * pipeline and returns the detected routes. Carries the working image,
   * the augmentation state gathered by `Finish Augment`, and the model
   * selection made just before `Recognition` is pressed -- all in one call,
   * since the backend is stateless and never stores the working image
   * between requests (see detectWorkingSegments above), so there's nowhere
   * else to combine what those two steps gathered. The pipeline needs the
   * image and both selections together to run at all.
   */
  inferWorkingPipeline(
    image: File | Blob,
    augmentation: AugmentWorkingImageRequest,
    config: PipelineConfig,
  ): Promise<InferenceResult>;

  getPipeline(signal?: AbortSignal): Promise<PipelineConfig>;
  setPipeline(config: PipelineConfig): Promise<PipelineConfig>;

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
