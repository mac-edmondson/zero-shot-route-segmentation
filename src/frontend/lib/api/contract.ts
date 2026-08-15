import type {
  AugmentWorkingImageRequest,
  AugmentWorkingImageResult,
  Coordinate,
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

  addWorkingSegment(coordinates: Coordinate[]): Promise<Segment>;
  deleteWorkingSegment(segmentId: string): Promise<void>;

  augmentWorkingImage(
    request: AugmentWorkingImageRequest,
  ): Promise<AugmentWorkingImageResult>;

  inferWorkingPipeline(
    config?: Partial<PipelineConfig>,
  ): Promise<InferenceResult>;

  getPipeline(signal?: AbortSignal): Promise<PipelineConfig>;
  setPipeline(config: PipelineConfig): Promise<PipelineConfig>;
}
