/**
 * Types mirroring the shared pipeline data models
 * (docs/spec/pipeline/interfaces/data-models.md, src/pipeline/interfaces/data_models.py)
 * and the sketched REST contract
 * (docs/spec/pipeline/interfaces/dashboard-backend.md, docs/diagrams/spec_rest_api.drawio.svg).
 *
 * The Python backend does not exist yet, and several fields on the whiteboard
 * sketch behind the REST diagram are ambiguous or marked TODO in the spec
 * docs. These types are a best-effort, provisional mapping -- expect to
 * adjust them once a real backend lands and the contract firms up.
 */

export interface Coordinate {
  x: number;
  y: number;
}

export interface Polygon {
  points: Coordinate[];
}

export interface RGBColor {
  r: number;
  g: number;
  b: number;
}

/** A detected/annotated climbing hold. */
export interface Hold {
  centroid: Coordinate;
  polygon: Polygon;
}

/** A group of holds that make up a single route. */
export interface Route {
  routeId: number;
  holds: Hold[];
}

/** Row shape returned by `GET /images`. */
export interface ImageSummary {
  id: string;
  title: string;
  category: string;
}

/** The image currently loaded into the working slot (`GET /image/working`). */
export interface WorkingImage {
  imageId: string;
  /** Data URL or backend-hosted URL for the image bytes. */
  image: string;
}

/**
 * A user-marked region on the working image -- "segment" in the REST spec,
 * conceptually a `Hold` in the shared data model. Until a real HoldDetector
 * exists, the frontend lets a user drop points manually so augmentations
 * have something to target.
 */
export interface Segment {
  segmentId: string;
  coordinates: Coordinate[];
}

/** Per-segment augmentation applied by `POST /image/working/augment`. */
export interface SegmentAugmentation {
  segmentId: string;
  chalkPercent: number;
}

export interface AugmentWorkingImageRequest {
  lightingPercent: number;
  segments: SegmentAugmentation[];
}

export interface AugmentWorkingImageResult {
  imageId: string;
  image: string;
}

/** Stable identifiers for swappable hold-detector implementations. */
export type HoldDetectorMethod =
  | "mask_cnn_hold_det"
  | "yolov8"
  | "sam3"
  | "ground_truth";

/** Stable identifiers for swappable route-classifier implementations. */
export type RouteClassifierMethod =
  | "color_only"
  | "color_spatial"
  | "dino_only"
  | "color_spatial_dino";

export interface PipelineConfig {
  holdDetector: HoldDetectorMethod;
  routeClassifier: RouteClassifierMethod;
}

export type InferenceStatus = "processing" | "completed";

export interface InferenceResult {
  status: InferenceStatus;
  /** One route list per input image; each route is a list of holds. */
  routes: Route[];
  inferenceMetrics: Record<string, number>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
