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

/**
 * Status of a backend job that runs in the background rather than blocking
 * the request that started it -- segmentation, augmentation, and inference
 * all follow this same start/poll shape (see `PUT|GET /image/working`,
 * `POST|GET /image/working/segment`, `POST /image/working/augment` +
 * `GET /image/working`, and `POST|GET /pipeline/infer/working`).
 */
export type JobStatus = "processing" | "completed" | "failed";

/** The image currently loaded into the working slot (`GET /image/working`). */
export interface WorkingImage {
  status: JobStatus;
  imageId: string | null;
  /** Data URL or backend-hosted URL for the image bytes. */
  image: string | null;
  error: string | null;
}

/**
 * A user-marked region on the working image -- "segment" in the REST spec,
 * conceptually a `Hold` in the shared data model. The user clicks a point,
 * the backend runs it through a segmentation model (currently a mock
 * stand-in for SAM3 -- see docs/spec/pipeline/interfaces/hold-detector.md)
 * and returns a polygon outline for it.
 */
export interface Segment {
  segmentId: string;
  /** The clicked point(s) that produced this segment. */
  coordinates: Coordinate[];
  /** Detected hold outline, normalized to [0, 1] of the image. */
  polygon: Coordinate[];
}

/** Per-segment augmentation applied by `POST /image/working/augment`. */
export interface SegmentAugmentation {
  segmentId: string;
  chalkPercent: number;
  color?: RGBColor;
}

export interface AugmentWorkingImageRequest {
  lightingPercent: number;
  segments: SegmentAugmentation[];
}

/**
 * Stable identifiers for swappable hold-detector implementations, per
 * docs/diagrams/spec_rest_api.drawio.svg's PUT /pipeline sketch. Not yet
 * wired to real distinct behavior server-side -- only "mock"
 * (src/pipeline/hold_detector/mock_hold_detector.py) does anything today,
 * so the real backend currently ignores this value's content (still sends
 * it, for forward compatibility once SAM3/yolov8/etc. land for real).
 * Left as plain `string` rather than a literal union since the actual set
 * of valid values is decided server-side (hold_detector_factory's own
 * registry, surfaced to the UI by {@link AvailableConfigs}/
 * `getAvailableConfigs` below) and can grow without a frontend deploy.
 */
export type HoldDetectorMethod = string;

/** Same situation as {@link HoldDetectorMethod}, for route-discriminator
 * implementations -- only "mock" (mock_route_discriminator.py) exists. */
export type RouteClassifierMethod = string;

export interface PipelineConfig {
  holdDetector: HoldDetectorMethod;
  routeClassifier: RouteClassifierMethod;
}

/**
 * Response of `GET /pipeline/available_configs` -- the hold-detector/
 * route-classifier implementation names the model-select dropdowns
 * (WallImageWorkspace) should offer, sourced from the backend's own
 * pipeline factory registries (src/pipeline/*\/*_factory.py) rather than a
 * hardcoded frontend list.
 */
export interface AvailableConfigs {
  holdDetector: HoldDetectorMethod[];
  routeClassifier: RouteClassifierMethod[];
}

export type InferenceStatus = JobStatus;

export interface InferenceResult {
  status: InferenceStatus;
  /** One route list per input image; each route is a list of holds. */
  routes: Route[];
  inferenceMetrics: Record<string, number>;
  error: string | null;
}

/**
 * One image available in the external sample-image gallery
 * (src/backend/gallery.py). `url` is the external server's direct URL --
 * safe for a plain `<img src>` (CORS only blocks JS from reading bytes, not
 * the browser from rendering an image) -- but actually fetching those bytes
 * must go through `fetchGalleryImage`/the backend proxy, since that server
 * sends no CORS headers.
 */
export interface GalleryImage {
  name: string;
  category: string;
  url: string;
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
