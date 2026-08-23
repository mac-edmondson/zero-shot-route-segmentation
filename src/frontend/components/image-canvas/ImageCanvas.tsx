"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import type { Coordinate, Segment } from "@/lib/api";
import { Button } from "@/components/button/Button";
import { FlyingLogoLoader } from "@/components/flying-logo-loader/FlyingLogoLoader";
import styles from "./ImageCanvas.module.css";

/** Zoom range for Ctrl+/Ctrl- (see the zoom/pan state below) -- 1 is the
 * image's normal fit-to-box size, so it also doubles as "no zoom"/the
 * floor pan gets clamped to. */
const ZOOM_MIN = 1;
const ZOOM_MAX = 4;
const ZOOM_STEP = 0.25;

/** Minimum pointer travel (px) before a mousedown-drag on the image counts
 * as a pan rather than a click -- see handleImageWrapMouseDown. */
const PAN_DRAG_THRESHOLD = 4;

/** How much one wheel-event's deltaY moves the zoom multiplicatively (see
 * handleWheel) -- e.g. a typical mouse-wheel notch (deltaY around 100)
 * works out to roughly +/-20% zoom, while a trackpad's many small-delta
 * events during a single pinch/scroll gesture each nudge it just a little,
 * summing to a smooth continuous zoom rather than jumping in fixed steps
 * like the keyboard shortcut does. */
const ZOOM_WHEEL_SENSITIVITY = 0.0025;

/** Chalk fill's opacity at 100% chalkPercent -- kept short of fully opaque
 * so the hold's own outline/fill stays visible underneath even at max. */
const MAX_CHALK_OPACITY = 0.8;

/** Standard ray-casting point-in-polygon test. Scale-invariant -- used both
 * to find which segment a click landed in (normalized [0, 1] coordinates)
 * and, inside sampleSegmentColor below, to mask a polygon's own pixels out
 * of its bounding box (pixel coordinates). Either way the vertices and the
 * test point just need to share one coordinate space. */
function pointInPolygon(point: Coordinate, polygon: Coordinate[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;
    const intersects =
      yi > point.y !== yj > point.y && point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
}

/** The topmost segment (last drawn, so last in the array) whose polygon
 * contains a normalized [0, 1] point, or null if the click missed every
 * hold. */
function findSegmentAtPoint(segments: Segment[], point: Coordinate): Segment | null {
  for (let i = segments.length - 1; i >= 0; i--) {
    if (pointInPolygon(point, segments[i].polygon)) return segments[i];
  }
  return null;
}

function toHexColor(r: number, g: number, b: number): string {
  const channel = (value: number) => Math.round(value).toString(16).padStart(2, "0");
  return `#${channel(r)}${channel(g)}${channel(b)}`;
}

/** Caps how many pixels sampleSegmentColor actually walks -- a hold's own
 * bounding box is normally small, but this keeps a rare huge one (a wide,
 * sprawling polygon on a high-res photo) from blocking on a full-res scan. */
const MAX_COLOR_SAMPLES = 20_000;

/**
 * The eyedropper behind each segment card's "Choose color" (see
 * SegmentCard): not just the one pixel under the click, but the *average*
 * pixel color across the clicked segment's own polygon -- a single pixel
 * can land on a highlight, a shadow, chalk, or the hold's edge, none of
 * which read as "this hold's color" the way an area average does. Drawn
 * onto a throwaway canvas rather than sampled from the <img> directly since
 * that's the only way the DOM exposes pixel data; safe to read back since
 * imageSrc is always a data: URL (see loadImage in WallImageWorkspace),
 * never a cross-origin one that would taint the canvas.
 */
function sampleSegmentColor(img: HTMLImageElement, polygon: Coordinate[]): string | null {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(img, 0, 0);

    const polygonPx = polygon.map((p) => ({ x: p.x * canvas.width, y: p.y * canvas.height }));
    const minX = Math.max(0, Math.floor(Math.min(...polygonPx.map((p) => p.x))));
    const maxX = Math.min(canvas.width - 1, Math.ceil(Math.max(...polygonPx.map((p) => p.x))));
    const minY = Math.max(0, Math.floor(Math.min(...polygonPx.map((p) => p.y))));
    const maxY = Math.min(canvas.height - 1, Math.ceil(Math.max(...polygonPx.map((p) => p.y))));
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    if (boxWidth <= 0 || boxHeight <= 0) return null;

    const stride = Math.max(1, Math.ceil(Math.sqrt((boxWidth * boxHeight) / MAX_COLOR_SAMPLES)));
    const { data } = context.getImageData(minX, minY, boxWidth, boxHeight);

    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
    for (let y = 0; y < boxHeight; y += stride) {
      for (let x = 0; x < boxWidth; x += stride) {
        if (!pointInPolygon({ x: minX + x, y: minY + y }, polygonPx)) continue;
        const i = (y * boxWidth + x) * 4;
        r += data[i];
        g += data[i + 1];
        b += data[i + 2];
        count++;
      }
    }
    if (count === 0) return null;
    return toHexColor(r / count, g / count, b / count);
  } catch {
    // SecurityError or similar -- fail quietly, the pick just doesn't land.
    return null;
  }
}

/** Fallback for sampleSegmentColor above, when a color pick's click misses
 * every polygon (there's no segment to average over) -- just the one pixel
 * under the cursor instead. Same throwaway-canvas approach and the same
 * data: URL safety note applies. */
function sampleColorAt(img: HTMLImageElement, coordinate: Coordinate): string | null {
  try {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.drawImage(img, 0, 0);
    const x = Math.min(canvas.width - 1, Math.max(0, Math.floor(coordinate.x * canvas.width)));
    const y = Math.min(canvas.height - 1, Math.max(0, Math.floor(coordinate.y * canvas.height)));
    const [r, g, b] = context.getImageData(x, y, 1, 1).data;
    return toHexColor(r, g, b);
  } catch {
    return null;
  }
}

interface ImageCanvasProps {
  imageSrc: string | null;
  /** 0-100. Previewed live on the image via a brightness filter; 50 is
   * unmodified (the original image), below 50 darkens it and above 50
   * lightens it. */
  lightingPercent: number;
  segments: Segment[];
  /** 0-100 per segmentId. Previewed live as a white fill clipped to that
   * segment's own polygon -- same "live, client-side, no backend round
   * trip" treatment as lightingPercent above, just scoped per-hold instead
   * of image-wide. */
  chalkBySegmentId: Record<string, number>;
  /** Hex color sampled from the image for each segment, if any -- fills
   * that segment's own polygon (see .polygonShapeColored) instead of the
   * shared accent color. */
  colorBySegmentId?: Record<string, string>;
  /** True while a segment card's "Choose color" is waiting on a click here
   * to sample from -- see sampleSegmentColor/sampleColorAt above.
   * Suppresses the normal click-to-add-point behavior while active. */
  pickingColor?: boolean;
  onPickColor?: (color: string) => void;
  /** Points clicked but not yet submitted for detection. */
  pendingPoints?: Coordinate[];
  webcamActive?: boolean;
  loading?: boolean;
  /** The real ROUTNet logo up in the header -- passed through to
   * FlyingLogoLoader (see the `loading` overlay below) as the "from"/"to"
   * rect its clone flies between. Required whenever `loading` can ever be
   * true, which in this app's one real usage (WallImageWorkspace) is
   * always. */
  logoRef: RefObject<HTMLElement | null>;
  onCaptureFrame?: (blob: Blob) => void;
  onWebcamError?: (message: string) => void;
  /** Coordinates are normalized to [0, 1] of the displayed image. */
  onAddSegmentPoint?: (coordinate: Coordinate) => void;
  /**
   * False for a read-only display -- the recognition page's carried-over
   * augmented image, past the point of editing it further. Disables
   * click-to-add-point and hides the numbered segment markers/pending-point
   * dots. The lighting filter/chalk overlay props still work whenever
   * they're passed a non-neutral value -- WallImageWorkspace just stops
   * doing that once Finish Augment has swapped `imageSrc` for the real
   * augmented image from the backend (see handleFinishAugment), so those
   * effects aren't double-applied on top of pixels that already have them
   * baked in. Defaults to true.
   */
  interactive?: boolean;
  /**
   * False hides the accent hold-outline polygon, leaving only the chalk
   * fill on top of the plain image -- used by the recognition page so the
   * "here's where you clicked" outline doesn't linger once editing is
   * done, while chalk (an actual augmentation the user applied) still
   * shows. Defaults to true.
   */
  showHoldOutline?: boolean;
  /**
   * Extra SVG content stacked on top of the polygon/chalk overlay, still
   * inside the same normalized [0, 1] coordinate box -- e.g. the
   * recognition page's animated route-hold highlight. Kept as a generic
   * slot rather than a route-specific prop so this component doesn't need
   * to know what "routes" are.
   */
  overlay?: React.ReactNode;
}

/**
 * The big rounded preview box from the ROUTNet wireframe
 * (Project stuff/UI_page_1.png): shows the working image, a live webcam
 * feed, or an empty state, and lets the user click the image to drop a
 * segment point (a stand-in for real hold detection -- see
 * docs/spec/pipeline/interfaces/hold-detector.md, not yet implemented).
 */
export function ImageCanvas({
  imageSrc,
  lightingPercent,
  segments,
  chalkBySegmentId,
  colorBySegmentId = {},
  pickingColor = false,
  onPickColor,
  pendingPoints = [],
  webcamActive = false,
  loading = false,
  logoRef,
  onCaptureFrame = () => {},
  onWebcamError = () => {},
  onAddSegmentPoint,
  interactive = true,
  showHoldOutline = true,
  overlay,
}: ImageCanvasProps) {
  // -1..1, matching the backend's own LightingAugmentationParams.intensity
  // (see src/backend/services/dashboard.py: request.lighting_percent / 100,
  // where lighting_percent is already WallImageWorkspace's
  // (lighting - 50) * 2). lightingPercent itself stays on this component's
  // own 0-100/50-neutral scale -- this is just that same value re-expressed
  // as a fraction, for the lighting overlay below to use directly as an
  // alpha.
  const lightingIntensity = (lightingPercent - 50) / 50;

  // .imageWrap has no height of its own -- it shrink-wraps to the image
  // (see the CSS) so overlays/markers positioned against it stay pinned to
  // the actual visible image, not empty letterboxed space around it. But
  // that means the <img>'s own `max-height: 100%` has nothing definite to
  // resolve against (a descendant's percentage height doesn't resolve
  // against an ancestor whose own height is itself auto/content-driven --
  // classic CSS circular dependency), so the browser silently drops that
  // constraint and sizes the image from width alone, ignoring available
  // vertical space entirely. Invisible whenever width happens to be the
  // tighter bound (the common case), but the moment height is the tighter
  // one -- e.g. this box at its full width, before any holds exist, see
  // showFullWidthCanvas in WallImageWorkspace -- the image overflows its
  // own wrapper and .box's overflow:hidden silently crops it top/bottom,
  // looking exactly like the old cover behavior despite object-fit:
  // contain. Recording the image's real aspect ratio once it loads and
  // setting it explicitly (below) sidesteps the whole percentage-
  // resolution problem: aspect-ratio + max-width + max-height together let
  // the browser solve the box's size directly, with no child percentage
  // involved.
  const [imageAspectRatio, setImageAspectRatio] = useState<number | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streamReady, setStreamReady] = useState(false);

  // --- Zoom/pan -------------------------------------------------------
  // Scales/translates .imageWrap as a whole (see the CSS) -- the image,
  // polygon overlay, chalk fill, and markers are all siblings inside it,
  // so transforming the wrapper keeps every one of them in registration
  // with the image at any zoom level without recomputing coordinates.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const imageWrapRef = useRef<HTMLDivElement>(null);
  // Whether the pointer is currently over the box -- gates the Ctrl+/Ctrl-
  // zoom shortcut below so it doesn't fight the browser's own page-zoom
  // when the user is scrolled elsewhere on the page.
  const hoveringRef = useRef(false);
  // Set on a real drag (see handleImageWrapMouseDown) so the click that
  // fires on mouseup doesn't also get read as an add-point/color-pick
  // click.
  const suppressClickRef = useRef(false);

  // Fresh image -- start over at no zoom/pan rather than carrying over
  // whatever the previous image was left at. Adjusted during render
  // (React's documented way to reset state on a prop change) rather than
  // in an effect, so it takes effect before that first render paints
  // instead of flashing the old zoom/pan for a frame first.
  const [prevImageSrc, setPrevImageSrc] = useState(imageSrc);
  if (imageSrc !== prevImageSrc) {
    setPrevImageSrc(imageSrc);
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }

  /** Keeps pan inside the bounds a given zoom level allows -- shared by the
   * keydown handler below and handleImageWrapMouseDown's drag, both of
   * which need it clamped against imageWrapRef's *unscaled* layout box
   * (offsetWidth/Height ignore the wrapper's own CSS transform, so this
   * reads the same regardless of the zoom currently applied). */
  function clampPan(pan: { x: number; y: number }, forZoom: number) {
    if (forZoom <= 1) return { x: 0, y: 0 };
    const wrap = imageWrapRef.current;
    if (!wrap) return pan;
    const maxX = ((forZoom - 1) * wrap.offsetWidth) / 2;
    const maxY = ((forZoom - 1) * wrap.offsetHeight) / 2;
    return { x: Math.min(maxX, Math.max(-maxX, pan.x)), y: Math.min(maxY, Math.max(-maxY, pan.y)) };
  }

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (!hoveringRef.current) return;
      if (!(event.ctrlKey || event.metaKey)) return;
      if (event.key !== "+" && event.key !== "=" && event.key !== "-" && event.key !== "_") return;
      event.preventDefault();
      const zoomingOut = event.key === "-" || event.key === "_";
      const nextZoom = zoomingOut
        ? Math.max(ZOOM_MIN, +(zoom - ZOOM_STEP).toFixed(2))
        : Math.min(ZOOM_MAX, +(zoom + ZOOM_STEP).toFixed(2));
      setZoom(nextZoom);
      setPan((current) => clampPan(current, nextZoom));
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
    // Re-bound on every zoom change so the handler always steps from the
    // current zoom rather than a stale closure over the value from when
    // the listener was first attached.
  }, [zoom]);

  /** Drag-to-pan, only meaningful once zoomed in. Distinguishes a pan from
   * a plain click by movement distance -- short of PAN_DRAG_THRESHOLD it's
   * left alone so a click still adds a point/samples a color as normal;
   * past it, suppressClickRef swallows the click that mouseup triggers. */
  function handleImageWrapMouseDown(event: React.MouseEvent) {
    if (zoom <= 1 || event.button !== 0) return;
    const startX = event.clientX;
    const startY = event.clientY;
    const startPan = pan;
    let dragged = false;

    function handleMove(moveEvent: MouseEvent) {
      const dx = moveEvent.clientX - startX;
      const dy = moveEvent.clientY - startY;
      if (!dragged && Math.hypot(dx, dy) > PAN_DRAG_THRESHOLD) {
        dragged = true;
        setIsPanning(true);
      }
      if (!dragged) return;
      setPan(clampPan({ x: startPan.x + dx, y: startPan.y + dy }, zoom));
    }
    function handleUp() {
      window.removeEventListener("mousemove", handleMove);
      window.removeEventListener("mouseup", handleUp);
      if (dragged) {
        suppressClickRef.current = true;
        setIsPanning(false);
      }
    }
    window.addEventListener("mousemove", handleMove);
    window.addEventListener("mouseup", handleUp);
  }

  /** Trackpad/mouse-wheel zoom and pan over the image -- the other half of
   * the zoom/pan gesture set alongside the Ctrl+/Ctrl- shortcut and
   * drag-to-pan above. A native (non-passive) listener, not React's
   * onWheel prop -- React has attached wheel/touchmove listeners passively
   * by default since v17 (for scroll performance), which makes
   * event.preventDefault() inside a JSX onWheel handler a silent no-op;
   * this needs a real preventDefault to stop the page zooming/scrolling
   * along with the image, so it's wired up here instead, the same way the
   * keydown listener above is.
   *
   * Browsers report a trackpad pinch as a wheel event with ctrlKey true
   * (there's no separate "pinch" event), so checking ctrlKey here handles
   * both a real Ctrl+scroll *and* a pinch gesture with the same code path
   * -- zooming, centered the same way the keyboard shortcut is rather than
   * anchored to the cursor, so all three zoom inputs (keyboard, wheel,
   * pinch) agree on what "zoom" visually does. Plain wheel/two-finger
   * scroll (no ctrlKey) pans instead, following the scroll direction the
   * same way a map or canvas app does (scroll down -> content shifts up).
   */
  useEffect(() => {
    const wrap = imageWrapRef.current;
    if (!wrap) return;
    function handleWheel(event: WheelEvent) {
      if (event.ctrlKey) {
        event.preventDefault();
        const nextZoom = Math.min(
          ZOOM_MAX,
          Math.max(ZOOM_MIN, zoom * Math.exp(-event.deltaY * ZOOM_WHEEL_SENSITIVITY)),
        );
        setZoom(nextZoom);
        setPan((current) => clampPan(current, nextZoom));
        return;
      }
      if (zoom <= 1) return;
      event.preventDefault();
      setPan((current) => clampPan({ x: current.x - event.deltaX, y: current.y - event.deltaY }, zoom));
    }
    wrap.addEventListener("wheel", handleWheel, { passive: false });
    return () => wrap.removeEventListener("wheel", handleWheel);
    // Re-bound whenever zoom changes (same reasoning as the keydown
    // effect's own comment) or the image itself changes (imageWrapRef only
    // has an element to attach to once imageSrc is truthy).
  }, [zoom, imageSrc]);
  // ---------------------------------------------------------------------

  // The live "drop" cursor while picking a color (see the JSX below and
  // .colorDrop) -- x/y are pixels relative to .imageWrap, color is whatever
  // pixel is currently under the pointer. null hides it (not picking, or
  // the pointer's left the image).
  const [colorDropHover, setColorDropHover] = useState<{
    x: number;
    y: number;
    color: string | null;
  } | null>(null);
  // Cached so mousemove doesn't redraw the whole image to a canvas on every
  // event -- just drawn once per image and read back a pixel at a time.
  // Reset whenever imageSrc changes out from under it.
  const hoverCanvasRef = useRef<CanvasRenderingContext2D | null>(null);

  useEffect(() => {
    hoverCanvasRef.current = null;
  }, [imageSrc]);

  function getHoverContext(img: HTMLImageElement): CanvasRenderingContext2D | null {
    if (hoverCanvasRef.current) return hoverCanvasRef.current;
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return null;
    try {
      context.drawImage(img, 0, 0);
    } catch {
      return null;
    }
    hoverCanvasRef.current = context;
    return context;
  }

  function handleImageMouseMove(event: React.MouseEvent<HTMLImageElement>) {
    if (!pickingColor) return;
    const wrapBounds = event.currentTarget.parentElement!.getBoundingClientRect();
    const imgBounds = event.currentTarget.getBoundingClientRect();
    const fx = (event.clientX - imgBounds.left) / imgBounds.width;
    const fy = (event.clientY - imgBounds.top) / imgBounds.height;
    const context = getHoverContext(event.currentTarget);
    let color: string | null = null;
    if (context) {
      const x = Math.min(context.canvas.width - 1, Math.max(0, Math.floor(fx * context.canvas.width)));
      const y = Math.min(
        context.canvas.height - 1,
        Math.max(0, Math.floor(fy * context.canvas.height)),
      );
      try {
        const [r, g, b] = context.getImageData(x, y, 1, 1).data;
        color = toHexColor(r, g, b);
      } catch {
        color = null;
      }
    }
    setColorDropHover({
      x: event.clientX - wrapBounds.left,
      y: event.clientY - wrapBounds.top,
      color,
    });
  }

  function handleImageMouseLeave() {
    setColorDropHover(null);
  }

  useEffect(() => {
    if (!webcamActive) {
      return;
    }

    let cancelled = false;
    navigator.mediaDevices
      ?.getUserMedia({ video: true })
      .then((stream) => {
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setStreamReady(true);
      })
      .catch(() => {
        onWebcamError("Couldn't access the webcam. Check browser permissions.");
      });

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setStreamReady(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [webcamActive]);

  function handleCapture() {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) onCaptureFrame(blob);
    }, "image/png");
  }

  // Mask-RCNN's contours come back effectively pixel-precise -- hundreds to
  // low-thousands of points per hold, unsimplified -- so serializing them
  // into SVG's `points` format is real work, not a rounding error. Doing
  // it inline in JSX (the old code) reran that work, for every segment,
  // on every render -- including the high-frequency ones that have
  // nothing to do with the polygons themselves: color-pick hover and pan
  // both setState on every mousemove. With enough holds/vertices that adds
  // up to a main-thread-blocking amount of string-building 60+ times a
  // second while dragging. Memoized here so it only redoes the work when
  // `segments` itself actually changes (a hold added/removed/redetected),
  // not on every unrelated re-render.
  const polygonPointsById = useMemo(() => {
    const map = new Map<string, string>();
    for (const segment of segments) {
      map.set(segment.segmentId, segment.polygon.map((p) => `${p.x},${p.y}`).join(" "));
    }
    return map;
  }, [segments]);

  function handleImageClick(event: React.MouseEvent<HTMLImageElement>) {
    if (suppressClickRef.current) {
      // This click is the tail end of a pan drag, not a real click --
      // see handleImageWrapMouseDown.
      suppressClickRef.current = false;
      return;
    }
    if (!interactive) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const coordinate: Coordinate = {
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
    };
    if (pickingColor) {
      // Prefer the clicked hold's own average color over the raw pixel
      // under the cursor -- see sampleSegmentColor above. Only falls back
      // to a single-pixel sample if the click missed every polygon.
      const clickedSegment = findSegmentAtPoint(segments, coordinate);
      const color = clickedSegment
        ? sampleSegmentColor(event.currentTarget, clickedSegment.polygon)
        : sampleColorAt(event.currentTarget, coordinate);
      if (color) onPickColor?.(color);
      return;
    }
    if (!onAddSegmentPoint) return;
    onAddSegmentPoint(coordinate);
  }

  return (
    <div
      className={styles.box}
      onMouseEnter={() => {
        hoveringRef.current = true;
      }}
      onMouseLeave={() => {
        hoveringRef.current = false;
      }}
    >
      {webcamActive ? (
        <>
          <video ref={videoRef} className={styles.video} autoPlay playsInline muted />
          {streamReady && (
            <>
              <span className={styles.liveBadge}>
                <span className={styles.liveDot} aria-hidden />
                Live
              </span>
              <Button
                type="button"
                variant="primary"
                className={styles.captureButton}
                onClick={handleCapture}
              >
                Capture
              </Button>
            </>
          )}
        </>
      ) : imageSrc ? (
        <div
          ref={imageWrapRef}
          className={styles.imageWrap}
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            // String, not a bare number -- React's unitless-property
            // allowlist for auto-appending "px" to numeric style values
            // predates aspect-ratio, so a raw number here risks becoming
            // the invalid (and silently ignored) "1.502px".
            aspectRatio: imageAspectRatio != null ? String(imageAspectRatio) : undefined,
          }}
          onMouseDown={handleImageWrapMouseDown}
        >
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageSrc}
            alt="Working climbing wall"
            // Browsers make <img> natively draggable (HTML5 drag-and-drop) --
            // left unchecked, holding and moving the mouse here starts the
            // browser's own drag-ghost operation instead of our pan, which
            // swallows mousemove until release (see handleImageWrapMouseDown
            // above). draggable=false plus swallowing dragstart disables that
            // so plain mouse-tracking drives the pan instead.
            draggable={false}
            onDragStart={(event) => event.preventDefault()}
            // Captures the real aspect ratio for .imageWrap's inline style
            // above -- see imageAspectRatio's own comment for why that's
            // needed at all.
            onLoad={(event) => {
              const { naturalWidth, naturalHeight } = event.currentTarget;
              if (naturalWidth > 0 && naturalHeight > 0) {
                setImageAspectRatio(naturalWidth / naturalHeight);
              }
            }}
            className={[
              styles.image,
              !interactive ? styles.imageStatic : "",
              pickingColor ? styles.imagePicking : "",
            ]
              .filter(Boolean)
              .join(" ")}
            style={{
              // Overrides .image's cursor once zoomed in -- panning takes
              // over as the primary drag gesture at that point, so the
              // crosshair/none cursors that signal click-to-add-point or
              // color-picking give way to the standard grab affordance.
              cursor: zoom > 1 ? (isPanning ? "grabbing" : "grab") : undefined,
            }}
            onClick={handleImageClick}
            onMouseMove={handleImageMouseMove}
            onMouseLeave={handleImageMouseLeave}
          />
          {lightingIntensity !== 0 && (
            // Matches the backend's own change_lighting math exactly (see
            // src/pipeline/preprocessing/augmentation_suite.py) rather than
            // approximating it with a CSS brightness() filter, which used
            // to live here -- brightness() is *multiplicative*
            // (value * factor), so shadows/blacks barely move even at max
            // intensity; the backend instead *blends every pixel toward
            // white or black*: value + (target - value) * intensity. A
            // solid white/black layer, alpha-blended normally at
            // opacity = intensity, is mathematically identical to that
            // per-pixel blend (that's literally what alpha compositing
            // computes), so this overlay is pixel-for-pixel what Finish
            // Augment will actually produce -- not a lookalike. Drawn right
            // after the image, before the polygon/chalk overlay below, so
            // chalk still visually sits on top of the lit/darkened surface
            // rather than under it.
            <span
              className={styles.lightingOverlay}
              style={{
                background: lightingIntensity > 0 ? "#fff" : "#000",
                opacity: Math.min(1, Math.abs(lightingIntensity)),
              }}
              aria-hidden
            />
          )}
          {pickingColor && (
            <span className={styles.pickingBadge}>Click a hold to sample its color</span>
          )}
          {pickingColor && colorDropHover && (
            <span
              className={styles.colorDrop}
              style={{ left: colorDropHover.x, top: colorDropHover.y }}
              aria-hidden
            >
              <svg width="32" height="32" viewBox="0 0 24 24" className={styles.colorDropShape}>
                {/* Point at the bottom (the exact sampled pixel, thanks to
                    .colorDrop's translate(-50%,-100%)), round head on top --
                    the whole drop is filled with the sampled color itself,
                    not just some separate swatch near it, so the tip really
                    does read as "this color, right here". */}
                <path
                  d="M12 1.5C7.86 1.5 4.5 4.86 4.5 9c0 5.63 7.5 13.5 7.5 13.5S19.5 14.63 19.5 9c0-4.14-3.36-7.5-7.5-7.5z"
                  fill={colorDropHover.color ?? "var(--muted-2)"}
                  stroke="#fff"
                  strokeWidth="1.4"
                  strokeLinejoin="round"
                />
                <ellipse cx="9.6" cy="7.6" rx="1.7" ry="2.4" fill="#fff" opacity="0.45" />
              </svg>
            </span>
          )}
          {segments.length > 0 && (
            <svg
              className={styles.polygonOverlay}
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              aria-hidden
            >
              {showHoldOutline &&
                segments.map((segment) => {
                  const colorHex = colorBySegmentId[segment.segmentId];
                  return (
                    <polygon
                      key={segment.segmentId}
                      className={colorHex ? styles.polygonShapeColored : styles.polygonShape}
                      points={polygonPointsById.get(segment.segmentId)}
                      style={colorHex ? ({ "--hold-color": colorHex } as CSSProperties) : undefined}
                    />
                  );
                })}
              {segments.map((segment) => {
                const chalkPercent = chalkBySegmentId[segment.segmentId] ?? 0;
                if (chalkPercent <= 0) return null;
                return (
                  <polygon
                    key={`chalk-${segment.segmentId}`}
                    className={styles.chalkShape}
                    points={polygonPointsById.get(segment.segmentId)}
                    style={{ fillOpacity: (chalkPercent / 100) * MAX_CHALK_OPACITY }}
                  />
                );
              })}
            </svg>
          )}
          {overlay}
          {interactive &&
            segments.map((segment, index) => {
              const point = segment.coordinates[0];
              if (!point) return null;
              return (
                <span
                  key={segment.segmentId}
                  className={styles.marker}
                  style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
                >
                  <span className={`${styles.markerLabel} mono`}>{index + 1}</span>
                </span>
              );
            })}
          {interactive &&
            pendingPoints.map((point, index) => (
              <span
                key={`pending-${index}`}
                className={styles.pendingMarker}
                style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
              />
            ))}
          <FlyingLogoLoader active={loading} logoRef={logoRef} className={styles.loadingOverlay} />
        </div>
      ) : (
        <div className={styles.empty}>
          <svg
            className={styles.emptyIcon}
            width="36"
            height="36"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden
          >
            <rect x="3" y="3" width="18" height="18" rx="3" stroke="currentColor" strokeWidth="1.5" />
            <circle cx="8.5" cy="8.5" r="1.75" stroke="currentColor" strokeWidth="1.5" />
            <path
              d="M21 15.5 15.5 10 6 19.5"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <p className={styles.emptyText}>Select a wall image to begin</p>
        </div>
      )}
      {imageSrc && zoom > 1 && (
        // A sibling of .imageWrap, not a child of it -- .imageWrap is what
        // pan/zoom's transform actually applies to (see the CSS), so a
        // badge living inside it would scale and slide around right along
        // with the image instead of staying put. Pinned here, directly
        // against .box itself, it stays fixed at the container's own
        // bottom-right corner regardless of how far zoomed/panned the
        // image inside it currently is.
        <span className={styles.zoomBadge} aria-hidden>
          {Math.round(zoom * 100)}%
        </span>
      )}
    </div>
  );
}
