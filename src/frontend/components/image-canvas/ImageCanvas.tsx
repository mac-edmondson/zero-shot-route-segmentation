"use client";

import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { Coordinate, Segment } from "@/lib/api";
import { Button } from "@/components/button/Button";
import styles from "./ImageCanvas.module.css";

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
  onCaptureFrame?: (blob: Blob) => void;
  onWebcamError?: (message: string) => void;
  /** Coordinates are normalized to [0, 1] of the displayed image. */
  onAddSegmentPoint?: (coordinate: Coordinate) => void;
  /**
   * False for a read-only display -- the recognition page's carried-over
   * augmented image, past the point of editing it further. Disables
   * click-to-add-point and hides the numbered segment markers/pending-point
   * dots, but keeps the lighting filter and chalk overlay showing (those
   * *are* the augmentation, not editing affordances). Defaults to true.
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
  onCaptureFrame = () => {},
  onWebcamError = () => {},
  onAddSegmentPoint,
  interactive = true,
  showHoldOutline = true,
  overlay,
}: ImageCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streamReady, setStreamReady] = useState(false);

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

  function handleImageClick(event: React.MouseEvent<HTMLImageElement>) {
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
    <div className={styles.box}>
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
        <div className={styles.imageWrap}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={imageSrc}
            alt="Working climbing wall"
            className={[
              styles.image,
              !interactive ? styles.imageStatic : "",
              pickingColor ? styles.imagePicking : "",
            ]
              .filter(Boolean)
              .join(" ")}
            style={{ filter: `brightness(${1 + ((lightingPercent - 50) / 50) * 0.9})` }}
            onClick={handleImageClick}
            onMouseMove={handleImageMouseMove}
            onMouseLeave={handleImageMouseLeave}
          />
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
                      points={segment.polygon.map((p) => `${p.x},${p.y}`).join(" ")}
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
                    points={segment.polygon.map((p) => `${p.x},${p.y}`).join(" ")}
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
          {loading && (
            <div className={styles.overlay}>
              <span className={styles.spinner} aria-hidden />
              Working…
            </div>
          )}
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
    </div>
  );
}
