"use client";

import { useEffect, useRef, useState } from "react";
import type { Coordinate, Segment } from "@/lib/api";
import { Button } from "@/components/button/Button";
import styles from "./ImageCanvas.module.css";

interface ImageCanvasProps {
  imageSrc: string | null;
  /** 0-100. Previewed live on the image via a brightness filter; 0 is unmodified. */
  lightingPercent: number;
  segments: Segment[];
  /** Points clicked but not yet submitted for detection. */
  pendingPoints: Coordinate[];
  webcamActive: boolean;
  loading: boolean;
  onCaptureFrame: (blob: Blob) => void;
  onWebcamError: (message: string) => void;
  /** Coordinates are normalized to [0, 1] of the displayed image. */
  onAddSegmentPoint: (coordinate: Coordinate) => void;
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
  pendingPoints,
  webcamActive,
  loading,
  onCaptureFrame,
  onWebcamError,
  onAddSegmentPoint,
}: ImageCanvasProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [streamReady, setStreamReady] = useState(false);

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
    const bounds = event.currentTarget.getBoundingClientRect();
    onAddSegmentPoint({
      x: (event.clientX - bounds.left) / bounds.width,
      y: (event.clientY - bounds.top) / bounds.height,
    });
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
            className={styles.image}
            style={{ filter: `brightness(${1 + (lightingPercent / 100) * 0.9})` }}
            onClick={handleImageClick}
          />
          {segments.length > 0 && (
            <svg
              className={styles.polygonOverlay}
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              aria-hidden
            >
              {segments.map((segment) => (
                <polygon
                  key={segment.segmentId}
                  className={styles.polygonShape}
                  points={segment.polygon.map((p) => `${p.x},${p.y}`).join(" ")}
                />
              ))}
            </svg>
          )}
          {segments.map((segment, index) => {
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
          {pendingPoints.map((point, index) => (
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
