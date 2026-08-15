"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { apiClient } from "@/lib/api";
import type { Coordinate, Segment } from "@/lib/api";
import { AppHeader } from "@/components/header/AppHeader";
import { ImageSourceButtons } from "@/components/image-source-buttons/ImageSourceButtons";
import { ImageCanvas } from "@/components/image-canvas/ImageCanvas";
import { SegmentPanel } from "@/components/segment-panel/SegmentPanel";
import { LabeledSlider } from "@/components/labeled-slider/LabeledSlider";
import { Button } from "@/components/button/Button";
import { StepIndicator } from "@/components/step-indicator/StepIndicator";
import styles from "./WallImageWorkspace.module.css";

/**
 * How long the handoff to recognition plays before we navigate away: the
 * step indicator's own unlock pop, and the toolbar/segment-panel collapse
 * below, both run inside this window (see .toolbarSlot / .segmentSlot in
 * the stylesheet -- kept in lockstep with this the same way SegmentPanel's
 * own JS timers stay in lockstep with its CSS transition durations).
 */
const HANDOFF_MS = 600;

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * The ROUTNet landing page (Project stuff/UI_page_1.png): pick or capture a
 * wall image, mark hold segments, adjust lighting/chalk, then hand off to
 * the recognition step. Talks only to `apiClient` (`@/lib/api`), which is
 * backed by an in-memory mock until the real backend
 * (docs/spec/pipeline/interfaces/dashboard-backend.md) exists.
 */
export function WallImageWorkspace() {
  const router = useRouter();

  const [imageId, setImageId] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  const [chalkBySegmentId, setChalkBySegmentId] = useState<Record<string, number>>({});
  const [lighting, setLighting] = useState(0);
  const [webcamActive, setWebcamActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [augmentDone, setAugmentDone] = useState(false);

  const loadImage = useCallback(async (file: File | Blob) => {
    setLoading(true);
    setError(null);
    try {
      const summary = await apiClient.uploadImage(file);
      const working = await apiClient.setWorkingImage(summary.id);
      setImageId(working.imageId);
      setImageSrc(working.image);
      setSegments([]);
      setChalkBySegmentId({});
      setWebcamActive(false);
      setAugmentDone(false);
    } catch {
      setError("Couldn't load that image. Try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  async function handleAddSegmentPoint(coordinate: Coordinate) {
    if (!imageId || loading) return;
    try {
      const segment = await apiClient.addWorkingSegment([coordinate]);
      setSegments((prev) => [...prev, segment]);
      setChalkBySegmentId((prev) => ({ ...prev, [segment.segmentId]: 0 }));
    } catch {
      setError("Couldn't add a segment there. Try again.");
    }
  }

  function handleChalkChange(segmentId: string, value: number) {
    setChalkBySegmentId((prev) => ({ ...prev, [segmentId]: value }));
  }

  async function handleRemoveSegment(segmentId: string) {
    setSegments((prev) => prev.filter((segment) => segment.segmentId !== segmentId));
    setChalkBySegmentId((prev) => {
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
    try {
      await apiClient.deleteWorkingSegment(segmentId);
    } catch {
      setError("Segment removed locally, but the backend couldn't confirm it.");
    }
  }

  async function handleFinishAugment() {
    if (!imageId) {
      setError("Select a wall image first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await apiClient.augmentWorkingImage({
        lightingPercent: lighting,
        segments: segments.map((segment) => ({
          segmentId: segment.segmentId,
          chalkPercent: chalkBySegmentId[segment.segmentId] ?? 0,
        })),
      });
      // The page itself doesn't jump anywhere yet: the step indicator shifts
      // to Recognition (shown as in-progress, not complete -- see
      // `handingOff` below) while the toolbar and segment panel collapse
      // away, leaving just the wall image and the header in place. Only
      // once that's played out do we actually navigate.
      setAugmentDone(true);
      await wait(HANDOFF_MS);
      router.push("/recognition");
    } catch {
      setError("Couldn't finish augmentation. Try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.workspace}>
      <AppHeader
        title="ROUTNet"
        right={
          <StepIndicator
            current="augment"
            uploaded={!!imageId}
            augmentDone={augmentDone}
            handingOff={augmentDone}
          />
        }
      />

      <div className={`${styles.toolbarSlot} ${augmentDone ? styles.leaving : ""}`}>
        <div className={styles.toolbar}>
          <ImageSourceButtons
            webcamActive={webcamActive}
            disabled={loading}
            onFileSelected={loadImage}
            onToggleWebcam={() => {
              setError(null);
              setWebcamActive((prev) => !prev);
            }}
          />
          <div className={styles.lighting}>
            <LabeledSlider label="Lighting" value={lighting} onChange={setLighting} />
          </div>
        </div>
      </div>

      {error && (
        <p className={styles.error}>
          <span className={styles.errorDot} aria-hidden />
          {error}
        </p>
      )}

      <div className={styles.mainGrid}>
        <div className={styles.canvasColumn}>
          <ImageCanvas
            imageSrc={imageSrc}
            lightingPercent={lighting}
            segments={segments}
            webcamActive={webcamActive}
            loading={loading}
            onCaptureFrame={loadImage}
            onWebcamError={(message) => {
              setError(message);
              setWebcamActive(false);
            }}
            onAddSegmentPoint={handleAddSegmentPoint}
          />
          <Button
            type="button"
            variant="primary"
            disabled={!imageId || loading}
            onClick={handleFinishAugment}
          >
            Finish Augment
          </Button>
        </div>

        <div className={`${styles.segmentSlot} ${augmentDone ? styles.leaving : ""}`}>
          <SegmentPanel
            hasImage={!!imageId}
            segments={segments}
            chalkBySegmentId={chalkBySegmentId}
            onChalkChange={handleChalkChange}
            onRemove={handleRemoveSegment}
          />
        </div>
      </div>
    </div>
  );
}
