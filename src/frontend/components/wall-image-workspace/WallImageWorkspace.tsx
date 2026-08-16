"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { apiClient } from "@/lib/api";
import type { Coordinate, Segment } from "@/lib/api";
import { AppHeader } from "@/components/header/AppHeader";
import { ImageSourceButtons } from "@/components/image-source-buttons/ImageSourceButtons";
import { GalleryPicker } from "@/components/gallery-picker/GalleryPicker";
import { ImageCanvas } from "@/components/image-canvas/ImageCanvas";
import { SegmentPanel } from "@/components/segment-panel/SegmentPanel";
import { LabeledSlider } from "@/components/labeled-slider/LabeledSlider";
import { Button } from "@/components/button/Button";
import { StepIndicator } from "@/components/step-indicator/StepIndicator";
import { ModelSelect } from "@/components/model-select/ModelSelect";
import styles from "./WallImageWorkspace.module.css";

const HOLD_MODEL_OPTIONS = ["Color-only", "DINO-only", "Combined"];
const ROUTE_MODEL_OPTIONS = ["Color-only", "Color + Spatial", "Combined"];

/**
 * How long SegmentPanel's own closing sequence takes end to end (shrink
 * 530ms + reform ~895ms + its final opacity fade 450ms -- see
 * SegmentPanel.tsx's SHRINK_MS/REFORM_MS and its .closing rule) before the
 * model pickers take its place. Kept as an explicit constant here, in the
 * same spirit as this codebase's other cross-timing comments, because
 * there's no way to observe "SegmentPanel's animation finished" from
 * outside it -- if those constants change, this needs to change with them.
 */
const MODEL_SELECT_REVEAL_MS = 1875;

function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

function createLocalImageId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
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
  /** Original image bytes, kept around so a detect call can send the whole
   * image again -- the backend is stateless and never stores it. */
  const [imageFile, setImageFile] = useState<File | Blob | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  /** Points clicked but not yet sent for detection. */
  const [pendingPoints, setPendingPoints] = useState<Coordinate[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [chalkBySegmentId, setChalkBySegmentId] = useState<Record<string, number>>({});
  const [lighting, setLighting] = useState(0);
  const [webcamActive, setWebcamActive] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [gallerySelecting, setGallerySelecting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [augmentDone, setAugmentDone] = useState(false);
  const [toolbarEntered, setToolbarEntered] = useState(false);
  const [showModelSelect, setShowModelSelect] = useState(false);
  const [modelSelectEntered, setModelSelectEntered] = useState(false);
  const [holdModel, setHoldModel] = useState<string | null>(null);
  const [routeModel, setRouteModel] = useState<string | null>(null);

  // Mount, wait a paint, then trigger -- without the gap there's no
  // "before" frame for the browser to animate from, so the toolbar's two
  // pieces would just appear already in place instead of rising in.
  useEffect(() => {
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setToolbarEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, []);

  // Once the toolbar and segment panel have both fully finished leaving,
  // the model pickers take over that same area -- mounted only then (not
  // shown-but-invisible from the start), so the same "wait a paint, then
  // trigger" entrance below has a real "before" frame to animate from.
  useEffect(() => {
    if (!augmentDone) return;
    const revealTimeout = setTimeout(() => setShowModelSelect(true), MODEL_SELECT_REVEAL_MS);
    return () => clearTimeout(revealTimeout);
  }, [augmentDone]);

  useEffect(() => {
    if (!showModelSelect) return;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setModelSelectEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [showModelSelect]);

  // Appears the moment either dropdown has a pick -- doesn't wait for both,
  // per how this was asked for ("once the user selects any of the model").
  const showRecognitionButton = holdModel !== null || routeModel !== null;
  const [recognitionEntered, setRecognitionEntered] = useState(false);

  useEffect(() => {
    if (!showRecognitionButton) {
      setRecognitionEntered(false);
      return;
    }
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setRecognitionEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [showRecognitionButton]);

  const loadImage = useCallback(async (file: File | Blob) => {
    setLoading(true);
    setError(null);
    try {
      // The image never goes to the backend here -- it's stateless and
      // never stores images (see the imageFile comment above); loading it
      // into the panel is purely a local render. The whole file only gets
      // sent over the wire later, per detect call, via detectWorkingSegments.
      const dataUrl = await fileToDataUrl(file);
      setImageId(createLocalImageId());
      setImageSrc(dataUrl);
      setImageFile(file);
      setSegments([]);
      setPendingPoints([]);
      setChalkBySegmentId({});
      setWebcamActive(false);
      setAugmentDone(false);
      setShowModelSelect(false);
      setModelSelectEntered(false);
      setHoldModel(null);
      setRouteModel(null);
    } catch {
      setError("Couldn't load that image. Try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  async function handleGallerySelect(category: string, name: string) {
    setGallerySelecting(true);
    setError(null);
    try {
      const blob = await apiClient.fetchGalleryImage(category, name);
      await loadImage(blob);
      setGalleryOpen(false);
    } catch {
      setError("Couldn't load that gallery image. Try again.");
    } finally {
      setGallerySelecting(false);
    }
  }

  function handleAddSegmentPoint(coordinate: Coordinate) {
    if (!imageId || loading || detecting) return;
    setPendingPoints((prev) => [...prev, coordinate]);
  }

  async function handleDetectSegments() {
    if (!imageId || !imageFile || pendingPoints.length === 0) return;
    setDetecting(true);
    setError(null);
    try {
      const detected = await apiClient.detectWorkingSegments(imageFile, pendingPoints);
      setSegments((prev) => [...prev, ...detected]);
      setChalkBySegmentId((prev) => {
        const next = { ...prev };
        for (const segment of detected) next[segment.segmentId] = 0;
        return next;
      });
      setPendingPoints([]);
    } catch {
      setError("Couldn't detect holds there. Try again.");
    } finally {
      setDetecting(false);
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
      // Stays on this same page -- no navigation. The step indicator shifts
      // to Recognition (shown as in-progress, not complete -- see
      // `handingOff` below) and the toolbar/segment panel fade out; nothing
      // else moves or resizes.
      setAugmentDone(true);
    } catch {
      setError("Couldn't finish augmentation. Try again.");
    } finally {
      setLoading(false);
    }
  }

  // Reverses the handoff: the toolbar/segment panel's own CSS transitions
  // are already bidirectional (removing the class they gained just plays
  // them backwards), so flipping augmentDone back to false is enough to
  // bring those back on its own. The model-select panel doesn't have that
  // built in (it's only ever mounted forward, via showModelSelect), so its
  // own reverse fade is played here explicitly before unmounting it.
  function handleBackToAugment() {
    setAugmentDone(false);
    if (showModelSelect) {
      setModelSelectEntered(false);
      setTimeout(() => setShowModelSelect(false), 550);
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
          <div className={`${styles.toolbarItem} ${toolbarEntered ? styles.toolbarItemIn : ""}`}>
            <ImageSourceButtons
              webcamActive={webcamActive}
              disabled={loading}
              onFileSelected={loadImage}
              onToggleWebcam={() => {
                setError(null);
                setWebcamActive((prev) => !prev);
              }}
              onOpenGallery={() => {
                setError(null);
                setGalleryOpen(true);
              }}
            />
          </div>
          <div
            className={`${styles.lighting} ${styles.toolbarItem} ${styles.toolbarItemLighting} ${toolbarEntered ? styles.toolbarItemIn : ""}`}
          >
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
            chalkBySegmentId={chalkBySegmentId}
            pendingPoints={pendingPoints}
            webcamActive={webcamActive}
            loading={loading || detecting}
            onCaptureFrame={loadImage}
            onWebcamError={(message) => {
              setError(message);
              setWebcamActive(false);
            }}
            onAddSegmentPoint={handleAddSegmentPoint}
          />
          <Button
            type="button"
            variant="secondary"
            disabled={!imageId || pendingPoints.length === 0 || detecting}
            onClick={handleDetectSegments}
          >
            {detecting ? "Detecting…" : `Detect Holds${pendingPoints.length > 0 ? ` (${pendingPoints.length})` : ""}`}
          </Button>
          <Button
            type="button"
            variant="primary"
            disabled={!imageId || loading}
            onClick={augmentDone ? handleBackToAugment : handleFinishAugment}
          >
            {augmentDone ? "Back to Augment" : "Finish Augment"}
          </Button>
        </div>

        <div className={styles.segmentSlot}>
          <SegmentPanel
            hasImage={!!imageId}
            segments={segments}
            chalkBySegmentId={chalkBySegmentId}
            onChalkChange={handleChalkChange}
            onRemove={handleRemoveSegment}
            closing={augmentDone}
          />

          {showModelSelect && (
            <div
              className={`${styles.modelSelectPanel} ${modelSelectEntered ? styles.modelSelectPanelIn : ""}`}
            >
              <div className={styles.modelSelectRow}>
                <ModelSelect
                  label="hold model"
                  options={HOLD_MODEL_OPTIONS}
                  value={holdModel}
                  onChange={setHoldModel}
                />
                <ModelSelect
                  label="route model"
                  options={ROUTE_MODEL_OPTIONS}
                  value={routeModel}
                  onChange={setRouteModel}
                />
              </div>

              {showRecognitionButton && (
                <div
                  className={`${styles.recognitionButtonWrap} ${recognitionEntered ? styles.recognitionButtonWrapIn : ""}`}
                >
                  <Button
                    type="button"
                    variant="primary"
                    onClick={() => router.push("/recognition")}
                  >
                    Recognition
                  </Button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {galleryOpen && (
        <GalleryPicker
          onClose={() => setGalleryOpen(false)}
          onSelect={handleGallerySelect}
          selecting={gallerySelecting}
        />
      )}
    </div>
  );
}
