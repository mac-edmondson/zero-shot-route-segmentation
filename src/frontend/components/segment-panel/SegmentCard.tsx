"use client";

import { LabeledSlider } from "@/components/labeled-slider/LabeledSlider";
import { Button } from "@/components/button/Button";
import styles from "./SegmentCard.module.css";

interface SegmentCardProps {
  index: number;
  chalkPercent: number;
  onChalkChange: (value: number) => void;
  onRemove: () => void;
  /** Hex color sampled from the image for this segment, if any -- shown as
   * a swatch and used to tint this segment's own outline on the canvas. */
  colorHex: string | null;
  /** True while this card is the one waiting for a click on the image to
   * sample from -- see ImageCanvas's pickingColor/onPickColor. */
  picking: boolean;
  onChooseColor: () => void;
}

/** One "Segment N" card with its Chalk slider and color eyedropper, from the ROUTNet wireframe. */
export function SegmentCard({
  index,
  chalkPercent,
  onChalkChange,
  onRemove,
  colorHex,
  picking,
  onChooseColor,
}: SegmentCardProps) {
  return (
    <div className={styles.card}>
      <div className={styles.titleRow}>
        <span className={styles.title}>
          <span className={`${styles.badge} mono`}>{index + 1}</span>
          Segment {index + 1}
        </span>
        <button
          type="button"
          className={styles.remove}
          onClick={onRemove}
          aria-label={`Remove segment ${index + 1}`}
        >
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden>
            <path
              d="M2 2l8 8M10 2l-8 8"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            />
          </svg>
        </button>
      </div>
      <LabeledSlider label="Chalk" value={chalkPercent} onChange={onChalkChange} />
      <div className={styles.colorRow}>
        <span
          className={styles.swatch}
          style={{ background: colorHex ?? "transparent" }}
          aria-hidden
        />
        <Button
          type="button"
          className={styles.colorButton}
          active={picking}
          onClick={onChooseColor}
        >
          {picking ? "Click a hold on the image…" : colorHex ? "Change color" : "Choose color"}
        </Button>
      </div>
    </div>
  );
}
