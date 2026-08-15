"use client";

import { LabeledSlider } from "@/components/labeled-slider/LabeledSlider";
import styles from "./SegmentCard.module.css";

interface SegmentCardProps {
  index: number;
  chalkPercent: number;
  onChalkChange: (value: number) => void;
  onRemove: () => void;
}

/** One "Segment N" card with its Chalk slider, from the ROUTNet wireframe. */
export function SegmentCard({ index, chalkPercent, onChalkChange, onRemove }: SegmentCardProps) {
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
    </div>
  );
}
