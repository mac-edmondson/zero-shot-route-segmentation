"use client";

import type { CSSProperties } from "react";
import styles from "./LabeledSlider.module.css";

interface LabeledSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  disabled?: boolean;
}

/**
 * Label + range input + percentage readout, matching the "Lighting" and
 * per-segment "Chalk" sliders in the ROUTNet wireframe
 * (Project stuff/UI_page_1.png), redressed with a filled accent track.
 */
export function LabeledSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  disabled = false,
}: LabeledSliderProps) {
  const percent = ((value - min) / (max - min)) * 100;

  return (
    <div className={styles.row}>
      <span className={styles.label}>{label}</span>
      <input
        type="range"
        className={styles.slider}
        style={{ "--fill": `${percent}%` } as CSSProperties}
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={label}
      />
      <span className={`${styles.value} mono`}>{value}%</span>
    </div>
  );
}
