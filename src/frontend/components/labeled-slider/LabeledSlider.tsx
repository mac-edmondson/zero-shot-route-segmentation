"use client";

import type { CSSProperties } from "react";
import styles from "./LabeledSlider.module.css";

interface LabeledSliderProps {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  /** Defaults to 1 (whole-number steps) -- matches the Chalk sliders'
   * existing 0-100 integer range. Lighting overrides this to move in finer
   * fractional steps across its own -1 to 1 range. */
  step?: number;
  disabled?: boolean;
  /** How the readout on the right renders `value` -- defaults to Chalk's
   * existing "N%" display. Lighting overrides this to show a plain
   * -1.00 to 1.00 number instead, since it isn't a percentage. */
  formatValue?: (value: number) => string;
}

/**
 * Label + range input + readout, matching the "Lighting" and per-segment
 * "Chalk" sliders in the ROUTNet wireframe (Project stuff/UI_page_1.png),
 * redressed with a filled accent track.
 */
export function LabeledSlider({
  label,
  value,
  onChange,
  min = 0,
  max = 100,
  step = 1,
  disabled = false,
  formatValue = (v) => `${v}%`,
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
        step={step}
        value={value}
        disabled={disabled}
        onChange={(event) => onChange(Number(event.target.value))}
        aria-label={label}
      />
      <span className={`${styles.value} mono`}>{formatValue(value)}</span>
    </div>
  );
}
