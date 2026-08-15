"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./ModelSelect.module.css";

interface ModelSelectProps {
  /** Used to build the closed-state placeholder, e.g. "hold model" -> "Select hold model". */
  label: string;
  /** Exactly the choices shown when open -- this component doesn't enforce a count itself. */
  options: string[];
  value: string | null;
  onChange: (value: string) => void;
}

/**
 * A single dropdown picker matching ROUTNet's glass-surface design system --
 * closed by default showing "Select {label}", opens into a short list on
 * click. Closes on an outside click, Escape, or picking an option.
 */
export function ModelSelect({ label, options, value, onChange }: ModelSelectProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function handlePointerDown(event: PointerEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className={styles.root}>
      <button
        type="button"
        className={`${styles.trigger} ${open ? styles.triggerOpen : ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen((prev) => !prev)}
      >
        <span className={value ? styles.triggerValue : styles.triggerPlaceholder}>
          {value ?? `Select ${label}`}
        </span>
        <svg className={styles.chevron} width="12" height="8" viewBox="0 0 12 8" aria-hidden focusable="false">
          <path
            d="M1 1.5 L6 6.5 L11 1.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>

      <div className={`${styles.menuOuter} ${open ? styles.menuOuterOpen : ""}`}>
        <div className={styles.menuInner}>
          <ul className={styles.menu} role="listbox">
            {options.map((option) => (
              <li key={option}>
                <button
                  type="button"
                  role="option"
                  aria-selected={value === option}
                  className={`${styles.option} ${value === option ? styles.optionSelected : ""}`}
                  onClick={() => {
                    onChange(option);
                    setOpen(false);
                  }}
                >
                  {option}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
