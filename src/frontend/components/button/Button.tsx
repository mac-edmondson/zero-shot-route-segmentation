"use client";

import type { ButtonHTMLAttributes } from "react";
import styles from "./Button.module.css";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** `primary` is the one accent-filled call to action per view (e.g. "Finish Augment"). */
  variant?: "primary" | "secondary";
  /** Toggled/selected state for secondary buttons, e.g. an active source picker. */
  active?: boolean;
}

/** Button matching ROUTNet's design system (see app/globals.css tokens). */
export function Button({
  variant = "secondary",
  active = false,
  className,
  ...props
}: ButtonProps) {
  const classes = [
    styles.button,
    variant === "primary" ? styles.primary : styles.secondary,
    active ? styles.active : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...props} />;
}
