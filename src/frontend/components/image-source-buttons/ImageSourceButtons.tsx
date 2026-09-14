"use client";

import { useRef } from "react";
import { Button } from "@/components/button/Button";
import styles from "./ImageSourceButtons.module.css";

interface ImageSourceButtonsProps {
  webcamActive: boolean;
  disabled?: boolean;
  onFileSelected: (file: File) => void;
  onToggleWebcam: () => void;
  /** Opens the gallery picker (src/backend/gallery.py-backed sample images). */
  onOpenGallery: () => void;
}

/** "Upload" / "Gallery" / "Webcam" source picker buttons. */
export function ImageSourceButtons({
  webcamActive,
  disabled = false,
  onFileSelected,
  onToggleWebcam,
  onOpenGallery,
}: ImageSourceButtonsProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  return (
    <div className={styles.row}>
      <Button
        type="button"
        disabled={disabled}
        onClick={() => fileInputRef.current?.click()}
      >
        Upload
      </Button>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className={styles.hiddenInput}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onFileSelected(file);
          }
          event.target.value = "";
        }}
      />
      <Button type="button" disabled={disabled} onClick={onOpenGallery}>
        Gallery
      </Button>
      <Button
        type="button"
        active={webcamActive}
        disabled={disabled}
        onClick={onToggleWebcam}
      >
        Webcam
      </Button>
    </div>
  );
}
