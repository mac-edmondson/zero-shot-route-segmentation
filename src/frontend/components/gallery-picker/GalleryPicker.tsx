"use client";

import { useEffect, useState } from "react";
import { apiClient } from "@/lib/api";
import type { GalleryImage } from "@/lib/api";
import styles from "./GalleryPicker.module.css";

interface GalleryPickerProps {
  onClose: () => void;
  /** Name of the picked image; the caller fetches its bytes and closes the picker. */
  onSelect: (name: string) => void;
  /** True while a selection is being fetched -- disables further picks. */
  selecting: boolean;
}

/**
 * Popup tile grid for picking a wall image from the external sample-image
 * gallery (proxied through src/backend/gallery.py -- see its module
 * docstring for why a proxy is needed at all). Thumbnails render straight
 * from the external server's URL (plain <img>, unaffected by that server's
 * missing CORS headers); only the actual selected image's bytes go through
 * the backend.
 */
export function GalleryPicker({ onClose, onSelect, selecting }: GalleryPickerProps) {
  const [images, setImages] = useState<GalleryImage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .listGalleryImages(controller.signal)
      .then(setImages)
      .catch(() => setError("Couldn't load the gallery. Try again."));
    return () => controller.abort();
  }, []);

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <div className={styles.panel} onClick={(event) => event.stopPropagation()}>
        <div className={styles.header}>
          <h2 className={styles.title}>Choose a wall image</h2>
          <button type="button" className={styles.closeButton} onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {error && <p className={styles.error}>{error}</p>}

        {!images && !error && <p className={styles.status}>Loading gallery…</p>}

        {images && images.length === 0 && <p className={styles.status}>No images found.</p>}

        {images && images.length > 0 && (
          <div className={styles.grid}>
            {images.map((image) => (
              <button
                key={image.name}
                type="button"
                className={styles.tile}
                disabled={selecting}
                onClick={() => onSelect(image.name)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image.url} alt={image.name} loading="lazy" className={styles.thumb} />
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
