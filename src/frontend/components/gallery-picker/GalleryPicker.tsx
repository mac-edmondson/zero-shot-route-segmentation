"use client";

import { useEffect, useState } from "react";
import { Button } from "@/components/button/Button";
import { apiClient } from "@/lib/api";
import type { GalleryImage } from "@/lib/api";
import styles from "./GalleryPicker.module.css";

/** How many tiles get mounted (and so how many thumbnail requests fire) per reveal step,
 * and also how many more tiles each "Load more" click adds. */
const REVEAL_BATCH_SIZE = 10;
/** Stagger between batches so the free-tier gallery server never sees more
 * than REVEAL_BATCH_SIZE concurrent thumbnail requests from one picker. */
const REVEAL_INTERVAL_MS = 350;

interface GalleryPickerProps {
  onClose: () => void;
  /** Category + name of the picked image; the caller fetches its bytes and closes the picker. */
  onSelect: (category: string, name: string) => void;
  /** True while a selection is being fetched -- disables further picks. */
  selecting: boolean;
}

/**
 * Popup picker for a wall image from the external sample-image gallery
 * (proxied through src/backend/gallery.py -- see its module docstring for
 * why a proxy is needed at all). Two steps, category then image, because
 * some categories run to 1000+ images -- loading every image across every
 * category at once was the original heaviness this replaced.
 *
 * Thumbnails render straight from the external server's URL (plain <img>,
 * unaffected by that server's missing CORS headers); only the actual
 * selected image's bytes go through the backend.
 */
export function GalleryPicker({ onClose, onSelect, selecting }: GalleryPickerProps) {
  const [categories, setCategories] = useState<string[] | null>(null);
  const [categoriesError, setCategoriesError] = useState<string | null>(null);

  const [category, setCategory] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .listGalleryCategories(controller.signal)
      .then(setCategories)
      .catch(() => {
        // Cleanup (e.g. React StrictMode's dev-only double-invoke, or the
        // picker closing mid-fetch) aborts this same fetch and lands here
        // too -- request()'s generic ApiError wrapping loses the distinct
        // AbortError, so check our own controller instead of the error.
        if (controller.signal.aborted) return;
        setCategoriesError("Couldn't load gallery categories. Try again.");
      });
    return () => controller.abort();
  }, []);

  const title = category ? `Choose a wall image — ${category}` : "Choose a category";

  return (
    <div className={styles.backdrop} onClick={onClose}>
      <div className={styles.panel} onClick={(event) => event.stopPropagation()}>
        <div className={styles.header}>
          <div className={styles.headerLeft}>
            {category && (
              <button
                type="button"
                className={styles.backButton}
                onClick={() => setCategory(null)}
                aria-label="Back to categories"
              >
                ←
              </button>
            )}
            <h2 className={styles.title}>{title}</h2>
          </div>
          <button type="button" className={styles.closeButton} onClick={onClose} aria-label="Close">
            ×
          </button>
        </div>

        {!category && (
          <>
            {categoriesError && <p className={styles.error}>{categoriesError}</p>}
            {!categories && !categoriesError && <p className={styles.status}>Loading categories…</p>}
            {categories && categories.length === 0 && (
              <p className={styles.status}>No categories found.</p>
            )}
            {categories && categories.length > 0 && (
              <div className={styles.categoryScroll}>
                <div className={styles.categoryList}>
                  {categories.map((name) => (
                    <button
                      key={name}
                      type="button"
                      className={styles.categoryTile}
                      onClick={() => setCategory(name)}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {category && (
          <GalleryCategoryImages key={category} category={category} onSelect={onSelect} selecting={selecting} />
        )}
      </div>
    </div>
  );
}

interface GalleryCategoryImagesProps {
  category: string;
  onSelect: (category: string, name: string) => void;
  selecting: boolean;
}

/**
 * Images within one category. A separate component keyed by `category`
 * (see the render site above) so switching categories remounts it fresh --
 * `images`/`error`/reveal progress naturally start back at their initial
 * values without the parent effect having to reset them itself.
 *
 * The listing itself (metadata only) is fetched in one shot, but tiles are
 * mounted progressively -- REVEAL_BATCH_SIZE at a time -- since each tile's
 * <img> fires its own request straight at the external gallery server for
 * its thumbnail; mounting all of them (a category can run to 1000+) at once
 * both jank the grid and hammers that server's free tier. Only the first
 * REVEAL_BATCH_SIZE tiles load on their own; every batch after that only
 * fires once the user clicks "Load more", so the server never sees more
 * requests than the user actually asked for.
 */
function GalleryCategoryImages({ category, onSelect, selecting }: GalleryCategoryImagesProps) {
  const [images, setImages] = useState<GalleryImage[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [revealed, setRevealed] = useState(REVEAL_BATCH_SIZE);
  const [ceiling, setCeiling] = useState(REVEAL_BATCH_SIZE);

  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .listGalleryImages(category, controller.signal)
      .then(setImages)
      .catch(() => {
        // See the categories effect above for why this checks our own
        // controller instead of the caught error.
        if (controller.signal.aborted) return;
        setError("Couldn't load that category. Try again.");
      });
    return () => controller.abort();
  }, [category]);

  useEffect(() => {
    if (!images || revealed >= images.length || revealed >= ceiling) return;
    const timer = setTimeout(() => {
      setRevealed((count) => Math.min(count + REVEAL_BATCH_SIZE, images.length, ceiling));
    }, REVEAL_INTERVAL_MS);
    return () => clearTimeout(timer);
  }, [images, revealed, ceiling]);

  const visible = images?.slice(0, revealed) ?? [];
  const hasMore = !!images && revealed < images.length && revealed >= ceiling;

  return (
    <>
      {error && <p className={styles.error}>{error}</p>}
      {!images && !error && <p className={styles.status}>Loading images…</p>}
      {images && images.length === 0 && <p className={styles.status}>No images found.</p>}
      {images && images.length > 0 && (
        <div className={styles.gridScroll}>
          <div className={styles.grid}>
            {visible.map((image) => (
              <button
                key={image.name}
                type="button"
                className={styles.tile}
                disabled={selecting}
                onClick={() => onSelect(image.category, image.name)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={image.url} alt={image.name} loading="lazy" className={styles.thumb} />
              </button>
            ))}
          </div>
        </div>
      )}
      {hasMore && (
        <Button
          type="button"
          className={styles.loadMore}
          onClick={() => setCeiling((c) => c + REVEAL_BATCH_SIZE)}
        >
          Load more
        </Button>
      )}
    </>
  );
}
