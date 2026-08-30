import type { ReactNode, RefObject } from "react";
import { Wordmark } from "@/components/wordmark/Wordmark";
import styles from "./AppHeader.module.css";

interface AppHeaderProps {
  title: string;
  /** Right-aligned slot -- typically a <StepIndicator />. */
  right?: ReactNode;
  /**
   * Attached to the wrapping <h1> -- lets a caller measure exactly where
   * the real logo sits on screen (getBoundingClientRect), e.g. as the
   * "from" rect for a FLIP animation that flies a clone of it somewhere
   * else (see WallImageWorkspace's recognition loader).
   */
  titleRef?: RefObject<HTMLHeadingElement | null>;
  /**
   * Makes the logo clickable -- e.g. WallImageWorkspace uses this to reset
   * back to its own landing state. Omitted entirely (not just a no-op),
   * the logo renders as plain, non-interactive text, same as before this
   * prop existed.
   */
  onLogoClick?: () => void;
}

/** Wordmark + right-side slot + divider, from Project stuff/UI_page_1.png, restyled. */
export function AppHeader({ title, right, titleRef, onLogoClick }: AppHeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.row}>
        <h1 className={styles.title} ref={titleRef}>
          {onLogoClick ? (
            <button type="button" className={styles.logoButton} onClick={onLogoClick}>
              <Wordmark text={title} />
              <span className="sr-only">{title}</span>
            </button>
          ) : (
            <>
              <Wordmark text={title} />
              <span className="sr-only">{title}</span>
            </>
          )}
        </h1>
        {right}
      </div>
      <div className={styles.rule} />
    </header>
  );
}
