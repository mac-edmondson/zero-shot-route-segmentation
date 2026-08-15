import type { ReactNode } from "react";
import { Wordmark } from "@/components/wordmark/Wordmark";
import styles from "./AppHeader.module.css";

interface AppHeaderProps {
  title: string;
  /** Right-aligned slot -- typically a <StepIndicator />. */
  right?: ReactNode;
}

/** Wordmark + right-side slot + divider, from Project stuff/UI_page_1.png, restyled. */
export function AppHeader({ title, right }: AppHeaderProps) {
  return (
    <header className={styles.header}>
      <div className={styles.row}>
        <h1 className={styles.title}>
          <Wordmark text={title} />
          <span className="sr-only">{title}</span>
        </h1>
        {right}
      </div>
      <div className={styles.rule} />
    </header>
  );
}
