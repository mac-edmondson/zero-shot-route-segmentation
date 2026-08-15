import Link from "next/link";
import { AppHeader } from "@/components/header/AppHeader";
import { StepIndicator } from "@/components/step-indicator/StepIndicator";
import styles from "./page.module.css";

/**
 * Placeholder for the recognition step. Should eventually call
 * `apiClient.inferWorkingPipeline()` (@/lib/api) and render routes per
 * docs/spec/pipeline/interfaces/route-classifier.md.
 *
 * TODO: build this out once RouteDetectionPipeline has a real backend.
 */
export default function RecognitionPage() {
  return (
    <div className={styles.page}>
      <AppHeader
        title="ROUTNet"
        right={<StepIndicator current="recognition" uploaded augmentDone />}
      />
      <p className={styles.body}>
        Recognition isn&apos;t built yet.{" "}
        <Link href="/" className={styles.link}>
          Back to augmentation
        </Link>
        .
      </p>
    </div>
  );
}
