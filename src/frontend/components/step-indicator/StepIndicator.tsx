"use client";

import { useEffect, useRef, useState } from "react";
import styles from "./StepIndicator.module.css";

interface StepIndicatorProps {
  /** Has a working image been loaded yet? */
  uploaded: boolean;
  /** Has "Finish Augment" succeeded yet? */
  augmentDone: boolean;
  /** Which page is actually being viewed -- only affects which lit node pulses as "active". */
  current: "augment" | "recognition";
  /**
   * True for the brief handoff window on the augment page itself, right
   * after "Finish Augment" succeeds but before the recognition step is
   * actually reached -- shows Recognition as in-progress (half-filled)
   * rather than snapping straight to fully complete before the user's
   * even there. Ignored once `current` is "recognition" (it's simply
   * "active" at that point).
   */
  handingOff?: boolean;
}

type NodeState = "locked" | "inProgress" | "complete" | "active";

const LIT_RANK: Record<NodeState, number> = { locked: 0, inProgress: 1, complete: 2, active: 2 };

/** True for ~600ms right after `state` ranks up (never on the way back down). */
function usePopOnLit(state: NodeState): boolean {
  const [pop, setPop] = useState(false);
  const prevRank = useRef(LIT_RANK[state]);

  useEffect(() => {
    const rank = LIT_RANK[state];
    if (rank > prevRank.current) {
      setPop(true);
      const timeout = setTimeout(() => setPop(false), 600);
      prevRank.current = rank;
      return () => clearTimeout(timeout);
    }
    prevRank.current = rank;
  }, [state]);

  return pop;
}

function Node({
  label,
  state,
  lockedHint,
}: {
  label: string;
  state: NodeState;
  lockedHint?: string;
}) {
  const pop = usePopOnLit(state);
  return (
    <span
      className={`${styles.node} ${styles[state]} ${pop ? styles.pop : ""}`}
      aria-current={state === "active" ? "step" : undefined}
      title={state === "locked" ? lockedHint : undefined}
    >
      <span className={styles.dot} aria-hidden />
      {label}
      {state === "locked" && lockedHint && <span className="sr-only"> ({lockedHint})</span>}
    </span>
  );
}

function Connector({ state }: { state: NodeState }) {
  // Mirrors the preceding node's own fill amount, so the connector visibly
  // "catches up" to it rather than snapping between dashed/solid outright --
  // width is natively transitionable, unlike swapping gradient values.
  const fillPercent = (LIT_RANK[state] / 2) * 100;
  return (
    <span className={styles.connector} aria-hidden>
      <span className={styles.connectorTrack} />
      <span className={styles.connectorFill} style={{ width: `${fillPercent}%` }} />
      {state === "inProgress" && <span className={styles.connectorShimmer} />}
    </span>
  );
}

/**
 * Three-step route through the product -- Upload, Augment, Recognition --
 * standing in for the old plain-sentence hint. Reuses the wordmark's own
 * hold/route visual language (lit vs. unlit holds, a connector that reads
 * as a little route segment) instead of a generic numbered stepper.
 *
 * Every node starts fully dim (unfilled, no glow) rather than pre-highlighted
 * as "current" -- each one only lights up as a *reaction* to something the
 * user actually did (uploading, finishing augment), with its own pop
 * animation, rather than passively marking position. Augment gets a third,
 * distinct "in progress" look (a half-filled hold) while the user is
 * actively working with an image but hasn't finished augmenting yet.
 */
export function StepIndicator({ uploaded, augmentDone, current, handingOff }: StepIndicatorProps) {
  const uploadState: NodeState = uploaded ? "complete" : "locked";
  const augmentState: NodeState = !uploaded ? "locked" : augmentDone ? "complete" : "inProgress";
  const recognitionState: NodeState = !augmentDone
    ? "locked"
    : current === "recognition"
      ? "active"
      : handingOff
        ? "inProgress"
        : "complete";

  return (
    <div className={styles.pill}>
      <Node label="Upload" state={uploadState} lockedHint="select a wall image first" />
      <Connector state={uploadState} />
      <Node label="Augment" state={augmentState} lockedHint="upload an image to unlock" />
      <Connector state={augmentState} />
      <Node
        label="Recognition"
        state={recognitionState}
        lockedHint="finish augmenting to unlock"
      />
    </div>
  );
}
