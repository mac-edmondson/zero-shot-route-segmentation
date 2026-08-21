"use client";

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import type { Segment } from "@/lib/api";
import { SegmentCard } from "./SegmentCard";
import styles from "./SegmentPanel.module.css";

interface SegmentPanelProps {
  hasImage: boolean;
  segments: Segment[];
  chalkBySegmentId: Record<string, number>;
  onChalkChange: (segmentId: string, value: number) => void;
  onRemove: (segmentId: string) => void;
  /** Hex color sampled from the image for each segment, if any -- see
   * ImageCanvas's colorBySegmentId. */
  colorBySegmentId: Record<string, string>;
  /** segmentId of the card currently waiting for a click on the image to
   * sample from, or null if no pick is in progress. */
  pickingSegmentId: string | null;
  onChooseColor: (segmentId: string) => void;
  /**
   * True once the parent is wrapping up and this panel should play its
   * exit -- reuses the exact same shrink-to-a-hold-then-reform sequence as
   * removing the last segment (the real `segments` data underneath is
   * untouched; only the *display* treats the count as 0 while this is
   * set), then fades the whole panel away once that settles.
   */
  closing?: boolean;
}

type Phase = "empty" | "prompt" | "collapse" | "expand" | "panel" | "shrink" | "reform";
type SequencePhase = "collapse" | "expand" | "shrink" | "reform" | null;

const EMPTY_TEXT = "Upload an image, take a picture, or choose one from your gallery.";
const PROMPT_TEXT = "Select a spot on the image to start augmenting.";

// Forward (0 -> 1 segments): the circle pops in first and is given a
// moment to fully settle -- only *then* do characters start falling into
// it, one after another across STAGGER_SPAN_MS, each taking CHAR_FALL_MS
// to actually disappear. CIRCLE_POP_MS matches morphCircle's own pop-in
// transition below.
const CIRCLE_POP_MS = 220;
const TEXT_START_DELAY_MS = CIRCLE_POP_MS + 60;
const STAGGER_SPAN_MS = 420;
const CHAR_FALL_MS = 560;
const COLLAPSE_MS = TEXT_START_DELAY_MS + STAGGER_SPAN_MS + CHAR_FALL_MS;
const EXPAND_MS = 550;

// Reverse (last segment removed): the box shrinks back to a circle, then
// the text reforms character by character. These must stay in lockstep
// with the matching transition durations in SegmentPanel.module.css
// (.morphShrinking, .charReform) -- these JS timers are what schedule the
// *next* phase, so if they're shorter than the CSS transition they gate,
// the animation gets cut off mid-motion.
const SHRINK_MS = 530;
const REFORM_STAGGER_MS = 365;
const REFORM_CHAR_MS = 530;
const REFORM_MS = REFORM_STAGGER_MS + REFORM_CHAR_MS;

// The empty <-> prompt copy swap: a real overlapping crossfade, not just
// an entrance on the new text -- both play at once for this long.
const TEXT_SWAP_MS = 700;

interface CharOrigin {
  x: number;
  y: number;
}

interface CharToken {
  char: string;
  flatIndex: number;
}

/** Words as non-breaking groups of per-character tokens, so wrapping still only ever happens between words. */
function tokenizeIntoWords(text: string): CharToken[][] {
  const words = text.split(" ");
  const groups: CharToken[][] = [];
  let flatIndex = 0;
  words.forEach((word, wordIndex) => {
    groups.push([...word].map((char) => ({ char, flatIndex: flatIndex++ })));
    if (wordIndex < words.length - 1) {
      groups.push([{ char: " ", flatIndex: flatIndex++ }]);
    }
  });
  return groups;
}

/**
 * Crossfades between two lines of copy -- both the outgoing and incoming
 * text are mounted and animating at once (rise + fade + a touch of blur,
 * unfocus-to-focus) rather than the new text just having its own entrance
 * while the old one gets yanked out instantly.
 */
function SwappableMessage({ text }: { text: string }) {
  const [current, setCurrent] = useState(text);
  const [outgoing, setOutgoing] = useState<string | null>(null);

  // Render-time adjustment (see the sequencePhase kickoff below for why):
  // detecting the copy change here, rather than in an effect, means the
  // very first render with new `text` already has both lines in place --
  // no render where the old line has already vanished before the new one
  // shows up to crossfade against.
  if (text !== current) {
    setOutgoing(current);
    setCurrent(text);
  }

  // Purely a timer, so a plain effect is fine for this part.
  useEffect(() => {
    if (outgoing === null) return;
    const timeout = setTimeout(() => setOutgoing(null), TEXT_SWAP_MS);
    return () => clearTimeout(timeout);
  }, [outgoing]);

  return (
    <div className={styles.messageStack}>
      {outgoing !== null && (
        <p key={`out-${outgoing}`} className={`${styles.message} ${styles.messageStackItem} ${styles.messageOut}`}>
          {outgoing}
        </p>
      )}
      <p key={`in-${current}`} className={`${styles.message} ${styles.messageStackItem} ${styles.messageIn}`}>
        {current}
      </p>
    </div>
  );
}

/**
 * The prompt text, either falling into the hold ("in") or reforming out of
 * it ("out") -- every character moves on its own, each with its own
 * `transform-origin` pointed at the panel's center, so scaling a character
 * to/from 0 around that distant origin makes it shrink-and-travel (or
 * grow-and-arrive) rather than just shrinking or fading in place.
 *
 * Positions are measured once, the first time this ever runs (while the
 * text is normally laid out, safe to measure), and cached in `originsRef`
 * by the parent so a later "out" pass can reuse them directly -- measuring
 * while the characters are already collapsed to scale(0) would just
 * return degenerate zero-size rects in the wrong place.
 */
function DrainingMessage({
  text,
  panelRef,
  originsRef,
  mode,
  active,
}: {
  text: string;
  panelRef: RefObject<HTMLElement | null>;
  originsRef: RefObject<CharOrigin[] | null>;
  mode: "in" | "out";
  active: boolean;
}) {
  const rootRef = useRef<HTMLParagraphElement>(null);
  // Refs can't be read during render, so the cached/measured origins get
  // copied into state here -- useLayoutEffect runs (and, if it triggers a
  // state update, that update also resolves) before the browser paints,
  // so there's no visible flash even though this technically starts null.
  const [origins, setOrigins] = useState<CharOrigin[] | null>(null);
  const groups = useMemo(() => tokenizeIntoWords(text), [text]);
  const charCount = useMemo(() => groups.reduce((sum, group) => sum + group.length, 0), [groups]);

  useLayoutEffect(() => {
    if (originsRef.current) {
      setOrigins(originsRef.current);
      return;
    }
    if (!rootRef.current || !panelRef.current) return;
    const panelRect = panelRef.current.getBoundingClientRect();
    const centerX = panelRect.left + panelRect.width / 2;
    const centerY = panelRect.top + panelRect.height / 2;
    const spans = rootRef.current.querySelectorAll<HTMLElement>("[data-char]");
    const measured = Array.from(spans).map((span) => {
      const rect = span.getBoundingClientRect();
      return { x: centerX - rect.left, y: centerY - rect.top };
    });
    originsRef.current = measured;
    setOrigins(measured);
  }, [panelRef, originsRef]);

  // "in" starts at rest (safe to measure) and only gets the hole class
  // once told to fall; "out" starts already at the hole (continuing from
  // wherever the shrinking box left off) and loses that class once told
  // to reform -- either way, `active` is what's currently transitioning.
  const showHole = mode === "in" ? active : !active;
  const staggerSpan = mode === "in" ? STAGGER_SPAN_MS : REFORM_STAGGER_MS;

  return (
    <p ref={rootRef} className={styles.message} aria-hidden>
      {groups.map((group, groupIndex) => (
        <span key={groupIndex} className={styles.word}>
          {group.map(({ char, flatIndex }) => {
            const origin = origins?.[flatIndex];
            const fraction = charCount > 1 ? flatIndex / (charCount - 1) : 0;
            const delayFraction = mode === "in" ? fraction : 1 - fraction;
            return (
              <span
                key={flatIndex}
                data-char
                className={[
                  styles.char,
                  mode === "out" ? styles.charReform : "",
                  showHole ? styles.charAtHole : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{
                  transformOrigin: origin ? `${origin.x}px ${origin.y}px` : undefined,
                  transitionDelay: active ? `${delayFraction * staggerSpan}ms` : undefined,
                }}
              >
                {char}
              </span>
            );
          })}
        </span>
      ))}
    </p>
  );
}

/**
 * "Selected Segments" -- but there's no box at all until there's something
 * to put in it. Before an image exists it's just centered copy prompting
 * an upload; once an image loads, the copy swaps to prompt a first click
 * (with its own little entrance). The moment that first segment lands, a
 * small hold-colored circle appears behind the text -- like a drain
 * opening -- settles, and *then* the text is pulled into it character by
 * character, after which the circle grows into the actual panel. Removing
 * the last segment plays the same sequence in reverse: the box shrinks
 * back to a circle, then the text reforms out of it. Every other add/
 * remove just updates the list normally.
 */
export function SegmentPanel({
  hasImage,
  segments,
  chalkBySegmentId,
  onChalkChange,
  onRemove,
  colorBySegmentId,
  pickingSegmentId,
  onChooseColor,
  closing = false,
}: SegmentPanelProps) {
  // "empty"/"prompt"/"panel" are plain functions of the props -- no effect
  // needed. Only the transient collapse/expand/shrink/reform sequence is
  // genuinely timer-driven, so it's the only thing that lives in state.
  // `closing` forces the display count to 0 regardless of the real
  // `segments` array -- from this component's point of view that's
  // indistinguishable from the user having removed the last segment
  // themselves, so it naturally falls into the exact same shrink/reform
  // sequence below without needing a separate code path. Both this and
  // `restingPhase` below have to agree on that override -- otherwise the
  // moment the reform sequence finishes, `restingPhase` would fall back
  // to judging the *real* segment count again and pop the panel straight
  // back to "panel" (full circle, real cards) right as it's meant to be
  // settling into the closing fade.
  const currentCount = closing ? 0 : hasImage ? segments.length : 0;
  const restingPhase: Phase = !hasImage ? "empty" : currentCount === 0 ? "prompt" : "panel";
  const [sequencePhase, setSequencePhase] = useState<SequencePhase>(null);
  const [circlePop, setCirclePop] = useState(false);
  const [textActive, setTextActive] = useState(false);
  const panelRef = useRef<HTMLElement>(null);
  const charOriginsRef = useRef<CharOrigin[] | null>(null);

  // Kicked off *during render*, not in an effect -- this is React's own
  // sanctioned pattern for "start a transition the instant a prop
  // changes" (see "Adjusting state when a prop changes" in the React
  // docs). An effect-based version fires after commit, so `phase` would
  // reflect the new resting value on its own (e.g. "prompt", no circle
  // rendered at all) for one real paint before an effect could correct
  // it -- and once the circle's been absent for even one committed
  // render, React treats the next one as a brand-new element with no
  // "before" style to transition from, so it just snaps into place
  // instead of shrinking. Doing it here means the very first render
  // after the prop change already has the right phase, so the circle
  // element itself never actually disappears from the tree.
  const [prevRenderedCount, setPrevRenderedCount] = useState(currentCount);
  // Once any collapse/reform sequence has ever run, the resting "prompt"
  // display switches from SwappableMessage to DrainingMessage's own settled
  // state (see the render below) -- otherwise reform handing off to the
  // resting phase would remount a brand new SwappableMessage, which plays
  // its own fade-in-from-nothing entrance over text that's already fully
  // visible, reading as a stray blink right after the text finishes
  // reforming.
  const [everSequenced, setEverSequenced] = useState(false);
  if (currentCount !== prevRenderedCount) {
    setPrevRenderedCount(currentCount);
    if (prevRenderedCount === 0 && currentCount > 0) {
      setSequencePhase("collapse");
      if (!everSequenced) setEverSequenced(true);
    } else if (prevRenderedCount > 0 && currentCount === 0) {
      setSequencePhase("shrink");
    }
  }

  // Each transient phase schedules its own successor -- purely a timer,
  // so a regular effect (not useLayoutEffect) is fine here.
  useEffect(() => {
    if (sequencePhase === "collapse") {
      const timeout = setTimeout(() => setSequencePhase("expand"), COLLAPSE_MS);
      return () => clearTimeout(timeout);
    }
    if (sequencePhase === "expand") {
      const timeout = setTimeout(() => setSequencePhase(null), EXPAND_MS);
      return () => clearTimeout(timeout);
    }
    if (sequencePhase === "shrink") {
      const timeout = setTimeout(() => setSequencePhase("reform"), SHRINK_MS);
      return () => clearTimeout(timeout);
    }
    if (sequencePhase === "reform") {
      const timeout = setTimeout(() => setSequencePhase(null), REFORM_MS);
      return () => clearTimeout(timeout);
    }
  }, [sequencePhase]);

  const phase: Phase = sequencePhase ?? restingPhase;

  // Only once the shrink/reform sequence has fully settled (sequencePhase
  // back to null) does the panel itself fade away -- waiting for that
  // keeps the two animations sequential rather than the fade racing the
  // reform. If there was nothing to shrink from (already resting empty),
  // sequencePhase is null right away and this fades immediately instead.
  const showClosingFade = closing && sequencePhase === null;

  // The circle mounts at 0x0 first, then -- one paint later -- gets the
  // class that transitions it to visible. Without that gap there's no
  // "before" frame for the browser to animate from; it would just appear
  // already full-size. Only needed for the forward pop-in: shrink/reform
  // reuse an already-mounted, already-painted circle, so a plain class
  // swap there transitions naturally on its own.
  useEffect(() => {
    if (phase !== "collapse") return;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setCirclePop(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      setCirclePop(false);
    };
  }, [phase]);

  // Text only starts moving once told to -- after the circle has settled
  // for "collapse", or right away (just past a paint) for "reform" (the
  // circle's already been sitting there since "shrink" began).
  useEffect(() => {
    if (phase === "collapse") {
      const timeout = setTimeout(() => setTextActive(true), TEXT_START_DELAY_MS);
      return () => {
        clearTimeout(timeout);
        setTextActive(false);
      };
    }
    if (phase === "reform") {
      let raf2 = 0;
      const raf1 = requestAnimationFrame(() => {
        raf2 = requestAnimationFrame(() => setTextActive(true));
      });
      return () => {
        cancelAnimationFrame(raf1);
        cancelAnimationFrame(raf2);
        setTextActive(false);
      };
    }
  }, [phase]);

  const showCircle =
    phase === "collapse" ||
    phase === "expand" ||
    phase === "panel" ||
    phase === "shrink" ||
    phase === "reform";
  const showContent = phase === "expand" || phase === "panel" || phase === "shrink";

  const circleClass = [
    styles.morphCircle,
    (phase === "collapse" && circlePop) || phase === "reform" ? styles.morphVisible : "",
    phase === "expand" || phase === "panel" ? styles.morphExpanded : "",
    phase === "shrink" ? styles.morphShrinking : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <aside
      ref={panelRef}
      className={`${styles.panel} ${showClosingFade ? styles.closing : ""}`}
    >
      {phase === "collapse" || phase === "reform" ? (
        <DrainingMessage
          text={PROMPT_TEXT}
          panelRef={panelRef}
          originsRef={charOriginsRef}
          mode={phase === "collapse" ? "in" : "out"}
          active={textActive}
        />
      ) : phase === "prompt" && everSequenced ? (
        // Resting on the prompt after at least one reform: same component,
        // same settled props as reform's final frame, so this is a no-op
        // re-render rather than a remount -- the text just stays put.
        <DrainingMessage
          text={PROMPT_TEXT}
          panelRef={panelRef}
          originsRef={charOriginsRef}
          mode="out"
          active
        />
      ) : (
        (phase === "empty" || phase === "prompt") && (
          <SwappableMessage text={phase === "empty" ? EMPTY_TEXT : PROMPT_TEXT} />
        )
      )}

      {showCircle && <span className={circleClass} aria-hidden />}

      {showContent && (
        <div className={`${styles.content} ${phase === "panel" ? styles.contentVisible : ""}`}>
          <div className={styles.heading}>
            <h2 className={styles.headingText}>Selected Segments</h2>
            {segments.length > 0 && (
              <span className={`${styles.count} mono`}>{segments.length}</span>
            )}
          </div>
          <div className={styles.list}>
            {segments.map((segment, index) => (
              <SegmentCard
                key={segment.segmentId}
                index={index}
                chalkPercent={chalkBySegmentId[segment.segmentId] ?? 0}
                onChalkChange={(value) => onChalkChange(segment.segmentId, value)}
                onRemove={() => onRemove(segment.segmentId)}
                colorHex={colorBySegmentId[segment.segmentId] ?? null}
                picking={pickingSegmentId === segment.segmentId}
                onChooseColor={() => onChooseColor(segment.segmentId)}
              />
            ))}
          </div>
        </div>
      )}
    </aside>
  );
}
