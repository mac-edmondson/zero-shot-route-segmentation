"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { Wordmark } from "@/components/wordmark/Wordmark";
import styles from "./FlyingLogoLoader.module.css";

/** How long the clone spends flying between the real logo and its resting
 * spot, each direction. Same duration used for both legs so the round trip
 * reads symmetrically. */
const FLIP_MS = 650;
/** var(--ease) (used for the "in" leg) is an ease-out-expo curve -- fast off
 * the start, gently settling at the end, which reads well for something
 * arriving but noticeably rushed-then-crawling for the reverse trip. This is
 * a symmetric ease-in-out instead, gentle at both ends, for the "out" leg --
 * a calmer, evenly-paced departure instead of the arrival curve run
 * backwards. */
const OUT_EASE = "cubic-bezier(0.45, 0, 0.2, 1)";

type Phase = "idle" | "in" | "waiting" | "out";

interface FlyingLogoLoaderProps {
  /** True for as long as whatever this stands in for is in flight. Flying
   * back out is triggered by this going false again -- success or failure
   * alike, since this loader means "still working", not "it worked". */
  active: boolean;
  /** The real logo's own element -- measured as this loader's clone's FLIP
   * "from" rect on the way in, and "to" rect on the way back out. */
  logoRef: RefObject<HTMLElement | null>;
  /** Applied to the outer wrapping element -- this component has no layout
   * opinion of its own (callers put it in very different contexts: a side
   * panel's grid cell vs. an absolute overlay on an image), so positioning
   * is entirely up to whatever className the caller supplies. */
  className?: string;
  label?: string;
  /** Fires once the "out" leg has fully finished and the clone has actually
   * unmounted (phase back to "idle") -- not on the very first idle render.
   * Callers that need to know "is the loader really gone yet" (e.g. to
   * decide when it's safe to show other controls again) should use this
   * rather than inferring it from `active` alone, since the clone is still
   * visibly flying/fading for FLIP_MS after `active` already went false. */
  onExited?: () => void;
}

/**
 * A cloned Wordmark that flies out of the real logo, loops its route-draw
 * for as long as `active` stays true, then flies back and fades into the
 * real logo once it goes false. Shared by every "still working" moment that
 * wants this treatment (Recognition, and ImageCanvas's own loading states)
 * rather than each one building its own copy of the flying/looping logic.
 */
export function FlyingLogoLoader({
  active,
  logoRef,
  className,
  label = "Working in progress…",
  onExited,
}: FlyingLogoLoaderProps) {
  const [phase, setPhase] = useState<Phase>("idle");
  const loaderRef = useRef<HTMLDivElement>(null);
  const fromRectRef = useRef<DOMRect | null>(null);

  // Leg 1 kickoff: the instant `active` turns true (and nothing's already
  // in flight), measure the real logo's current rect and start the "in"
  // leg. The rect itself is captured synchronously here so it can't go
  // stale; only the resulting setPhase is deferred a frame (rAF, same
  // reasoning as showBackToAugment's own reset effect in
  // WallImageWorkspace) -- idle renders nothing, so that one extra frame
  // before the clone appears is invisible, and it keeps this from
  // triggering a synchronous cascading re-render straight out of an effect.
  useLayoutEffect(() => {
    if (!active || phase !== "idle") return;
    fromRectRef.current = logoRef.current?.getBoundingClientRect() ?? null;
    const raf = requestAnimationFrame(() => setPhase("in"));
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active]);

  // Leg 1, the actual FLIP: the clone mounts in its real, final resting
  // position; this offsets it (via transform, before paint) by the delta
  // from fromRectRef's measured rect, so the first painted frame still
  // looks like it's over at the real logo, then clears the offset with a
  // transition on the next frame (double-rAF) -- which is what reads as
  // "flying" from there to here.
  useLayoutEffect(() => {
    if (phase !== "in") return;
    const el = loaderRef.current;
    const fromRect = fromRectRef.current;
    if (!el || !fromRect) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    const toRect = el.getBoundingClientRect();
    const dx = fromRect.left - toRect.left;
    const dy = fromRect.top - toRect.top;

    el.style.transition = "none";
    el.style.transform = `translate(${dx}px, ${dy}px)`;

    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        el.style.transition = `transform ${FLIP_MS}ms var(--ease)`;
        el.style.transform = "translate(0, 0)";
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [phase]);

  // Once the "in" flight's done, settle into "waiting" -- the loop keeps
  // running (it's Wordmark's own CSS animation, not driven from here) for
  // as long as that phase holds.
  useEffect(() => {
    if (phase !== "in") return;
    const timeout = setTimeout(() => setPhase("waiting"), FLIP_MS);
    return () => clearTimeout(timeout);
  }, [phase]);

  // The actual trigger for leg 2.
  useEffect(() => {
    if (!active && phase === "waiting") {
      const raf = requestAnimationFrame(() => setPhase("out"));
      return () => cancelAnimationFrame(raf);
    }
  }, [active, phase]);

  // Leg 2: flies back to wherever the real logo currently is (re-measured
  // now, not reused from leg 1, in case the layout shifted while it was
  // away) and fades out over it, rather than just vanishing -- reads as the
  // clone rejoining/blending into the real logo instead of two separate
  // marks. The layout effect above only ever drives an *entrance* (a
  // translate that decays to rest), so this leg -- a translate that grows
  // away from rest, paired with a fade -- is its own block instead.
  useLayoutEffect(() => {
    if (phase !== "out") return;
    const el = loaderRef.current;
    const toRect = logoRef.current?.getBoundingClientRect();
    if (!el || !toRect || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const raf = requestAnimationFrame(() => setPhase("idle"));
      return () => cancelAnimationFrame(raf);
    }
    const fromRect = el.getBoundingClientRect();
    const dx = toRect.left - fromRect.left;
    const dy = toRect.top - fromRect.top;

    el.style.transition = "none";
    el.style.transform = "translate(0, 0)";
    el.style.opacity = "1";

    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => {
        el.style.transition = `transform ${FLIP_MS}ms ${OUT_EASE}, opacity ${FLIP_MS}ms ${OUT_EASE}`;
        el.style.transform = `translate(${dx}px, ${dy}px)`;
        el.style.opacity = "0";
      });
    });
    const timeout = setTimeout(() => setPhase("idle"), FLIP_MS);
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      clearTimeout(timeout);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase]);

  // Reports "fully gone" to the caller -- fires on mount too (phase starts
  // idle), which is harmless: any caller's own "is it gone" state should
  // already default to true/not-shown before this ever runs once for real.
  useEffect(() => {
    if (phase === "idle") onExited?.();
  }, [phase, onExited]);

  if (phase === "idle") return null;

  return (
    <div className={className}>
      <div ref={loaderRef} className={styles.clone}>
        <Wordmark text="ROUTNet" loop paused={phase === "out"} />
      </div>
      <p className={styles.text}>{label}</p>
    </div>
  );
}
