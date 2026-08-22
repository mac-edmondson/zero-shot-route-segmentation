"use client";

import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, RefObject } from "react";
import { apiClient } from "@/lib/api";
import type { Coordinate, Hold, InferenceResult, RGBColor, Segment } from "@/lib/api";
import { AppHeader } from "@/components/header/AppHeader";
import { Wordmark } from "@/components/wordmark/Wordmark";
import { ImageSourceButtons } from "@/components/image-source-buttons/ImageSourceButtons";
import { GalleryPicker } from "@/components/gallery-picker/GalleryPicker";
import { ImageCanvas } from "@/components/image-canvas/ImageCanvas";
import { SegmentPanel } from "@/components/segment-panel/SegmentPanel";
import { LabeledSlider } from "@/components/labeled-slider/LabeledSlider";
import { Button } from "@/components/button/Button";
import { StepIndicator } from "@/components/step-indicator/StepIndicator";
import { ModelSelect } from "@/components/model-select/ModelSelect";
import styles from "./WallImageWorkspace.module.css";

/**
 * How long SegmentPanel's own closing sequence takes end to end (shrink
 * 530ms + reform ~895ms + its final opacity fade 450ms -- see
 * SegmentPanel.tsx's SHRINK_MS/REFORM_MS and its .closing rule) before the
 * model pickers take its place. Kept as an explicit constant here, in the
 * same spirit as this codebase's other cross-timing comments, because
 * there's no way to observe "SegmentPanel's animation finished" from
 * outside it -- if those constants change, this needs to change with them.
 */
const MODEL_SELECT_REVEAL_MS = 1875;

/**
 * Same wait as MODEL_SELECT_REVEAL_MS above, but for going straight to
 * Finish Augment with zero segments marked (hold detection skipped
 * entirely). SegmentPanel's shrink/reform sequence only ever plays when
 * there were segments to animate away -- with none, `closing` just plays
 * its own plain opacity/filter fade (see SegmentPanel.module.css's
 * `.closing`, 450ms) instead, so waiting the full 1875ms here left the
 * model pickers appearing well after that fade had already finished --
 * the "bit late" this constant fixes.
 */
const MODEL_SELECT_REVEAL_MS_NO_SEGMENTS = 450;

/** Must match .modelSelectPanel's own transition-duration in
 * WallImageWorkspace.module.css -- handleGoToRecognition plays that same
 * leave transition (the same way handleBackToAugment already does) before
 * the routes panel takes its place in the same grid cell. */
const MODEL_SELECT_LEAVE_MS = 550;

/** How long the recognition loader's clone spends flying between the real
 * logo and its resting spot, each direction -- see loaderPhase. Same
 * duration used for both legs (in via useFlipSlide, out via the dedicated
 * effect below) so the round trip reads symmetrically. */
const LOADER_FLIP_MS = 650;
/** var(--ease) (used for the "in" leg, via useFlipSlide) is an ease-out-expo
 * curve -- fast off the start, gently settling at the end, which reads well
 * for something arriving but noticeably rushed-then-crawling for the
 * reverse trip. This is a symmetric ease-in-out instead, gentle at both
 * ends, for the "out" leg's own effect below -- a calmer, evenly-paced
 * "smooth" departure instead of the arrival curve run backwards. */
const LOADER_OUT_EASE = "cubic-bezier(0.45, 0, 0.2, 1)";

/** Per-hold reveal stagger, bottom-first (see holdsByRoute below). Purely a
 * CSS animation-delay multiplier -- no JS timer depends on it. */
const ROUTE_HOLD_REVEAL_STAGGER_MS = 70;
/** Must match .highlightHoldLeave's animation-duration in
 * WallImageWorkspace.module.css -- the one JS timer in the route-selection
 * state machine below, which needs to know when the CSS fade-out actually
 * finishes before swapping in the next route's holds. */
const ROUTE_HOLD_LEAVE_MS = 220;

// Same palette RouteDiscriminator.mark_routes uses server-side
// (src/pipeline/route_discriminator/route_discriminator.py) -- so a route's
// color here matches what a rendered overlay image would eventually use.
const ROUTE_COLORS = ["#ff5000", "#00b4ff", "#b450ff", "#50dc50", "#ffc800", "#ff50b4"];

type RouteHighlightPhase = "idle" | "active" | "exiting";

function createLocalImageId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2);
}

function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** Converts a "#rrggbb" hex color (as sampled by ImageCanvas's eyedropper)
 * into the RGBColor shape the augmentation API expects. */
function hexToRgb(hex: string | undefined): RGBColor | undefined {
  const match = hex ? /^#?([0-9a-f]{6})$/i.exec(hex) : null;
  if (!match) return undefined;
  const value = parseInt(match[1], 16);
  return { r: (value >> 16) & 0xff, g: (value >> 8) & 0xff, b: value & 0xff };
}

/**
 * FLIP animation (First/Last/Invert/Play) -- used twice below, once for
 * each direction of the model-select <-> locked-chips slide, hence
 * factored out rather than inlined. `el` mounts directly in its real,
 * final position; this offsets it (via `transform`, before paint, hence
 * useLayoutEffect) by the delta from `fromRectRef`'s last-measured rect,
 * so the first painted frame still looks like it's over there, then clears
 * the offset with a transition on the next frame (double-rAF, this
 * codebase's usual entrance-animation trick) -- which is what reads as
 * "sliding" from the old spot to the new one. Only ever animates
 * `transform` -- cheap, GPU-composited, same lesson as the route-holds
 * overlay's earlier flicker fix. A null `fromRectRef.current` (e.g. the
 * model-select row's very first appearance, which isn't a slide from
 * anywhere) makes this a no-op, leaving whatever other entrance transition
 * the element already has to run on its own.
 */
function useFlipSlide(
  active: boolean,
  elRef: RefObject<HTMLElement | null>,
  fromRectRef: RefObject<DOMRect | null>,
  durationMs: number,
) {
  useLayoutEffect(() => {
    if (!active) return;
    const el = elRef.current;
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
        el.style.transition = `transform ${durationMs}ms var(--ease)`;
        el.style.transform = "translate(0, 0)";
      });
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [active, elRef, fromRectRef, durationMs]);
}

/**
 * The ROUTNet landing page (Project stuff/UI_page_1.png): pick or capture a
 * wall image, mark hold segments, adjust lighting/chalk, then run
 * recognition on it -- all as phases of this one component/page, never a
 * route change. That's deliberate, not an oversight: recognition needs the
 * augmented image to keep showing in exactly the same spot, at exactly the
 * same size, with no reload -- which a page navigation (even client-side)
 * can't guarantee, since it unmounts/remounts the tree. ImageCanvas is
 * mounted exactly once, for this component's whole lifetime; only its
 * props (and the side panel next to it) change as the phase advances.
 * Talks only to `apiClient` (`@/lib/api`), which is backed by an in-memory
 * mock until the real backend (docs/spec/pipeline/interfaces/dashboard-backend.md)
 * exists.
 */
export function WallImageWorkspace() {
  const [imageId, setImageId] = useState<string | null>(null);
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  // The true pre-augmentation upload, kept aside so handleBackToAugment can
  // restore it -- `imageSrc` itself gets overwritten with the real
  // *augmented* image once Finish Augment lands (see handleFinishAugment),
  // which is correct while looking at Recognition, but going back to edit
  // needs the original back underneath: ImageCanvas's live lighting/chalk/
  // color preview overlays are computed as if `imageSrc` were still the
  // unaugmented base, so previewing again on top of the already-baked
  // result would double the effect (and the slider could never visually
  // get back to "no change" -- exactly the bug this fixes -- since 50%
  // stops meaning "identical to what's showing" the moment what's showing
  // is the augmented image instead of the original one).
  const originalImageSrcRef = useRef<string | null>(null);
  const [segments, setSegments] = useState<Segment[]>([]);
  /** Points clicked but not yet sent for detection. */
  const [pendingPoints, setPendingPoints] = useState<Coordinate[]>([]);
  const [detecting, setDetecting] = useState(false);
  const [chalkBySegmentId, setChalkBySegmentId] = useState<Record<string, number>>({});
  // Hex color sampled from the image for each segment (SegmentCard's
  // "Choose color" eyedropper) -- feeds into buildAugmentationPayload below
  // as that segment's augmentation color, alongside its chalk percent.
  const [colorBySegmentId, setColorBySegmentId] = useState<Record<string, string>>({});
  // segmentId of the card currently waiting on a click on the image to
  // sample from, or null if no pick is in progress.
  const [pickingColorSegmentId, setPickingColorSegmentId] = useState<string | null>(null);
  // -1 to 1, 0 = the slider's midpoint = the original, unmodified image --
  // same scale and neutral point as the backend's own
  // LightingAugmentationParams.intensity (src/pipeline/interfaces/
  // augmentation.py), sent to it as-is (see buildAugmentationPayload) with
  // no conversion needed either direction. ImageCanvas's own lightingPercent
  // prop is a different, older 0-100/50-neutral scale it was already built
  // around -- converted to that at the callsite below rather than changing
  // ImageCanvas itself, so this is the only place that scale switch exists.
  const [lightingIntensity, setLightingIntensity] = useState(0);
  const [webcamActive, setWebcamActive] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [gallerySelecting, setGallerySelecting] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [augmentDone, setAugmentDone] = useState(false);
  const [toolbarEntered, setToolbarEntered] = useState(false);
  // Detect Holds/Finish Augment and the Lighting slider's own entrance --
  // separate from toolbarEntered above since that one only ever fires
  // once, right at mount, while these two first mount later (once an
  // image is loaded, possibly long after) and need their own "before"
  // frame to animate from each time they (re)appear.
  const [actionsEntered, setActionsEntered] = useState(false);
  // The small standalone "Back to Augment" that takes the toolbar's place,
  // top-right, once augmentDone hides it -- see showBackToAugment below.
  const [backToAugmentEntered, setBackToAugmentEntered] = useState(false);
  const [showModelSelect, setShowModelSelect] = useState(false);
  const [modelSelectEntered, setModelSelectEntered] = useState(false);
  const [holdModel, setHoldModel] = useState<string | null>(null);
  const [routeModel, setRouteModel] = useState<string | null>(null);
  // The two model-select dropdowns' own option lists -- fetched once from
  // GET /pipeline/available_configs rather than hardcoded here, so a newly
  // registered pipeline method (src/pipeline/*/*_factory.py) shows up
  // without a frontend deploy. Empty until that first fetch resolves.
  const [holdModelOptions, setHoldModelOptions] = useState<string[]>([]);
  const [routeModelOptions, setRouteModelOptions] = useState<string[]>([]);
  const [inferring, setInferring] = useState(false);

  // --- Recognition loader -----------------------------------------------
  // A cloned Wordmark that flies out of the real logo up in the header,
  // down into the spot where the routes list is about to appear, loops its
  // route-draw there for as long as inference is running (Wordmark's
  // `loop` prop), then flies back and fades into the real logo once the
  // backend responds -- success or failure alike, since this is purely
  // "still working" -> "done working", not a result indicator itself.
  // "idle": not shown. "in": flying from the header to its resting spot
  // (useFlipSlide below drives this leg). "waiting": resting, looping.
  // "out": flying back to the header and fading (a dedicated effect below
  // drives this leg -- useFlipSlide only ever animates an *entrance*).
  const [loaderPhase, setLoaderPhase] = useState<"idle" | "in" | "waiting" | "out">("idle");
  const logoRef = useRef<HTMLHeadingElement>(null);
  const loaderRef = useRef<HTMLDivElement>(null);
  const loaderFromRectRef = useRef<DOMRect | null>(null);
  // ------------------------------------------------------------------------

  // Recognition results -- another phase of this same page, not a route
  // (see the component docstring above).
  const [recognitionResult, setRecognitionResult] = useState<InferenceResult | null>(null);
  const [recognitionDone, setRecognitionDone] = useState(false);
  const [routesPanelEntered, setRoutesPanelEntered] = useState(false);
  // Distinct from recognitionDone: this one stays true through
  // handleChangeModel (which flips recognitionDone back to false to hide
  // the routes panel and let the models be re-picked) -- the image itself
  // should stay in its clean, non-interactive recognition look while
  // re-picking models, not revert to click-to-add-point editing just
  // because the routes panel is temporarily hidden. Only a real "Back to
  // Augment" clears it.
  const [pastRecognition, setPastRecognition] = useState(false);

  // Locked-model slide: once Recognition succeeds, the two chosen models
  // stop being editable and slide from the model-select row (right of the
  // image) up into the toolbar's own reserved spot (above the image, below
  // the header) -- see useFlipSlide above. "Change model" (handleChangeModel)
  // plays the same slide in reverse. modelSelectRowRef/lockedModelsRef are
  // each one direction's "to" element (mounted in its own real, final
  // position); modelSlideFromRectRef/modelSlideBackFromRectRef hold the
  // *other* element's last-measured rect, i.e. where each slide starts from.
  const modelSelectRowRef = useRef<HTMLDivElement>(null);
  const lockedModelsRef = useRef<HTMLDivElement>(null);
  const modelSlideFromRectRef = useRef<DOMRect | null>(null);
  const modelSlideBackFromRectRef = useRef<DOMRect | null>(null);
  const [showLockedModels, setShowLockedModels] = useState(false);

  // Set by handleFinishAugment, read by the model-select reveal effect
  // below -- whether there were any segments to shrink/reform away at the
  // moment Finish Augment was pressed, which is what picks between
  // MODEL_SELECT_REVEAL_MS and its _NO_SEGMENTS counterpart. A ref, not a
  // dependency read off `segments` directly, so this stays pinned to that
  // one moment rather than drifting if `segments` were ever to change
  // again before the timeout fires.
  const hadSegmentsOnFinishRef = useRef(false);

  // Which route's holds are currently drawn on the image. selectedRouteId
  // stays set through "exiting" (it hasn't been replaced yet at that
  // point); queuedRouteId is only meaningful while exiting: null means "go
  // to idle once the fade finishes", a routeId means "switch to that
  // route's holds once the fade finishes". selectedHoldIndex drills into
  // one specific hold within the currently-shown route (see selectHold).
  const [selectedRouteId, setSelectedRouteId] = useState<number | null>(null);
  const [routeHighlightPhase, setRouteHighlightPhase] = useState<RouteHighlightPhase>("idle");
  const [queuedRouteId, setQueuedRouteId] = useState<number | null>(null);
  const [selectedHoldIndex, setSelectedHoldIndex] = useState<number | null>(null);

  // Fetched once on mount -- well before the user could ever reach the
  // model-select step -- rather than on-demand when that step first shows,
  // so the dropdowns already have their options the instant they appear
  // instead of opening on an empty list and populating a beat later.
  useEffect(() => {
    const controller = new AbortController();
    apiClient
      .getAvailableConfigs(controller.signal)
      .then(({ holdDetector, routeClassifier }) => {
        setHoldModelOptions(holdDetector);
        setRouteModelOptions(routeClassifier);
      })
      .catch(() => {
        // See GalleryPicker's identical categories effect for why this
        // checks our own controller instead of the caught error.
        if (controller.signal.aborted) return;
        setError("Couldn't load the available models. Try again.");
      });
    return () => controller.abort();
  }, []);

  // Mount, wait a paint, then trigger -- without the gap there's no
  // "before" frame for the browser to animate from, so the toolbar's two
  // pieces would just appear already in place instead of rising in.
  useEffect(() => {
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setToolbarEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, []);

  // Detect Holds/Finish Augment and the Lighting slider only ever make
  // sense once there's an image to point at/adjust -- hidden entirely
  // (not just disabled) until then, with their own "wait a paint, then
  // trigger" entrance each time imageId goes from unset to set, same
  // reasoning as the toolbar's own mount effect above but re-armed on
  // this narrower trigger instead of firing once.
  useEffect(() => {
    if (!imageId) {
      // Deferred (not called synchronously in the effect body) same as the
      // "arm" branch below -- doesn't need to be immediate, since this
      // group is already unmounted the instant imageId clears (see the
      // {imageId && ...} guard around it); this just resets the flag so a
      // *later* reappearance gets a real entrance again instead of
      // snapping straight to "in".
      const raf = requestAnimationFrame(() => setActionsEntered(false));
      return () => cancelAnimationFrame(raf);
    }
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setActionsEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [imageId]);

  // Full-width canvas for as long as there's nothing real for the segment
  // sidebar to show yet -- SegmentPanel's own "empty" (no image) and
  // "prompt" (image, no segments) phases are just placeholder copy
  // ("Upload an image..."/"Select a spot..."), so the image (or its own
  // empty-state placeholder) gets that space instead, from first paint
  // right up until Detect Holds lands a real segment. .mainGrid's own
  // comment on why its tracks are otherwise fixed-size still holds for
  // every stage after this one, this is a deliberate, narrowly-scoped
  // exception to it, not a rule change. augmentDone excluded too: once
  // Finish Augment is pressed with zero segments (skipping hold-marking
  // entirely), the model-select/recognition stages afterward should still
  // look like they always have, not full-width.
  const showFullWidthCanvas = segments.length === 0 && !augmentDone;

  // The standalone "Back to Augment" that stands in for the toolbar, top
  // right, for exactly the model-select gap: augmentDone hides the toolbar
  // (and, with it, the "Back to Augment" that used to live inside it) the
  // instant Finish Augment is pressed, but showLockedModels' own "Change
  // model" bar doesn't take over that spot until Recognition actually
  // succeeds. Without this, there'd be no way back to Augment in between.
  const showBackToAugment = augmentDone && !showLockedModels;

  useEffect(() => {
    if (!showBackToAugment) {
      // Deferred for the same reason as actionsEntered's reset above --
      // this slot is already unmounted by the time this branch runs, so
      // only a *later* reappearance (e.g. showing again after "Change
      // model" un-shows the locked-model chips) is what actually needs
      // this flag back at false.
      const raf = requestAnimationFrame(() => setBackToAugmentEntered(false));
      return () => cancelAnimationFrame(raf);
    }
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setBackToAugmentEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [showBackToAugment]);

  // Once the toolbar and segment panel have both fully finished leaving,
  // the model pickers take over that same area -- mounted only then (not
  // shown-but-invisible from the start), so the same "wait a paint, then
  // trigger" entrance below has a real "before" frame to animate from.
  useEffect(() => {
    if (!augmentDone) return;
    const revealDelay = hadSegmentsOnFinishRef.current
      ? MODEL_SELECT_REVEAL_MS
      : MODEL_SELECT_REVEAL_MS_NO_SEGMENTS;
    const revealTimeout = setTimeout(() => setShowModelSelect(true), revealDelay);
    return () => clearTimeout(revealTimeout);
  }, [augmentDone]);

  useEffect(() => {
    if (!showModelSelect) return;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setModelSelectEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [showModelSelect]);

  // Same "wait a paint, then trigger" entrance for the routes panel, once
  // it's mounted (handleGoToRecognition below only flips recognitionDone
  // after the model pickers have fully finished leaving).
  useEffect(() => {
    if (!recognitionDone) return;
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setRoutesPanelEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [recognitionDone]);

  // Forward slide (model-select row -> locked chips, on Recognition) and
  // reverse slide (locked chips -> model-select row, on "Change model") --
  // see useFlipSlide above. The reverse one is a no-op on the model-select
  // row's very first appearance (modelSlideBackFromRectRef.current is only
  // ever set by handleChangeModel below), so its own ordinary rise/fade
  // entrance (the effect above this one) is untouched for that case.
  useFlipSlide(showLockedModels, lockedModelsRef, modelSlideFromRectRef, MODEL_SELECT_LEAVE_MS);
  useFlipSlide(showModelSelect, modelSelectRowRef, modelSlideBackFromRectRef, MODEL_SELECT_LEAVE_MS);

  // Recognition loader, leg 1: flies in from the real logo (loaderFromRectRef
  // is measured in handleGoToRecognition, right when loaderPhase is first
  // set to "in", the same way modelSlideFromRectRef etc. are measured right
  // before their own FLIP-driving state flips). Once the flight's done,
  // settle into "waiting" -- the loop keeps running (it's Wordmark's own
  // CSS animation, not driven from here) for as long as that phase holds.
  useFlipSlide(loaderPhase === "in", loaderRef, loaderFromRectRef, LOADER_FLIP_MS);
  useEffect(() => {
    if (loaderPhase !== "in") return;
    const timeout = setTimeout(() => setLoaderPhase("waiting"), LOADER_FLIP_MS);
    return () => clearTimeout(timeout);
  }, [loaderPhase]);

  // Recognition loader, leg 2: flies back to wherever the real logo
  // currently is (re-measured now, not reused from leg 1, in case the
  // layout shifted while it was away) and fades out over it, rather than
  // just vanishing -- reads as the clone rejoining/blending into the real
  // logo instead of two separate marks. useFlipSlide only ever drives an
  // *entrance* (a translate that decays to rest), so this leg -- a
  // translate that grows away from rest, paired with a fade -- is its own
  // effect instead of a second useFlipSlide call.
  useLayoutEffect(() => {
    if (loaderPhase !== "out") return;
    const el = loaderRef.current;
    const toRect = logoRef.current?.getBoundingClientRect();
    if (!el || !toRect || window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      // Deferred, not called inline here, same reasoning as
      // showBackToAugment's own reset effect further up.
      const raf = requestAnimationFrame(() => setLoaderPhase("idle"));
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
        el.style.transition = `transform ${LOADER_FLIP_MS}ms ${LOADER_OUT_EASE}, opacity ${LOADER_FLIP_MS}ms ${LOADER_OUT_EASE}`;
        el.style.transform = `translate(${dx}px, ${dy}px)`;
        el.style.opacity = "0";
      });
    });
    const timeout = setTimeout(() => setLoaderPhase("idle"), LOADER_FLIP_MS);
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
      clearTimeout(timeout);
    };
  }, [loaderPhase]);

  // The actual trigger for leg 2 -- inferring's own true -> false edge,
  // which fires identically on success or failure (handleGoToRecognition's
  // finally), since this loader means "still working", not "it worked".
  useEffect(() => {
    if (!inferring && loaderPhase === "waiting") {
      const raf = requestAnimationFrame(() => setLoaderPhase("out"));
      return () => cancelAnimationFrame(raf);
    }
  }, [inferring, loaderPhase]);

  // Appears only once BOTH dropdowns have a pick -- unlike the entrance
  // timing above, this one can't be "either", because the Recognition call
  // itself (handleGoToRecognition below) needs both to build a real
  // PipelineConfig: the pipeline requires a hold_detector AND a
  // route_discriminator to run at all.
  const showRecognitionButton = holdModel !== null && routeModel !== null;
  const [recognitionEntered, setRecognitionEntered] = useState(false);

  // Also keyed on showModelSelect (not just showRecognitionButton) so
  // "Change model" re-arms this entrance too. handleChangeModel doesn't
  // clear holdModel/routeModel, so showRecognitionButton is already true
  // the moment the panel remounts -- without showModelSelect in the deps,
  // this effect wouldn't rerun on that round trip, recognitionEntered would
  // stay stale-true from the first time around, and the button would render
  // already fully "in" while modelSelectRow is still mid-FLIP-slide back
  // from the locked chips, i.e. it'd appear before the dropdown ever shows.
  useEffect(() => {
    if (!showRecognitionButton || !showModelSelect) {
      setRecognitionEntered(false);
      return;
    }
    let raf2 = 0;
    const raf1 = requestAnimationFrame(() => {
      raf2 = requestAnimationFrame(() => setRecognitionEntered(true));
    });
    return () => {
      cancelAnimationFrame(raf1);
      cancelAnimationFrame(raf2);
    };
  }, [showRecognitionButton, showModelSelect]);

  const routes = useMemo(() => recognitionResult?.routes ?? [], [recognitionResult]);

  // Bottom-first per route: normalized y grows downward, so the largest y
  // is the lowest hold on the image -- "the bottom hold should appear on
  // the image 1st", per how this was asked for. Also determines the
  // "Hold 1"/"Hold 2"/... numbering in the per-route hold buttons, so
  // "Hold 1" is always the one that reveals first.
  const holdsByRoute = useMemo(() => {
    const map = new Map<number, Hold[]>();
    for (const route of routes) {
      map.set(route.routeId, [...route.holds].sort((a, b) => b.centroid.y - a.centroid.y));
    }
    return map;
  }, [routes]);

  /** Same color a route's card/swatch uses (ROUTE_COLORS, cycled by its
   * position in the list) -- looked up by id rather than index so the
   * image overlay (which only knows selectedRouteId) can match whichever
   * card is actually highlighted. */
  function colorForRoute(routeId: number): string {
    const index = routes.findIndex((route) => route.routeId === routeId);
    return ROUTE_COLORS[(index < 0 ? 0 : index) % ROUTE_COLORS.length];
  }

  function selectRoute(routeId: number) {
    if (routeHighlightPhase === "idle") {
      setSelectedRouteId(routeId);
      setRouteHighlightPhase("active");
      return;
    }
    if (routeHighlightPhase === "active") {
      // Clicking the already-selected card again hides it (queuedRouteId
      // stays null); clicking a different one queues the switch. Either
      // way, the currently-shown holds have to vanish first.
      setQueuedRouteId(routeId === selectedRouteId ? null : routeId);
      setRouteHighlightPhase("exiting");
      return;
    }
    // Already exiting -- just update what happens once that finishes,
    // rather than starting a second overlapping exit.
    setQueuedRouteId(routeId === selectedRouteId ? null : routeId);
  }

  /** Drills into one specific hold within the currently-shown route --
   * click again to clear it. Purely a re-color/emphasis on the image (see
   * .highlightHoldFocused); it doesn't affect which holds are visible. */
  function selectHold(index: number) {
    setSelectedHoldIndex((prev) => (prev === index ? null : index));
  }

  useEffect(() => {
    if (routeHighlightPhase !== "exiting") return;
    const timeout = setTimeout(() => {
      setSelectedHoldIndex(null);
      if (queuedRouteId != null) {
        setSelectedRouteId(queuedRouteId);
        setQueuedRouteId(null);
        setRouteHighlightPhase("active");
      } else {
        setSelectedRouteId(null);
        setRouteHighlightPhase("idle");
      }
    }, ROUTE_HOLD_LEAVE_MS);
    return () => clearTimeout(timeout);
  }, [routeHighlightPhase, queuedRouteId]);

  const visibleHolds = useMemo(
    () =>
      routeHighlightPhase !== "idle" && selectedRouteId != null
        ? (holdsByRoute.get(selectedRouteId) ?? [])
        : [],
    [routeHighlightPhase, selectedRouteId, holdsByRoute],
  );

  // Same issue, same fix as ImageCanvas's polygonPointsById -- Mask-RCNN's
  // contours are effectively pixel-precise (hundreds to low-thousands of
  // points per hold), so serializing one into SVG's `points` format is
  // real work. Left inline in JSX, that work reran for every hold, on
  // every render of this whole component -- not just the one render where
  // the highlight actually mounts, but every later one too (hovering a
  // hold in the list, selecting a different one, anything else in this
  // fairly large component that triggers a re-render). With a wall that
  // has enough holds, that recurring cost is exactly what shows up as a
  // visible stutter/"blink" partway through the staggered reveal below.
  // Memoized on `visibleHolds` itself (a stable reference from the
  // memoized holdsByRoute map, unless the route selection actually
  // changes) so this only redoes the work when the visible set of holds
  // actually changes.
  const visibleHoldPoints = useMemo(
    () => visibleHolds.map((hold) => hold.polygon.points.map((p) => `${p.x},${p.y}`).join(" ")),
    [visibleHolds],
  );

  const loadImage = useCallback(async (file: File | Blob) => {
    setLoading(true);
    setError(null);
    try {
      // Render is purely local -- decoding `file` into a data URL never
      // touches the backend, so it can't fail because of it.
      const dataUrl = await fileToDataUrl(file);

      // Best-effort: also upload the file as the working image
      // (PUT /image/working) so the backend session actually has it -- this
      // must not block the local render. `PUT /image/working` returns no
      // body (it's just a session-side reference), so the id shown here is
      // always a local one purely for UI gating; it never has to match
      // anything server-side, since later steps identify the working image
      // implicitly, through the session, rather than by this id.
      try {
        await apiClient.setWorkingImage(file);
      } catch (err) {
        console.warn("setWorkingImage failed; continuing with a local image id", err);
      }

      setImageId(createLocalImageId());
      setImageSrc(dataUrl);
      originalImageSrcRef.current = dataUrl;
      setSegments([]);
      setPendingPoints([]);
      setChalkBySegmentId({});
      setColorBySegmentId({});
      setPickingColorSegmentId(null);
      setWebcamActive(false);
      setAugmentDone(false);
      setShowModelSelect(false);
      setModelSelectEntered(false);
      setHoldModel(null);
      setRouteModel(null);
      setRecognitionResult(null);
      setRecognitionDone(false);
      setPastRecognition(false);
      setRoutesPanelEntered(false);
      setShowLockedModels(false);
      modelSlideBackFromRectRef.current = null;
      setSelectedRouteId(null);
      setRouteHighlightPhase("idle");
      setQueuedRouteId(null);
      setSelectedHoldIndex(null);
    } catch {
      setError("Couldn't load that image. Try again.");
    } finally {
      setLoading(false);
    }
  }, []);

  async function handleGallerySelect(category: string, name: string) {
    setGallerySelecting(true);
    setError(null);
    try {
      const blob = await apiClient.fetchGalleryImage(category, name);
      await loadImage(blob);
      setGalleryOpen(false);
    } catch {
      setError("Couldn't load that gallery image. Try again.");
    } finally {
      setGallerySelecting(false);
    }
  }

  function handleAddSegmentPoint(coordinate: Coordinate) {
    if (!imageId || loading || detecting) return;
    setPendingPoints((prev) => [...prev, coordinate]);
  }

  async function handleDetectSegments() {
    if (!imageId || pendingPoints.length === 0) return;
    setDetecting(true);
    setError(null);
    try {
      const detected = await apiClient.detectWorkingSegments(pendingPoints);
      setSegments((prev) => [...prev, ...detected]);
      setChalkBySegmentId((prev) => {
        const next = { ...prev };
        for (const segment of detected) next[segment.segmentId] = 0;
        return next;
      });
      setPendingPoints([]);
    } catch {
      setError("Couldn't detect holds there. Try again.");
    } finally {
      setDetecting(false);
    }
  }

  function handleChalkChange(segmentId: string, value: number) {
    setChalkBySegmentId((prev) => ({ ...prev, [segmentId]: value }));
  }

  // Toggling the same card's "Choose color" again cancels the pick instead
  // of restarting it; choosing a different card just moves the pick over.
  function handleChooseColor(segmentId: string) {
    setPickingColorSegmentId((prev) => (prev === segmentId ? null : segmentId));
  }

  // ImageCanvas's onPickColor -- fires once the user clicks the image while
  // a pick is in progress.
  function handlePickColor(color: string) {
    if (!pickingColorSegmentId) return;
    setColorBySegmentId((prev) => ({ ...prev, [pickingColorSegmentId]: color }));
    setPickingColorSegmentId(null);
  }

  // Escape backs out of a color pick without sampling anything, same as it
  // closes a ModelSelect dropdown.
  useEffect(() => {
    if (!pickingColorSegmentId) return;
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") setPickingColorSegmentId(null);
    }
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [pickingColorSegmentId]);

  async function handleRemoveSegment(segmentId: string) {
    setSegments((prev) => prev.filter((segment) => segment.segmentId !== segmentId));
    setChalkBySegmentId((prev) => {
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
    setColorBySegmentId((prev) => {
      const next = { ...prev };
      delete next[segmentId];
      return next;
    });
    setPickingColorSegmentId((prev) => (prev === segmentId ? null : prev));
    try {
      await apiClient.deleteWorkingSegment(segmentId);
    } catch {
      setError("Segment removed locally, but the backend couldn't confirm it.");
    }
  }

  // Used by handleFinishAugment to snapshot the current lighting/chalk/color
  // state into the shape POST /image/working/augment expects. Recognition
  // no longer needs this itself -- the backend session keeps the augmented
  // working image between requests, so inferWorkingPipeline just runs
  // against whatever Finish Augment last left there.
  function buildAugmentationPayload() {
    return {
      // lightingIntensity already lives on the exact -1 to 1 scale the
      // backend's AugmentWorkingImageRequest.lighting_percent expects (see
      // src/backend/schemas.py) and passes straight through, unconverted,
      // into LightingAugmentationParams.intensity (src/pipeline/interfaces/
      // augmentation.py) -- no more percent-scale round-trip in between.
      lightingPercent: lightingIntensity,
      segments: segments.map((segment) => ({
        segmentId: segment.segmentId,
        chalkPercent: chalkBySegmentId[segment.segmentId] ?? 0,
        color: hexToRgb(colorBySegmentId[segment.segmentId]),
      })),
    };
  }

  async function handleFinishAugment() {
    if (!imageId) {
      setError("Select a wall image first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const augmented = await apiClient.augmentWorkingImage(buildAugmentationPayload());
      // From here on (model-select, Recognition) the canvas shows the real
      // augmented image the backend just produced -- and, critically, the
      // exact same image /pipeline/infer/working runs against -- rather
      // than continuing to show the original upload with lighting/chalk/
      // color only *simulated* on top via ImageCanvas's CSS filter and SVG
      // overlays. Those live-preview props are neutralized below once
      // augmentDone flips, so this doesn't end up double-applying the
      // effect on top of pixels that already have it baked in.
      if (augmented.image) {
        setImageSrc(augmented.image);
      }
      // Stays on this same page -- no navigation. The step indicator shifts
      // to Recognition (shown as in-progress, not complete -- see
      // `handingOff` below) and the toolbar/segment panel fade out; nothing
      // else moves or resizes.
      hadSegmentsOnFinishRef.current = segments.length > 0;
      setAugmentDone(true);
    } catch {
      setError("Couldn't finish augmentation. Try again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleGoToRecognition() {
    if (!imageId || !holdModel || !routeModel) return;
    // Measured now, before anything below starts moving -- this is the
    // recognition loader's own FLIP "from" rect (see loaderPhase and
    // useFlipSlide above), same pattern as modelSlideFromRectRef further
    // down: capture the real logo's current position before the state
    // flip that triggers the clone's entrance.
    loaderFromRectRef.current = logoRef.current?.getBoundingClientRect() ?? null;
    setLoaderPhase("in");
    setInferring(true);
    setError(null);
    try {
      // Set the selected models server-side first (PUT /pipeline), then
      // start inference with no body -- it runs against whatever working
      // image + augmentation + config the session already has.
      await apiClient.setPipeline({ holdDetector: holdModel, routeClassifier: routeModel });
      // Run alongside inference, not after it -- GET /image/working doesn't
      // depend on the inference result, it's just the same "what does the
      // backend currently have as the working image" fetch handleFinishAugment
      // already uses its own augmentWorkingImage response for. That earlier
      // swap covers the common case already, but the route/hold overlay
      // shown here should always be paired with an image explicitly
      // confirmed from the backend at this exact moment, via the actual
      // GET /image/working endpoint, rather than only ever trusting a
      // snapshot captured back when Finish Augment ran.
      const [result, workingImage] = await Promise.all([
        apiClient.inferWorkingPipeline(),
        apiClient.getWorkingImage(),
      ]);
      setRecognitionResult(result);
      if (workingImage.image) {
        setImageSrc(workingImage.image);
      }
      setPastRecognition(true);
      // Measured now, while the row is still on-screen at its normal
      // position -- this is the FLIP slide's "from" rect (see
      // useFlipSlide). Must happen before anything below starts that row
      // fading out.
      modelSlideFromRectRef.current = modelSelectRowRef.current?.getBoundingClientRect() ?? null;
      setShowLockedModels(true);
      // Same "play the leave transition, then swap what's mounted"
      // technique handleBackToAugment already uses below for
      // modelSelectPanel -- ImageCanvas itself never unmounts through any
      // of this (see the component docstring), so the image can't
      // reload/resize/jump; only this side panel's contents change, and
      // only the polygon/click-outline overlay on the image (showHoldOutline
      // below) goes away.
      setModelSelectEntered(false);
      setTimeout(() => {
        setShowModelSelect(false);
        setRecognitionDone(true);
      }, MODEL_SELECT_LEAVE_MS);
    } catch {
      setError("Couldn't run recognition. Try again.");
    } finally {
      setInferring(false);
    }
  }

  // A smaller reversal than handleBackToAugment below: only undoes the
  // "lock" step, not the whole augment->recognition arc. The routes panel
  // fades out, the locked chips vanish, and the model-select row reappears
  // -- sliding in FROM where the chips just were (useFlipSlide's reverse
  // direction, mirroring handleGoToRecognition's forward slide). Doesn't
  // touch augmentDone or pastRecognition: the image stays in its clean
  // recognition look throughout (no outlines, not click-to-add-point) --
  // this is purely "let me revisit the two model choices", not "let me
  // re-edit holds". No new API call either -- holdModel/routeModel are
  // still whatever was last picked, ready to be changed or resubmitted.
  function handleChangeModel() {
    modelSlideBackFromRectRef.current = lockedModelsRef.current?.getBoundingClientRect() ?? null;
    setShowLockedModels(false);
    setRoutesPanelEntered(false);
    setSelectedRouteId(null);
    setRouteHighlightPhase("idle");
    setQueuedRouteId(null);
    setSelectedHoldIndex(null);
    setTimeout(() => {
      setRecognitionDone(false);
      setShowModelSelect(true);
    }, MODEL_SELECT_LEAVE_MS);
  }

  // Reverses the whole augment->model-select->recognition arc in one step,
  // back to full editing. The toolbar/segment panel's own CSS transitions
  // are already bidirectional (removing the class they gained just plays
  // them backwards), so flipping augmentDone back to false is enough to
  // bring those back on its own. modelSelectPanel/routesPanel don't have
  // that built in (only ever mounted forward), so their own reverse fades
  // are played here explicitly before unmounting them.
  function handleBackToAugment() {
    // Swap the real (post-augmentation) image back out for the original --
    // see originalImageSrcRef's own comment for why this has to happen
    // before augmentDone flips back to false and ImageCanvas's live preview
    // overlays reactivate.
    if (originalImageSrcRef.current) {
      setImageSrc(originalImageSrcRef.current);
    }
    setAugmentDone(false);
    setRecognitionDone(false);
    setPastRecognition(false);
    setRoutesPanelEntered(false);
    setShowLockedModels(false);
    modelSlideBackFromRectRef.current = null;
    setSelectedRouteId(null);
    setRouteHighlightPhase("idle");
    setQueuedRouteId(null);
    setSelectedHoldIndex(null);
    if (showModelSelect) {
      setModelSelectEntered(false);
      setTimeout(() => setShowModelSelect(false), MODEL_SELECT_LEAVE_MS);
    }
  }

  return (
    <div className={styles.workspace}>
      <AppHeader
        title="ROUTNet"
        titleRef={logoRef}
        right={
          <StepIndicator
            current={recognitionDone ? "recognition" : "augment"}
            uploaded={!!imageId}
            augmentDone={augmentDone}
            handingOff={augmentDone}
          />
        }
      />

      {/* Both children share this one reserved spot below the header --
          same "stack in one grid cell" technique used for segmentSlot
          below. The toolbar fades out here once augmentDone (unchanged);
          the locked-model chips fade/slide into this same spot once
          Recognition succeeds (see showLockedModels above). */}
      <div className={styles.toolbarArea}>
        <div className={`${styles.toolbarSlot} ${augmentDone ? styles.leaving : ""}`}>
          <div className={styles.toolbar}>
            <div className={`${styles.toolbarItem} ${toolbarEntered ? styles.toolbarItemIn : ""}`}>
              <ImageSourceButtons
                webcamActive={webcamActive}
                disabled={loading}
                onFileSelected={loadImage}
                onToggleWebcam={() => {
                  setError(null);
                  setWebcamActive((prev) => !prev);
                }}
                onOpenGallery={() => {
                  setError(null);
                  setGalleryOpen(true);
                }}
              />
            </div>
            {imageId && (
              <div
                className={`${styles.toolbarActions} ${styles.toolbarItem} ${styles.toolbarItemActions} ${actionsEntered ? styles.toolbarItemIn : ""}`}
              >
                <Button
                  type="button"
                  variant="secondary"
                  disabled={pendingPoints.length === 0 || detecting}
                  onClick={handleDetectSegments}
                >
                  {detecting
                    ? "Detecting…"
                    : `Detect Holds${pendingPoints.length > 0 ? ` (${pendingPoints.length})` : ""}`}
                </Button>
                <Button
                  type="button"
                  variant="primary"
                  disabled={loading}
                  onClick={augmentDone ? handleBackToAugment : handleFinishAugment}
                >
                  {augmentDone ? "Back to Augment" : "Finish Augment"}
                </Button>
              </div>
            )}
            {imageId && (
              <div
                className={`${styles.lighting} ${styles.toolbarItem} ${styles.toolbarItemLighting} ${actionsEntered ? styles.toolbarItemIn : ""}`}
              >
                <LabeledSlider
                  label="Lighting"
                  value={lightingIntensity}
                  onChange={setLightingIntensity}
                  min={-1}
                  max={1}
                  step={0.01}
                  formatValue={(v) => v.toFixed(2)}
                />
              </div>
            )}
          </div>
        </div>

        {showLockedModels && (
          <div ref={lockedModelsRef} className={styles.lockedModelsSlot}>
            <span className={styles.lockedModelChip}>
              <span className={styles.lockedModelChipLabel}>Hold model</span>
              <span className={styles.lockedModelChipValue}>{holdModel}</span>
            </span>
            <span className={styles.lockedModelChip}>
              <span className={styles.lockedModelChipLabel}>Route model</span>
              <span className={styles.lockedModelChipValue}>{routeModel}</span>
            </span>
            <button type="button" className={styles.changeModelButton} onClick={handleChangeModel}>
              Change model
            </button>
            {/* Right end of this same bar -- same handleBackToAugment (full
                reversal of the augment->model-select->recognition arc) and
                same styling as the standalone .backToAugmentSlot button
                above, just docked in this row instead of floating alone
                now that the row itself is what's on screen at this point. */}
            <Button
              type="button"
              variant="primary"
              className={styles.lockedModelsBackButton}
              onClick={handleBackToAugment}
            >
              Back to Augment
            </Button>
          </div>
        )}

        {showBackToAugment && (
          <div
            className={`${styles.backToAugmentSlot} ${backToAugmentEntered ? styles.backToAugmentSlotIn : ""}`}
          >
            <Button type="button" variant="primary" onClick={handleBackToAugment}>
              Back to Augment
            </Button>
          </div>
        )}
      </div>

      {error && (
        <p className={styles.error}>
          <span className={styles.errorDot} aria-hidden />
          {error}
        </p>
      )}

      <div className={`${styles.mainGrid} ${showFullWidthCanvas ? styles.mainGridFull : ""}`}>
        <div className={styles.canvasColumn}>
          <ImageCanvas
            imageSrc={imageSrc}
            // Live simulation while still editing (lightingPercent's
            // overlay, chalkBySegmentId/colorBySegmentId's SVG overlays) --
            // once Finish Augment lands the real augmented image in
            // imageSrc above, those effects are already baked into its
            // pixels, so continuing to apply them here would double them
            // up. Neutral values past that point: 50 is lightingPercent's
            // own documented no-op (see ImageCanvas -- it's still on its
            // own older 0-100/50-neutral scale, so lightingIntensity's -1
            // to 1 is converted at this one callsite rather than changing
            // ImageCanvas itself), {} shows no chalk/color fill.
            lightingPercent={augmentDone ? 50 : lightingIntensity * 50 + 50}
            segments={segments}
            chalkBySegmentId={augmentDone ? {} : chalkBySegmentId}
            colorBySegmentId={augmentDone ? {} : colorBySegmentId}
            pickingColor={pickingColorSegmentId !== null}
            onPickColor={handlePickColor}
            pendingPoints={pendingPoints}
            webcamActive={webcamActive}
            loading={loading || detecting}
            onCaptureFrame={loadImage}
            onWebcamError={(message) => {
              setError(message);
              setWebcamActive(false);
            }}
            onAddSegmentPoint={handleAddSegmentPoint}
            interactive={!pastRecognition && !augmentDone}
            showHoldOutline={!pastRecognition}
            overlay={
              recognitionDone &&
              visibleHolds.length > 0 && (
                <svg
                  className={styles.highlightOverlay}
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                  aria-hidden
                  style={
                    selectedRouteId != null
                      ? ({ "--route-highlight": colorForRoute(selectedRouteId) } as CSSProperties)
                      : undefined
                  }
                >
                  {visibleHolds.map((hold, index) => {
                    const leaving = routeHighlightPhase === "exiting";
                    const focused = !leaving && index === selectedHoldIndex;
                    return (
                      <polygon
                        key={`${selectedRouteId}-${index}`}
                        className={[
                          leaving ? styles.highlightHoldLeave : styles.highlightHoldEnter,
                          focused ? styles.highlightHoldFocused : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                        points={visibleHoldPoints[index]}
                        style={leaving ? undefined : { animationDelay: `${index * ROUTE_HOLD_REVEAL_STAGGER_MS}ms` }}
                      />
                    );
                  })}
                </svg>
              )
            }
          />
        </div>

        <div
          className={`${styles.segmentSlot} ${showFullWidthCanvas ? styles.segmentSlotHidden : ""}`}
        >
          <SegmentPanel
            hasImage={!!imageId}
            segments={segments}
            chalkBySegmentId={chalkBySegmentId}
            onChalkChange={handleChalkChange}
            onRemove={handleRemoveSegment}
            colorBySegmentId={colorBySegmentId}
            pickingSegmentId={pickingColorSegmentId}
            onChooseColor={handleChooseColor}
            closing={augmentDone}
          />

          {/* Hidden for as long as the recognition loader is on screen (see
              loaderPhase) -- the two dropdowns/Recognition button read as
              still-live controls sitting right next to it otherwise, when
              actually a request is already in flight and nothing here is
              interactive. Reappears on its own once the loader returns to
              idle -- either the ordinary route (handleGoToRecognition's own
              success path already flips showModelSelect false before that
              happens) or, on a failed attempt, so the controls come back
              and the user can retry. */}
          {showModelSelect && loaderPhase === "idle" && (
            <div
              className={`${styles.modelSelectPanel} ${modelSelectEntered ? styles.modelSelectPanelIn : ""}`}
            >
              <div className={styles.modelSelectRow} ref={modelSelectRowRef}>
                <ModelSelect
                  label="hold model"
                  options={holdModelOptions}
                  value={holdModel}
                  onChange={setHoldModel}
                />
                <ModelSelect
                  label="route model"
                  options={routeModelOptions}
                  value={routeModel}
                  onChange={setRouteModel}
                />
              </div>

              {showRecognitionButton && (
                <div
                  className={`${styles.recognitionButtonWrap} ${recognitionEntered ? styles.recognitionButtonWrapIn : ""}`}
                >
                  <Button
                    type="button"
                    variant="primary"
                    disabled={inferring}
                    onClick={handleGoToRecognition}
                  >
                    {inferring ? "Running…" : "Recognition"}
                  </Button>
                </div>
              )}
            </div>
          )}

          {loaderPhase !== "idle" && (
            <div className={styles.recognitionLoaderSlot}>
              <div ref={loaderRef} className={styles.recognitionLoaderClone}>
                <Wordmark text="ROUTNet" loop paused={loaderPhase === "out"} />
              </div>
              <p className={styles.recognitionLoaderText}>Working in progress…</p>
            </div>
          )}

          {recognitionDone && recognitionResult && (
            <div
              className={`${styles.routesPanel} ${routesPanelEntered ? styles.routesPanelIn : ""}`}
            >
              <p className={styles.routesSummary}>
                {routes.length} route{routes.length === 1 ? "" : "s"} detected
                {typeof recognitionResult.inferenceMetrics.hold_count === "number"
                  ? ` across ${recognitionResult.inferenceMetrics.hold_count} holds`
                  : ""}
                .
              </p>

              <ul className={styles.routeList}>
                {routes.map((route, index) => {
                  const isSelected = routeHighlightPhase !== "idle" && selectedRouteId === route.routeId;
                  const holdsForRoute = holdsByRoute.get(route.routeId) ?? [];
                  return (
                    <li key={route.routeId}>
                      {/* A plain div, not a button -- it now wraps the
                          hold-button row too (nested buttons aren't valid
                          HTML), so the card's own border/glow visually
                          contains both, per how this was asked for
                          ("under the route card, not outside it"). */}
                      <div
                        className={`${styles.routeCard} ${isSelected ? styles.routeCardSelected : ""}`}
                        style={
                          {
                            // The card's own glow border uses this route's own
                            // color rather than one shared neon for every route --
                            // see the --route-highlight comment in the CSS.
                            "--route-highlight": ROUTE_COLORS[index % ROUTE_COLORS.length],
                          } as CSSProperties
                        }
                      >
                        <button
                          type="button"
                          className={styles.routeCardHeader}
                          aria-pressed={isSelected}
                          onClick={() => selectRoute(route.routeId)}
                        >
                          <span
                            className={styles.routeSwatch}
                            style={{ background: ROUTE_COLORS[index % ROUTE_COLORS.length] }}
                            aria-hidden
                          />
                          <span className={styles.routeLabel}>Route {route.routeId}</span>
                          <span className={styles.routeHoldCount}>
                            {route.holds.length} hold{route.holds.length === 1 ? "" : "s"}
                          </span>
                        </button>

                        {/* Always mounted (not `isSelected &&`), just
                            collapsed via CSS -- switching which route is
                            selected used to unmount one card's row and
                            mount another's in the same instant, reflowing
                            every card in between at once (visible as a
                            page-wide "blink"). Height-animating a row
                            that's always there avoids that. */}
                        <div
                          className={`${styles.holdButtonRow} ${isSelected ? styles.holdButtonRowExpanded : ""}`}
                          aria-hidden={!isSelected}
                        >
                          {holdsForRoute.map((_, holdIndex) => (
                            <button
                              key={holdIndex}
                              type="button"
                              tabIndex={isSelected ? 0 : -1}
                              className={`${styles.holdButton} ${selectedHoldIndex === holdIndex ? styles.holdButtonSelected : ""}`}
                              aria-pressed={selectedHoldIndex === holdIndex}
                              onClick={() => selectHold(holdIndex)}
                            >
                              Hold {holdIndex + 1}
                            </button>
                          ))}
                        </div>
                      </div>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}
        </div>
      </div>

      {galleryOpen && (
        <GalleryPicker
          onClose={() => setGalleryOpen(false)}
          onSelect={handleGallerySelect}
          selecting={gallerySelecting}
        />
      )}
    </div>
  );
}
