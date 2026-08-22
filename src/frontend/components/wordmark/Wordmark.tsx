"use client";

import { useId, useLayoutEffect, useRef, useState } from "react";
import styles from "./Wordmark.module.css";

interface Point {
  x: number;
  y: number;
}

interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface WordmarkProps {
  text: string;
  /**
   * Draws the route in on a repeating cycle instead of once-and-settled --
   * used by WallImageWorkspace's recognition loader, a cloned Wordmark
   * standing in for "still working" while the backend infers. The climber
   * (a "topped out"/done pose) doesn't fit that, so it's skipped entirely
   * in this mode rather than popping in in sync with a cycle that has no
   * real end.
   */
  loop?: boolean;
  /**
   * Freezes the loop's animations exactly wherever they currently are --
   * used while the recognition loader is flying back to the real logo
   * (see WallImageWorkspace's loaderPhase), so the route/spark aren't
   * still visibly redrawing/traveling at the same time the whole clone is
   * moving and fading. Only meaningful alongside loop.
   */
  paused?: boolean;
}

// PAD_X_BASE/RIGHT_PAD_BASE were tuned at REFERENCE_FONT_SIZE and scale with
// the ".word" class's actual font-size (read at runtime) so they don't need
// re-tuning by hand every time the font-size in Wordmark.module.css changes.
const REFERENCE_FONT_SIZE = 34;
const PAD_X_BASE = 16;
const RIGHT_PAD_BASE = 24;
const BASELINE_GUESS = 60;
// Fixed (not font-scaled) headroom for the decorative bits that don't scale
// with text size: TOP_MARGIN clears the climber above the cap line,
// BOTTOM_MARGIN clears the start hold below it.
const TOP_MARGIN = 22;
const BOTTOM_MARGIN = 10;
const FALLBACK_WIDTH = 220;
const FALLBACK_HEIGHT = 90;

/** Letters the route should duck behind instead of crossing in front of. */
const ROUTE_BEHIND_LETTERS = new Set(["O", "N"]);

/** Smooth, flowing curve through every point (uniform Catmull-Rom -> cubic bezier). */
function smoothPath(points: Point[]): string {
  if (points.length < 2) return "";
  if (points.length === 2) {
    return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
  }
  let d = `M ${points[0].x} ${points[0].y}`;
  for (let i = 0; i < points.length - 1; i++) {
    const p0 = points[i - 1] ?? points[i];
    const p1 = points[i];
    const p2 = points[i + 1];
    const p3 = points[i + 2] ?? p2;
    const c1x = p1.x + (p2.x - p0.x) / 6;
    const c1y = p1.y + (p2.y - p0.y) / 6;
    const c2x = p2.x - (p3.x - p1.x) / 6;
    const c2y = p2.y - (p3.y - p1.y) / 6;
    d += ` C ${c1x} ${c1y}, ${c2x} ${c2y}, ${p2.x} ${p2.y}`;
  }
  return d;
}

/**
 * The ROUTNet logotype, following the hand sketch in
 * Project stuff/Logo.jpeg: a marked hold before the "R", a route that
 * weaves through the word at hand height, breaking into a dashed line as
 * it climbs past "Ne", and a tiny climber topping out with both arms
 * raised above the "t". Waypoints are fractions of the *measured* word
 * bounding box (via SVG getBBox), so the route stays anchored to the real
 * glyphs regardless of font or text changes -- only the summit uses the
 * last character's own box, so it lands precisely at the final letter.
 */
export function Wordmark({ text, loop = false, paused = false }: WordmarkProps) {
  const uid = useId();
  const gradientId = `route-gradient-${uid}`;
  const maskId = `route-mask-${uid}`;
  const textRef = useRef<SVGTextElement>(null);
  const charRefs = useRef<(SVGTSpanElement | null)[]>([]);
  const solidRef = useRef<SVGPathElement>(null);
  const dashedRef = useRef<SVGPathElement>(null);
  const [wordBox, setWordBox] = useState<Rect | null>(null);
  const [charBoxes, setCharBoxes] = useState<(Rect | null)[]>([]);
  const [fontSize, setFontSize] = useState<number | null>(null);
  const [drawn, setDrawn] = useState(false);

  useLayoutEffect(() => {
    if (textRef.current) {
      setWordBox(textRef.current.getBBox());
      setFontSize(parseFloat(getComputedStyle(textRef.current).fontSize) || REFERENCE_FONT_SIZE);
    }
    setCharBoxes(charRefs.current.map((el) => (el ? el.getBBox() : null)));
  }, [text]);

  const lastBox = charBoxes[charBoxes.length - 1] ?? null;

  useLayoutEffect(() => {
    if (!wordBox || !lastBox || !solidRef.current || !dashedRef.current) return;
    const length = solidRef.current.getTotalLength();
    solidRef.current.style.setProperty("--path-length", `${length}`);

    // The dashed segment starts with a short solid "bridge" so its glow
    // never thins out right where it meets the solid line -- two
    // independently-blurred paths tapering toward the same point can read
    // as a break even though the underlying geometry is continuous. After
    // the bridge it falls into the normal dash/gap rhythm, capped with one
    // oversized gap so the pattern never actually repeats.
    const dashedLength = dashedRef.current.getTotalLength();
    const BRIDGE = 12;
    const DASH = 6;
    const GAP = 4.5;
    const pattern = [BRIDGE, 0];
    for (let covered = BRIDGE; covered < dashedLength; covered += DASH + GAP) {
      pattern.push(DASH, GAP);
    }
    pattern.push(9999);
    dashedRef.current.style.strokeDasharray = pattern.join(" ");

    solidRef.current.getBoundingClientRect(); // force layout before the class flip
    const raf = requestAnimationFrame(() => setDrawn(true));
    return () => cancelAnimationFrame(raf);
  }, [wordBox, lastBox]);

  // The text is first measured at (PAD_X_BASE, BASELINE_GUESS); everything
  // below re-fits the box tightly around the real glyphs at whatever the
  // actual font-size turns out to be, then nudges the text to match --
  // so this stays correct without hand-tuning constants for each font-size.
  const scale = fontSize ? fontSize / REFERENCE_FONT_SIZE : 1;
  const padX = PAD_X_BASE * scale;
  const rightPad = RIGHT_PAD_BASE * scale;
  const shiftX = wordBox ? padX - PAD_X_BASE : 0;
  const shiftY = wordBox ? TOP_MARGIN - wordBox.y : 0;
  const baseline = BASELINE_GUESS + shiftY;

  const effectiveWordBox: Rect | null = wordBox
    ? { x: wordBox.x + shiftX, y: wordBox.y + shiftY, width: wordBox.width, height: wordBox.height }
    : null;
  const effectiveLastBox: Rect | null = lastBox
    ? { x: lastBox.x + shiftX, y: lastBox.y + shiftY, width: lastBox.width, height: lastBox.height }
    : null;

  const ready = effectiveWordBox !== null && effectiveLastBox !== null;
  const viewWidth = wordBox ? Math.ceil(padX + wordBox.width + rightPad) : FALLBACK_WIDTH;
  const viewHeight = wordBox ? Math.ceil(TOP_MARGIN + wordBox.height + BOTTOM_MARGIN) : FALLBACK_HEIGHT;

  // Full-height mask cutouts, one per ROUTE_BEHIND_LETTERS character, spanning
  // that character's entire measured cell width. Masking out the route for
  // the whole cell (rather than repainting the glyph's literal ink on top of
  // it) avoids the route peeking back through a letter's open counters --
  // e.g. "N" is just two verticals and a diagonal, so a route that grazes
  // its gaps would flicker visible/hidden/visible instead of reading as one
  // clean pass behind the letter.
  const behindBoxes: Rect[] = charBoxes
    .map((box, index) => (box && ROUTE_BEHIND_LETTERS.has(text[index]) ? box : null))
    .filter((box): box is Rect => box !== null)
    .map((box) => ({ x: box.x + shiftX, y: box.y + shiftY, width: box.width, height: box.height }));

  let solidD: string | undefined;
  let dashedD: string | undefined;
  let fullD: string | undefined;
  let start: Point | undefined;
  let summit: Point | undefined;

  if (effectiveWordBox && effectiveLastBox) {
    const fx = (f: number) => effectiveWordBox.x + f * effectiveWordBox.width;
    const fy = (f: number) => effectiveWordBox.y + f * effectiveWordBox.height;

    // Weaves through the lower-middle of "ROUTN" at roughly hand height,
    // but arcs over the top of the "R" right after the start hold.
    const solidPoints: Point[] = [
      { x: fx(-0.04), y: fy(0.5) },
      { x: fx(0.07), y: fy(-0.01) },
      { x: fx(0.22), y: fy(0.4) },
      { x: fx(0.37), y: fy(0.73) },
      { x: fx(0.53), y: fy(0.1) },
      { x: fx(0.65), y: fy(0.6) },
    ];
    start = solidPoints[0];
    summit = { x: effectiveLastBox.x + effectiveLastBox.width * 0.48, y: effectiveLastBox.y - 3 };

    const dashedPoints: Point[] = [
      solidPoints[solidPoints.length - 1],
      { x: fx(0.8), y: fy(0.18) },
      summit,
    ];

    solidD = smoothPath(solidPoints);
    dashedD = smoothPath(dashedPoints);
    fullD = smoothPath([...solidPoints, ...dashedPoints.slice(1)]);
  }

  return (
    <svg
      viewBox={`0 0 ${viewWidth} ${viewHeight}`}
      className={styles.svg}
      aria-hidden
      focusable="false"
    >
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="100%" x2="100%" y2="0%">
          <stop offset="0%" className={styles.gradStart} />
          <stop offset="100%" className={styles.gradEnd} />
        </linearGradient>
        {/* Feathers the mask cutouts below so the glow fades out gradually
            at a letter's edge instead of hitting a hard rectangle -- the
            route's own drop-shadow blur is soft, so an unblurred mask edge
            would clip it into a visible box. */}
        <filter id={`${maskId}-feather`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="3.5" />
        </filter>
        {/* Cuts the route out across each behind-letter's full cell width
            (not just its ink) -- see the behindBoxes comment above. */}
        <mask id={maskId}>
          <rect x={-20} y={-20} width={viewWidth + 40} height={viewHeight + 40} fill="white" />
          {behindBoxes.map((box, index) => (
            <rect
              key={index}
              x={box.x}
              y={-10}
              width={box.width}
              height={viewHeight + 20}
              fill="black"
              filter={`url(#${maskId}-feather)`}
            />
          ))}
        </mask>
      </defs>

      {/*
        Base text: every letter, normal fill. The route paints on top of
        this layer next (masked out under O/N), so it reads as running in
        front of every letter except where it ducks behind those two.
      */}
      <text ref={textRef} x={padX} y={baseline} className={styles.word}>
        {[...text].map((char, index) => (
          <tspan key={index} ref={(el) => { charRefs.current[index] = el; }}>
            {char}
          </tspan>
        ))}
      </text>

      {ready && solidD && dashedD && summit && (
        <g
          className={[
            drawn ? (loop ? styles.routeLoop : styles.routeIn) : "",
            loop && paused ? styles.routePaused : "",
          ]
            .filter(Boolean)
            .join(" ")}
        >
          <g className={styles.routeGlow}>
            <g mask={`url(#${maskId})`}>
              <path
                ref={solidRef}
                d={solidD}
                className={styles.routeLine}
                style={{ stroke: `url(#${gradientId})` }}
              />
              <path
                ref={dashedRef}
                d={dashedD}
                className={styles.routeDashed}
                style={{ stroke: `url(#${gradientId})` }}
              />
              {/* A little light that runs the route once the draw-in settles
                  (or continuously in loop mode -- this animation is already
                  its own infinite loop either way, see .spark below). */}
              {drawn && fullD && (
                <circle
                  className={styles.spark}
                  r="2.4"
                  cx="0"
                  cy="0"
                  style={{ offsetPath: `path('${fullD}')` }}
                />
              )}
            </g>
          </g>

          {/*
            Climber topping out, arms raised -- a "done" pose, so it's
            skipped in loop mode (see WordmarkProps.loop) rather than
            popping in against a draw-in that never actually finishes.
            Position (attribute transform) and the pop-in scale (CSS
            transform) are split across two <g>s -- an SVG element's
            transform attribute and its CSS transform property don't
            compose; CSS silently wins and drops the attribute's translate
            if both land on one node.
          */}
          {!loop && (
            <g transform={`translate(${summit.x}, ${summit.y})`}>
              <g className={styles.climber}>
                <path d="M0 0 L -6.5 -8 M0 0 L -2.5 -10.5" className={styles.climberLine} />
                <circle cx="3.5" cy="-6.5" r="2.4" className={styles.climberHead} />
              </g>
            </g>
          )}
        </g>
      )}

      {/* Painted last so the route reads as emerging from behind the hold. */}
      {start && <circle className={styles.startHold} cx={start.x} cy={start.y} r={6.5} />}
    </svg>
  );
}
