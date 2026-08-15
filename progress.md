# Project Progress

Living status doc for `zero-shot-route-segmentation`. Read this first when resuming work — it says what's actually built vs. what's still spec-only, and the state of in-flight decisions that aren't obvious from the code alone.

Last updated: 2026-08-15.

## One-paragraph status

The **frontend landing page** (`src/frontend/`) is built out and polished — augmentation workspace, custom animated wordmark, a 3-step progress indicator, and a fairly elaborate "black hole" entrance animation for the segments panel. It runs entirely against an **in-memory mock API client**, because the **Python backend does not exist yet** — `src/pipeline/` is still just the shared data-model stub from the original spec pass. The interface specs in `docs/spec/pipeline/` are mature reading material but mostly unimplemented (most files literally say "TODO: Port to code and clean"). Nothing in this repo is committed to git yet (`src/frontend/` and `Project stuff/` are both untracked).

## What's actually done

### Frontend (`src/frontend/`) — Next.js 16, App Router, TypeScript, CSS Modules, npm

**Design system** (`app/globals.css`): dark mesh-gradient background (four blurred color blobs — violet/pink/orange/teal — Canva-Magic-Design-inspired), glass/`backdrop-filter` surfaces throughout, single accent hue (`--accent: #ff7a45` → `--gold: #ffcf7a`), fluid `clamp()`-based spacing that scales from phones up through ultra-wide monitors. Whole layout is tuned to fit one viewport with **no page scroll** (`100dvh`, `overflow-y: auto` as a safety net rather than a hard `hidden`) — verified repeatedly at 360×640 through 2560×1440.

**Wordmark logo** (`components/wordmark/Wordmark.tsx` + `.module.css`) — the most technically involved piece:
- Renders "ROUTNet" as one SVG, measures the *actual* rendered glyph positions at runtime (`getBBox`, `getComputedStyle` for font-size) rather than hardcoded coordinates, so it self-corrects if the font/size ever changes again.
- A route line starts at a hold behind the "R", arcs over it, weaves through "OUTN" (passing **behind** the O and N specifically, in front of everything else), transitions to a dashed line, and climbs to a tiny climber figure topping out above the "t", arms raised.
- Orange→gold gradient, glow via layered `drop-shadow`, a one-time draw-in animation on load, and a looping "spark" that travels the route (`offset-path`).
- Current font: **Courier Prime** (bold) — went through several rounds (Poppins → Sansita Swashed → Courier Prime) based on live feedback; treat this as *not necessarily final*, just where it landed last.
- Two real bugs found and fixed here, worth knowing about if you touch this file again: (1) an SVG element's `transform` *attribute* and CSS `transform` *property* don't compose — CSS silently wins; (2) two adjacent `<path>`s with independent `drop-shadow` glows can visually "cut" at the seam even with continuous geometry — fixed with a measured solid "bridge" segment.

**Step indicator** (`components/step-indicator/StepIndicator.tsx`) — replaces a static "Finish Augmentation to proceed to Recognition" sentence with a 3-node route-themed progress bar: **Upload → Augment → Recognition**. Each node is `locked` / `inProgress` (diagonal half-fill, both dot and text) / `complete` / `active`, using `@property`-registered CSS custom properties so the fills genuinely animate (not snap) between states. Nodes only light up as a *reaction* to real events (image loaded, augmentation finished) — never pre-highlighted.

**Landing page** (`components/wall-image-workspace/`, `image-canvas/`, `segment-panel/`, `labeled-slider/`, `button/`, `image-source-buttons/`):
- Gallery upload / webcam capture / click-on-image to mark hold segments.
- Lighting slider live-previews as a CSS `brightness()` filter directly on the image (default 0%, unmodified).
- **`SegmentPanel` is the other big set-piece** — no static box. Before an image exists it's just centered copy; once an image loads the copy swaps (real crossfade: fade+rise+blur, both lines overlapping, ~700ms) to a prompt; the moment the first segment lands, a circle appears *behind* the text (dimmed, "hole" motif), the text drains into it **character by character** (each with its own measured `transform-origin`, staggered like drops down a drain, circle arrives *before* any character starts moving), then the circle grows into the actual panel. Removing the last segment **reverses the entire sequence** (box shrinks to a circle, text reforms out of it, last-character-first).
- Two non-obvious React bugs surfaced and fixed while building this, both worth remembering for future animation work in this codebase:
  1. `useEffect`-based phase corrections run *after* paint — if a derived value (like `phase`) can briefly compute the "wrong" resting state before an effect corrects it, an element can vanish-and-remount for one committed render, silently breaking any CSS transition that depended on DOM node identity. Fix: adjust state **during render** (comparing against a previous-value-in-state), which is React's own documented pattern for this — not `useLayoutEffect`, which only stops the *visible* flash, not the underlying reconciliation issue.
  2. Reading `ref.current` during render is unsafe (flagged by React's own lint rules) — cache measured values into `useState` via `useLayoutEffect` instead of reading the ref directly in JSX.
- Everything in this file is **verified via computed-style/geometry sampling** (`getComputedStyle`, `getBoundingClientRect` polled across frames), not just screenshots — screenshots repeatedly looked "fine" while masking real timing bugs.

**API layer** (`lib/api/`) — typed `RouteDetectionApiClient` interface derived from the REST sketch in `docs/diagrams/spec_rest_api.drawio.svg`; two implementations (`endpoints.ts` real fetch-based, `mocks/mockClient.ts` in-memory), switched via `NEXT_PUBLIC_USE_MOCK_API` (defaults to mock, since there's no backend). Components only ever import `apiClient` from `lib/api/index.ts` — swapping to a real backend later is a one-line env change, no UI code changes.

**Recognition page** (`app/recognition/page.tsx`) — still just a stub ("Recognition isn't built yet" + a link back). This is the planned next real chunk of frontend work — see Next Steps.

### Backend / ML pipeline (`src/pipeline/`)

Essentially **not started**. Only `src/pipeline/interfaces/data_models.py` has real content (the shared `Coordinate`/`Polygon`/`Hold`/`Route`/`RGBColor` dataclasses, with several `# TODO` constraint markers). `HoldDetector`, `RouteClassifier`, their factories, `RouteDetectionPipeline`, `AugmentationSuite`, `DataPreprocessingPipeline`, and `EvaluationSuite` are all fully speced in `docs/spec/pipeline/interfaces/*.md` (MUST/SHOULD/MAY contracts, board-derived) but have zero corresponding Python. The specs' own suggested build order (`docs/spec/pipeline/README.md`) is: data models → detector/classifier + factories → pipeline → augmentation/preprocessing → evaluation → dashboard backend/frontend.

### Reference material (untracked, in `Project stuff/`)

- `edmondson-dsouza-hedge-cv-proposal.pdf` — the actual research proposal (color-only vs. DINO vs. combined route classification, evaluated under chalk/lighting/similar-color distortions).
- `UI_page_1.png` — original wireframe the landing page was built from.
- `Logo.jpeg` — hand sketch the wordmark's route-line concept was built from.

## Key decisions / current state worth knowing

- **No backend exists.** Every "real" interaction (upload, segment, augment, infer) currently hits the mock client, which fabricates plausible responses with artificial latency. Nothing persists past a page reload.
- **Git**: `src/frontend/` and `Project stuff/` are untracked. Nothing from this whole frontend build has been committed yet.
- **Design language is settled and reused deliberately**: the route/hold visual vocabulary from the wordmark (dot = hold, line = route, glow, orange→gold) is intentionally echoed in the step indicator and the segment panel's circle. Keep new UI consistent with that rather than inventing a new motif.
- **No animation library** — everything is hand-rolled CSS transitions/`@keyframes` + a handful of React timing patterns (double-rAF for enter animations, `@property` for animatable custom properties, render-time state adjustment for prop-driven phase changes). No framer-motion or similar is installed.
- **Package manager is npm**, styling is CSS Modules (not Tailwind) — both explicit user choices from early in the build.
- The **reverse animation on segment removal** was a real ask that got fully built (not a stub) — don't assume it's a "nice to have that got skipped."

## Verification habits established in this project

Worth continuing: this build repeatedly found that **screenshots alone are not sufficient** to verify an animation is correct — several real bugs (snap-instead-of-transition, wrong easing curve, parallel-instead-of-sequential timing) only surfaced by sampling `getComputedStyle`/`getBoundingClientRect` across animation frames via Playwright and reading the actual numbers. Default to that level of verification for future animation work here, not just an eyeballed screenshot.

## Next steps (not yet started, roughly in likely order)

1. **Build out `/recognition`** — currently a stub. Needs the classifier-method controls (color-only / color+spatial / DINO-only / combined per the proposal) and whatever "cool animation" transition was being planned for the Augment → Recognition handoff (discussed but not built — the step indicator was deliberately built to *not* spoil it).
2. **Decide the Python backend's starting point** — the spec's own suggested order is data models → `HoldDetector`/`RouteClassifier` + factories → `RouteDetectionPipeline`. None of this exists yet.
3. **Wire the frontend to a real backend** once one exists — flip `NEXT_PUBLIC_USE_MOCK_API=false`, point `NEXT_PUBLIC_API_BASE_URL` at it, and expect the REST contract in `lib/api/contract.ts` to need adjusting (it's a best-effort reading of an ambiguous whiteboard sketch, explicitly flagged as provisional).
4. **Commit the work.** Nothing here is in git yet.
5. Resolve the open questions the specs themselves flag but never answer — e.g. whether a hold can belong to more than one route (`docs/spec/pipeline/interfaces/route-classifier.md`), referenced from a `docs/spec/decisions/open-questions.md` that doesn't currently exist in the repo.
