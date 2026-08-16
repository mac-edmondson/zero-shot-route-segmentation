# Project Progress

Living status doc for `zero-shot-route-segmentation`. Read this first when resuming work — it says what's actually built vs. what's still spec-only, and the state of in-flight decisions that aren't obvious from the code alone.

Last updated: 2026-08-16.

## One-paragraph status

The **frontend landing page** (`src/frontend/`) is built out and polished — augmentation workspace, custom animated wordmark, a 3-step progress indicator, and a fairly elaborate "black hole" entrance animation for the segments panel. A real **Python backend now exists** (`src/backend/`) — FastAPI, stateless, no database — implementing a batched hold-segmentation pipeline (mock SAM3 stand-in for now) and a proxy for an external sample-image gallery server. The frontend can run against either this real backend or an in-memory mock client, switched with one env var; both are kept behaviorally in sync. `src/pipeline/` (the shared, model-agnostic data-model/interface layer described in `docs/spec/pipeline/`) is still just the original stub — the real segmentation model (SAM3) itself is still unbuilt, being worked on separately by a teammate. Everything described here is committed to git on branch `praveen-frontend`.

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

**Landing page** (`components/wall-image-workspace/`, `image-canvas/`, `segment-panel/`, `gallery-picker/`, `labeled-slider/`, `button/`, `image-source-buttons/`):
- Three ways to get a wall image in: **Upload** (local file picker), **Gallery** (see below), **Webcam** capture.
- **Gallery picker** (`components/gallery-picker/GalleryPicker.tsx`) is a two-step popup: categories first (`bh`, `bh-phone`, `model`, `sm` — pulled live from the external server, not hardcoded), then images within the chosen category. Images reveal in batches of 10, auto-advancing every 350ms up to 50, then pausing behind a "Load more" button (repeats per +50) — this exists specifically so a 1000+-image category doesn't fire hundreds of concurrent thumbnail requests at the external server's free tier or jank the grid. Thumbnails render via plain `<img src>` straight to the external server (fine — CORS doesn't block image display); the actual selected image's bytes are proxied through the backend (see Backend section — that server sends no CORS headers, so a real `fetch()` to it is blocked outright).
- Click-on-image marks hold points, which batch up (`pendingPoints`) until **Detect Holds** is pressed — one request, whole image + every point, not one call per click (see Backend section for why).
- Lighting slider live-previews as a CSS `brightness()` filter on the whole image (default 0%, unmodified).
- **Chalk slider is per-segment** and live-previews as a white SVG polygon fill **clipped to that segment's own detected polygon** (same overlay SVG that draws the hold outline), opacity scaled from the slider and capped short of fully opaque so the outline stays visible underneath. Same "live, client-side, no backend round trip" treatment as Lighting, just scoped per-hold instead of image-wide.
- **`SegmentPanel` is the other big set-piece** — no static box. Before an image exists it's just centered copy; once an image loads the copy swaps (real crossfade: fade+rise+blur, both lines overlapping, ~700ms) to a prompt; the moment the first segment lands, a circle appears *behind* the text (dimmed, "hole" motif), the text drains into it **character by character** (each with its own measured `transform-origin`, staggered like drops down a drain, circle arrives *before* any character starts moving), then the circle grows into the actual panel. Removing the last segment **reverses the entire sequence** (box shrinks to a circle, text reforms out of it, last-character-first).
- Two non-obvious React bugs surfaced and fixed while building this, both worth remembering for future animation work in this codebase:
  1. `useEffect`-based phase corrections run *after* paint — if a derived value (like `phase`) can briefly compute the "wrong" resting state before an effect corrects it, an element can vanish-and-remount for one committed render, silently breaking any CSS transition that depended on DOM node identity. Fix: adjust state **during render** (comparing against a previous-value-in-state), which is React's own documented pattern for this — not `useLayoutEffect`, which only stops the *visible* flash, not the underlying reconciliation issue.
  2. Reading `ref.current` during render is unsafe (flagged by React's own lint rules) — cache measured values into `useState` via `useLayoutEffect` instead of reading the ref directly in JSX.
- A third bug worth remembering, found while building the gallery grid: a flex column item that also has `display: grid` on it will have CSS Grid's `auto` row-tracks stop sizing to content once flexbox stretches that element to a definite height — rows instead divide that fixed height evenly, squashing tall content into overlapping slivers. Fix is structural, not a CSS tweak: put the flex/scroll sizing on a wrapper element and `display: grid` on a plain child inside it, never both on the same element.
- Everything in this file is **verified via computed-style/geometry sampling** (`getComputedStyle`, `getBoundingClientRect` polled across frames), not just screenshots — screenshots repeatedly looked "fine" while masking real timing bugs.

**API layer** (`lib/api/`) — typed `RouteDetectionApiClient` interface; two implementations kept behaviorally in sync: `endpoints.ts` (real fetch calls to `src/backend/`) and `mocks/mockClient.ts` (in-memory, including a randomized-per-point mock polygon generator matching the backend's own). Switched via `NEXT_PUBLIC_USE_MOCK_API` (`.env.local`, defaults to mock). Components only ever import `apiClient` from `lib/api/index.ts`.

**Recognition page** (`app/recognition/page.tsx`) — still just a stub ("Recognition isn't built yet" + a link back). This is the planned next real chunk of frontend work — see Next Steps.

### Backend (`src/backend/`) — FastAPI, Python 3.12+, `.venv`, stateless (no database, no stored images)

Built this session, replacing the "no backend exists" state from earlier. Two things it does:

**Batched hold segmentation** (`routes.py`, `services/mock_segmentation.py`, `schemas.py`):
- `POST /image/working/segments` — the frontend sends the **whole image file** (not a reference/ID — deliberate: the backend never stores uploaded images) plus every clicked point in one batched multipart request (VIA-style `all_points_x`/`all_points_y`, JSON-encoded form fields alongside the file). The backend loops each point through a segmentation call and returns one polygon per point, same order as the input.
- The actual segmentation call (`mock_segment_point`) is currently a **mock stand-in for SAM3** — a randomized small polygon (6–10 sides, randomized radius, per-vertex jitter) synthesized around each clicked point, re-randomized on every call so repeated points don't produce identical shapes. This is the one function meant to get swapped for a real SAM3 point-prompt call once a teammate's model script is ready — the request/response contract is built to not need to change when that happens.
- `DELETE /image/working/segment/{id}` — kept as a no-op 204 purely for frontend contract compatibility (there's nothing server-side to actually delete, since nothing is stored).

**External gallery proxy** (`gallery.py`):
- Proxies `GET /gallery/categories` and `GET /gallery/images?category=...` (JSON listings) and `GET /gallery/images/{category}/{name}` (raw bytes) to an external sample-image server, configured entirely via env vars (`IMAGE_GALLERY_BASE_URL`, `IMAGE_GALLERY_PATH` in `src/backend/.env`) — never hardcoded, so the source server can change without a code change.
- Exists **only** because that external server sends no CORS headers at all — confirmed by direct curl, re-verified since — so a browser `fetch()` to it (for the JSON listing, or to read an image's bytes) is blocked outright by the browser itself. Thumbnail *display* (`<img src>`) is unaffected by this and intentionally bypasses the proxy, pointing straight at the external server.
- Image listing is filtered to real image extensions (some categories, e.g. `model`, hold non-image artifacts) and validated against path traversal on the filename/category segments.

**Not yet built on the backend:**
- `POST /image/working/augment` (the "Finish Augment" endpoint the frontend already calls) — doesn't exist yet. The chalk/lighting sliders are live-previewed entirely client-side right now (see Frontend section); nothing bakes those into the actual image yet.
- `GET /images`, `POST /images`, `GET /image/{id}`, `GET /image/working`, `PUT /image/working` — part of the original REST sketch, superseded by the stateless "send the whole image, no ID" design; the frontend no longer calls these (was fixed this session — `loadImage()` used to call the equivalent `uploadImage`/`setWorkingImage` mock-only flow, which 404s against the real backend since those routes were deliberately never built).

### Backend / ML pipeline (`src/pipeline/`)

Essentially **not started** — this is the shared, framework-level interface layer from the original spec pass (distinct from `src/backend/`, which is the working REST API built this session). Only `src/pipeline/interfaces/data_models.py` has real content (the shared `Coordinate`/`Polygon`/`Hold`/`Route`/`RGBColor` dataclasses, with several `# TODO` constraint markers). `HoldDetector`, `RouteClassifier`, their factories, `RouteDetectionPipeline`, `AugmentationSuite`, `DataPreprocessingPipeline`, and `EvaluationSuite` are all fully speced in `docs/spec/pipeline/interfaces/*.md` (MUST/SHOULD/MAY contracts, board-derived) but have zero corresponding Python. The real SAM3 hold detector is being scripted by a teammate separately; `src/backend/services/mock_segmentation.py` is the placeholder until that lands.

### Reference material (untracked, in `Project stuff/`)

- `edmondson-dsouza-hedge-cv-proposal.pdf` — the actual research proposal (color-only vs. DINO vs. combined route classification, evaluated under chalk/lighting/similar-color distortions).
- `UI_page_1.png` — original wireframe the landing page was built from.
- `Logo.jpeg` — hand sketch the wordmark's route-line concept was built from.

## Key decisions / current state worth knowing

- **The backend is deliberately stateless regarding images.** No image ID, no server-side storage, no database. The frontend always holds the original `File`/`Blob` in state and re-sends the whole thing on every call that needs it (detect-segments). This was an explicit architecture decision, not a shortcut — it's why the old `uploadImage`/`setWorkingImage` REST-sketch endpoints were dropped rather than implemented.
- **Hold detection is batched, not per-click.** One HTTP call carries every pending clicked point plus the image; the backend loops per-point internally. Frontend UX matches this: clicks accumulate (`pendingPoints`) until an explicit "Detect Holds" press.
- **The external gallery server has no CORS headers** — confirmed by direct testing, not assumed. This is *why* `src/backend/gallery.py` exists as a proxy at all; without that constraint the frontend could call it directly. If that server ever adds CORS headers, the proxy could be removed for the listing/byte-fetch paths (thumbnail `<img>` display was never proxied, since CORS doesn't affect that).
- **Git**: everything above is committed, on branch `praveen-frontend`. `.venv/` and `__pycache__/` are gitignored; `.env`/`.env.local` are gitignored with `.env.example`/`.env.local.example` committed as templates.
- **Design language is settled and reused deliberately**: the route/hold visual vocabulary from the wordmark (dot = hold, line = route, glow, orange→gold) is intentionally echoed in the step indicator and the segment panel's circle. Keep new UI consistent with that rather than inventing a new motif.
- **No animation library** — everything is hand-rolled CSS transitions/`@keyframes` + a handful of React timing patterns (double-rAF for enter animations, `@property` for animatable custom properties, render-time state adjustment for prop-driven phase changes). No framer-motion or similar is installed.
- **Package manager is npm**, styling is CSS Modules (not Tailwind) — both explicit user choices from early in the build. Backend has no package manager beyond pip/venv (no `uv`/`pyenv` available in this environment; `pyproject.toml`'s `requires-python` was relaxed to `>=3.12` to match what's actually installed).
- The **reverse animation on segment removal** was a real ask that got fully built (not a stub) — don't assume it's a "nice to have that got skipped."

## Verification habits established in this project

Worth continuing: this build repeatedly found that **screenshots alone are not sufficient** to verify an animation or layout is correct — several real bugs (snap-instead-of-transition, wrong easing curve, parallel-instead-of-sequential timing, and later the CSS Grid row-collapse bug in the gallery) only surfaced by sampling `getComputedStyle`/`getBoundingClientRect` across frames/elements and reading the actual numbers, not by eyeballing a screenshot. Same instinct applied to backend claims this session — CORS behavior, gallery server directory structure, and endpoint existence were all confirmed with direct `curl` calls rather than assumed from memory or docs. Default to that level of verification (real DOM/network inspection, not just visual inspection) for future work here.

## Next steps (not yet started, roughly in likely order)

1. **Wire in the real SAM3 model** once the teammate's script is ready — the only change needed is `mock_segment_point`'s body in `src/backend/services/mock_segmentation.py`; the request/response contract was built to not need to change.
2. **Build `POST /image/working/augment`** on the real backend so Lighting/Chalk (currently client-side-preview-only) actually bake into the image on "Finish Augment."
3. **Build out `/recognition`** — currently a stub. Needs the classifier-method controls (color-only / color+spatial / DINO-only / combined per the proposal) and whatever "cool animation" transition was being planned for the Augment → Recognition handoff (discussed but not built — the step indicator was deliberately built to *not* spoil it).
4. Decide the rest of the `src/pipeline/` build order for real (non-mock) detection/classification — spec's own suggested order is data models → `HoldDetector`/`RouteClassifier` + factories → `RouteDetectionPipeline` → augmentation/preprocessing → evaluation.
5. Resolve the open questions the specs themselves flag but never answer — e.g. whether a hold can belong to more than one route (`docs/spec/pipeline/interfaces/route-classifier.md`), referenced from a `docs/spec/decisions/open-questions.md` that doesn't currently exist in the repo.
