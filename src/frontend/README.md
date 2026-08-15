# ROUTNet frontend

Next.js (App Router, TypeScript, CSS Modules) demo UI for the
zero-shot-route-segmentation project. Implements the landing page from
`Project stuff/UI_page_1.png`: pick or capture a wall image, mark hold
segments, tune lighting/chalk, then hand off to recognition.

## Getting started

```bash
npm install
npm run dev
```

Open http://localhost:3000. Node version is pinned in `.nvmrc`
(`nvm use`); `npm install` only ever writes to this folder's own
`node_modules/`, nothing global.

## Talking to the backend

The [Dashboard Backend](/docs/spec/pipeline/interfaces/dashboard-backend.md)
doesn't exist yet, so every API call in this app goes through a single
client at `lib/api/index.ts`:

- `lib/api/contract.ts` — the `RouteDetectionApiClient` interface, derived
  from the REST sketch in `docs/diagrams/spec_rest_api.drawio.svg`.
- `lib/api/endpoints.ts` — the real `fetch`-based implementation, ready to
  point at a live backend.
- `lib/api/mocks/mockClient.ts` — an in-memory implementation so the UI is
  fully usable today.

Components only ever import `apiClient` from `@/lib/api` — never either
implementation directly — so switching backends doesn't touch UI code.
Copy `.env.local.example` to `.env.local` and flip
`NEXT_PUBLIC_USE_MOCK_API=false` once a real backend is reachable at
`NEXT_PUBLIC_API_BASE_URL`.

## Structure

```
app/                  routes (App Router)
components/           UI components, one folder per component
lib/api/              typed REST client + mocks (see above)
```
