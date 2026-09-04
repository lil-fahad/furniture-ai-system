# FurnitureAI Platform V2 Architecture

## Objective

V2 keeps the verified deterministic core while replacing the prototype presentation layer with an auditable design platform. Existing `/api/v1/*` contracts remain compatible. New orchestration is additive under `/api/v2/*`.

## Runtime layers

1. **Web Studio (`apps/web`)** — Next.js, React, TypeScript, RTL/LTR, Vercel-ready. The browser never receives the FurnitureAI service key.
2. **API Entry (`furniture_ai.api_entry`)** — composes legacy v1 and additive v2 routers.
3. **Perception** — OpenCV floor-plan geometry plus optional locally pinned DETR and relative-depth models, with NVIDIA runtime selection where available.
4. **Spatial Twin** — strict Pydantic geometry contracts are the source of truth for rooms, openings and placements.
5. **Portfolio Engine** — generates deterministic layout candidates using independent placement search policies.
6. **Constraint Gate** — rejects out-of-room geometry, collisions, blocked doors, and explicit clearance violations independently from model confidence.
7. **Decision Graph** — acyclic graph linking inputs, rooms, products, candidates, validation evidence and the selected deterministic result.
8. **Supplier/Product Provenance** — production supplier records remain fail-closed behind explicit authorization and SHA-256 provenance.
9. **AI Layer** — OpenAI may refine semantic labels or produce text, but it cannot override geometric truth or execution validation.
10. **Evaluation** — real-image/model benchmarks and pinned model artifacts remain prerequisites for production model promotion.

## Candidate ranking

Candidate ordering is intentionally not called an AI score or confidence. The current v2 selector is lexicographic and measurable: valid candidates first, then more placed catalog items, then fewer validation issues, then stable policy order. A learned preference ranker can be added only after a real human-rated evaluation dataset exists.

## Migration

Streamlit remains temporarily available as a rollback surface. It should be removed only after the Next.js build, API v2 tests, preview deployment, and end-to-end workflow pass. This avoids deleting the existing UI before its replacement is proven.
