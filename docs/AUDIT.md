# Repository consolidation audit

Audit date: 2026-08-02

Scope: repositories in the connected `lil-fahad` GitHub installation whose purpose is furniture, furnishing, floor-plan analysis, or interior design. Presentation design, trading, industrial, and unrelated ML repositories were excluded.

## Decision summary

| Source | Tier | Decision | Main reason |
|---|---|---|---|
| `furniture_ai_suite` | core | Include | Broad classification/API/Streamlit workflow; useful floor-plan module |
| `floorplan_furnisher_pro` | core | Include | Strongest explicit segmentation → vectorization → constrained placement pipeline |
| `furniture_designer_ai` | experimental | Include, isolated | SAM and visualization prototypes, but layout inference is not production-ready |
| `Internal-designer-` | experimental | Include, isolated | Generative and YOLO training experiments; README states demo/synthetic limitations |
| `Furnishings.s` | legacy | Include as reference | Simple Streamlit UX; room detection is simulated |
| `Furnivers` | legacy | Include as reference | Early OCR/placement/3D concepts; randomized placement |
| `home_furnishing_app` | private | Include as private submodule | React/Node/Mongo commerce application |
| `Furnishings.app` | private | Include as private submodule | Catalog, JWT, and designer-booking backend |
| `furniture-ai-demo` | private | Include as private submodule | Lightweight static demo |
| `-furniture-ai` | blocked | Do not import | Exposed credential in history, duplicated app trees, placeholder training, unverified catalog data |

## Technical findings

### 1. `furniture_ai_suite`

Strengths:

- FastAPI, Streamlit, and CLI entry points.
- Image preparation, deduplication, classification, and model export concepts.
- A substantial OpenCV floor-plan analyzer and recommendation rules.

Risks and corrections:

- Alibaba integration is described as demo/mock data and must not be presented as live inventory.
- Room-type inference relies on pixel area and aspect-ratio heuristics; it needs calibrated scale or a trained semantic model.
- Consolidation keeps this service separate from the gateway to avoid dependency conflicts.

### 2. `floorplan_furnisher_pro`

Strengths:

- Coherent pipeline: segmentation, vectorization, room polygons, gate clearance, collision checks, and overlay rendering.
- API, Streamlit UI, Prometheus metrics, and detector training scripts.

Risks and corrections:

- The checked configuration uses a default JWT secret and wildcard CORS.
- Authentication token issuance is a demo path and must be replaced with real identity management.
- The segmenter requires external trained weights; startup should fail clearly when weights are absent.

### 3. `furniture_designer_ai`

Strengths:

- Segment Anything integration and room visualization utilities.
- Useful prototype boundaries for vision and layout services.

Risks and corrections:

- `plan_layout.py` creates a random state before selecting an action, so output is not conditioned on the uploaded plan.
- SAM is prompted only at the image center, which is insufficient for multi-room plans.
- Loaded PyTorch objects require a trusted model artifact and safer state-dict loading.

### 4. `Internal-designer-`

Strengths:

- Pix2Pix/PatchGAN direction, streaming data experiments, Docker setup, and YOLO furniture training.

Risks and corrections:

- The repository explicitly notes synthetic/demo behavior and possible untrained fallbacks.
- Flask/TensorFlow dependencies are isolated from the FastAPI/PyTorch services to avoid one oversized environment.

### 5. Legacy UI repositories

`Furnishings.s` manually asks the user to select detected rooms and fetches products from a generic demonstration API. `Furnivers` uses fixed or randomized furniture rectangles. Both are retained for UX history only and are excluded from the core runtime.

### 6. Private product repositories

The private commerce, booking, and static demo repositories are represented as authenticated submodules. Their code is not copied into this public repository. A user without access can initialize public components with `make sync-public`.

## Unified architecture

The target repository acts as a control plane rather than forcing incompatible frameworks into one process:

1. `furniture_system` provides a secure FastAPI registry and governance API.
2. Core model services remain independently deployable.
3. Experimental services remain opt-in.
4. Private applications remain private.
5. Legacy prototypes remain read-only references.
6. `sources.lock.json` pins every component to a reviewed commit.

## Recommended next engineering phase

1. Standardize an `AnalyzeFloorPlan` JSON schema shared by the core components.
2. Replace heuristic room labels with a calibrated semantic segmentation model.
3. Add a deterministic constraint solver using real-world dimensions and doorway clearances.
4. Replace mock commerce records with authorized APIs or a licensed catalog snapshot.
5. Add model cards, dataset cards, and reproducible evaluation fixtures.
6. Remove or archive the blocked source after credentials are rotated and history is cleaned.
