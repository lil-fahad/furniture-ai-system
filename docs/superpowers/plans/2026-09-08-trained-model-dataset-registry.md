# Trained Model Dataset Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Link every verifiably trained FurnitureAI model to its dataset provenance, artifact integrity and evaluation metadata without storing model binaries inside dataset directories.

**Architecture:** Add a strict JSON lineage registry in `data/` and a small runtime validator/query module. Existing model manifests remain the inference artifact registry; trained classifier entries gain dataset IDs, while the lineage registry also records the supplier ranker that is not part of the vision runtime manifest. Large/raw datasets and checkpoint binaries stay outside Git data records.

**Tech Stack:** Python 3.11+, JSON, Pydantic-free standard-library validation, pytest, existing FurnitureAI model manifest and training metadata.

**Spec:** User request in this task plus existing `docs/REAL_FURNITURE_DATASETS.md`, `models/professional/checkpoints.json`, and `models/supplier_ranker/metrics.json`.

## Global Constraints

- Preserve the existing eight-entry runtime model manifest count.
- Keep `models/manifest.json` byte-identical to `src/furniture_ai/resources/model_manifest.json` after edits.
- Do not label placeholder room/floor-plan entries as trained without a verified checkpoint.
- Do not put checkpoint/model binary paths under `data/`.
- Every trained-model record must reference at least one declared dataset and use a valid 64-character SHA-256 when an artifact hash is known.

---

### Task 1: Define lineage behavior with failing tests

**Files:**
- Create: `tests/test_model_dataset_registry.py`

**Interfaces:**
- Consumes: project registry at `data/model_dataset_registry.json`.
- Produces: expected API `ModelDatasetRegistry.load(path)`, `trained_models_for_dataset(id)`, and `datasets_for_model(id)`.

- [ ] Write tests asserting the four verified trained artifacts are present, placeholder models are excluded, dataset references resolve, artifact paths never live under `data/`, malformed references fail, and existing trained EfficientNet manifest entries expose dataset IDs.
- [ ] Run focused CI and confirm the tests fail because the registry/module do not exist yet.

### Task 2: Implement strict registry and project lineage data

**Files:**
- Create: `src/furniture_ai/model_dataset_registry.py`
- Create: `data/model_dataset_registry.json`
- Modify: `models/manifest.json`
- Modify: `src/furniture_ai/resources/model_manifest.json`

**Interfaces:**
- `ModelDatasetRegistry.load(path: Path) -> ModelDatasetRegistry`
- `trained_models_for_dataset(dataset_id: str) -> tuple[TrainedModelRecord, ...]`
- `datasets_for_model(model_id: str) -> tuple[DatasetRecord, ...]`

- [ ] Implement strict schema validation: unique IDs, known dataset references, non-empty lineage, safe artifact paths, SHA-256 format, and typed metric/provenance fields.
- [ ] Add two recovered ABO dataset lineage records and the committed supplier-master lineage record.
- [ ] Register the three verified EfficientNet checkpoints and the trained supplier Ridge ranker; retain source limitations where training provenance is incomplete.
- [ ] Add `dataset_ids` only to the three trained EfficientNet runtime manifest entries and keep both manifest mirrors byte-identical.
- [ ] Run focused tests until green.

### Task 3: Document and verify repository integration

**Files:**
- Modify: `README.md`
- Create: `docs/TRAINED_MODEL_DATASET_LINEAGE.md`

**Interfaces:**
- Documents the separation between dataset content, model artifacts and lineage metadata.

- [ ] Document how to query lineage and how future training runs should preserve dataset fingerprints/manifest hashes.
- [ ] Run lint, compile, secret scan, repository audit, focused tests and full CI.
- [ ] Open a PR only from the leased branch; merge only if exact-head coordination and CI are green.
