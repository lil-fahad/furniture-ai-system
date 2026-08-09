# Furniture AI System — Swarm Upgrade Report

## v1.3.0 — Performance & code-quality pass
**Release:** `1.2.0 → 1.3.0` · **Tests:** 103 passed (100 prior + 3 new) · **Ruff:** clean · **Audits (repo + secrets):** pass

Focused, behavior-preserving performance changes:

- **`floorplan.py`** — vectorized the exterior-seed border scan (NumPy `flatnonzero` instead of a per-pixel Python loop) and replaced `np.where(...).astype(np.uint8)` mask builds with single-allocation `cv2.compare`. Micro-benchmarks (1600×1200 noisy plan): mask build **8.38 ms → 0.11 ms (~76×)**, seed scan **0.55 ms → ~0.002 ms**; end-to-end `_extract_room_polygons` **68.3 ms → 67.0 ms (~2%)** — the remaining cost is morphology/connectedComponents, which is already native OpenCV.
- **`image_io.py`** — single-pass validate+decode: removed the probe/`verify()`/reopen cycle; one `Image.open`, dimension guard before pixel decode, `convert("RGB")` forces the validating decode. 4.2 MB PNG: **33.1 ms → 31.1 ms (~6%)**, plus one less full container parse.
- **`models.py`** — new `load_room_classifier_cached()`: thread-safe lazy singleton keyed on resolved path + class count + file size/mtime (a replaced checkpoint reloads). Repeat `timm` classifier load: **130 ms → 42 µs (~3000×)**. New tests cover instance identity, reload-on-replacement, and 8-thread concurrent first load.
- **`api.py`** — `/api/v1/analyze` now offloads blocking image decode and the OpenCV pipeline to `run_in_threadpool`, so the async event loop is no longer blocked by CPU-bound CV work. No request/response changes.
- **Investigated, intentionally unchanged:** `storage.py` (per-operation connections with `busy_timeout` are enforced by regression tests; queries are single round-trips on the PK — no N+1, no missing index, no batch insert path exists) and `layout.load_catalog` (already `lru_cache`d since 1.2.0).
- **CI (`.github/workflows/ci.yml`)** — added `concurrency` cancel-in-progress, `timeout-minutes` on both jobs, XML coverage uploaded via `actions/upload-artifact@v4`; existing pip cache kept.
- **Version sync:** `__init__.py` / `pyproject.toml` / `Dockerfile` label all at **1.3.0**; `/health` reports it.

---

## v1.2.0 (previous report)
**Date:** 2026-08-02 · **Repo:** https://github.com/lil-fahad/furniture-ai-system · **Release:** `1.1.0 → 1.2.0`

## Outcome
| Metric | Baseline | After swarm |
|---|---|---|
| Tests | 12 passed | **100 passed** |
| Coverage | 82% | **92%** (CI floor enforced at 80) |
| Ruff | clean | clean |
| Audits (repo + secrets) | pass | pass |
| Training scripts | untested, broken offline | both smoke-run verified on CPU, checkpoints loadable |

Deliverable: `/mnt/agents/output/furniture-ai-system` (git repo, branch `master`, 9 commits over baseline `ab01375`) + `furniture-ai-system-upgraded.zip`.

## How it was done
Multi-agent swarm (git-worktree isolated squads, merged to master): baseline audit → issue inventory (49 items: 8 HIGH / 14 MED / 27 LOW) → 4 parallel fix squads (API+storage / CV core / training+apps / packaging+CI) → independent reviewer + clean-clone verifier → convergence fix → final gate. Note: GitHub push was unavailable (plugin disconnected); deliverable is the local repo + zip.

## HIGH-severity fixes (all verified fixed by independent reviewer)
1. **Floor-plan exterior misread as a room** — `floodFill` hard-coded seed (0,0); a dark corner title block flipped walls/exterior. Now a verified free-space border seed + border-touching-component discard. Regression test included.
2. **Decompression-bomb / memory DoS** — pixel limit now checked from container dimensions *before* decode; `DecompressionBombError` → 422 (was 500). Spy test proves no decode on oversize.
3. **500 on degenerate polygons** — collinear/zero-area polygons rejected at two layers (pydantic contract validator + endpoint `ValueError`→422). Collinear, bowtie, NaN covered by tests.
4. **500 on malformed OpenAI payloads** — `{"rooms": null}` and junk confidences handled (isinstance validation + per-item skip); pipeline warning instead of crash.
5. **Classifier training died offline** — `--pretrained/--no-pretrained` + `FURNITURE_PRETRAINED`, graceful fallback to random init. Verified with `HF_HUB_OFFLINE=1`.
6. **Segmenter crashed on 0/255 masks** — per-sample mask-range validation with clear errors + `--mask-remap auto` (`[0,128,255] → [0,1,2]`).
7. **Silent mask corruption** — masks resized with `InterpolationMode.NEAREST` (was bilinear → spurious classes). Regression test.
8. **SQLite connection leak** — `contextlib.closing` on every connection; WAL + per-connection `busy_timeout`. Leak-proof test counts `close()` calls.

## Other notable improvements
- **Version sync:** `__init__.py`/pyproject/Docker/docs all at **1.2.0**; `/health` reports it.
- **Runtime paths:** `database_path`/`catalog_path`/`model_manifest_path` anchored to project root with cwd fallback — API, CLI, and Docker work from any working directory (previously 500s outside repo root).
- **Security deps:** `python-multipart>=0.0.18` (CVE-2024-53981), `pillow>=10.3` (CVE-2024-28219). Oversize uploads → 413. Catalog/models endpoints now require the service key.
- **Model loading:** classifier class-count inferred from checkpoint head width (12-class professional bundles now strict-load); atomic bundle install with SHA-256 member verification + rollback.
- **Docker:** non-root user, `HEALTHCHECK`, python:3.12-slim, dropped libgl1, runtime `ENV`s for manifest/db/catalog paths.
- **CI:** Python 3.11+3.12 matrix, coverage floor 80, docker-build **and container-run smoke** job (curls `/health` + `/ready`), secret scoped to one step.
- **API/CLI UX:** clean exit-2 errors on bad images; Streamlit timeout/error banners + client-side size guard.
- **Docs:** README/SECURITY/MODELS updated to match behavior; `.env.example` documents all settings.

## Verified end-to-end (clean clone, neutral cwd)
- 100/100 tests, ruff, repo audit, secret scan ✅
- uvicorn from `/tmp`: `/health` 200 (v1.2.0), `/ready` 200, `/layout` degenerate → 422, bomb PNG → 422, 11 MB upload → 413, `/analyze` 200 ✅
- CLI from `/tmp` → valid `DesignResult` JSON ✅
- Training smoke (CPU, offline): classifier checkpoint (16 MB, loadable) and TorchScript segmenter (forward pass shape-checked) ✅

## Known leftovers (non-blocking)
- Docker image build + CI container smoke written but not executed here (no Docker in sandbox) — first CI run will exercise it.
- Degenerate-geometry thresholds duplicated in `contracts.py`/`layout.py` (could be unified).
- `Image.MAX_IMAGE_PIXELS` set process-globally per request (harmless).
- Legacy `np.random.seed` in training scripts (cosmetic); Streamlit app still lacks a contract test.
- GitHub push not performed (plugin disconnected) — push `master` manually or reconnect the plugin.

## Quick start (upgraded copy)
```bash
cd furniture-ai-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,ui,training]"
pytest -q                      # 100 passed
uvicorn furniture_ai.api:app   # http://127.0.0.1:8000/docs
streamlit run apps/streamlit_app.py
```
