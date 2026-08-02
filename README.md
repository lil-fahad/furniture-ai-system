# Furniture AI System

A single production-oriented repository for interior-design automation. It replaces the former collection of overlapping repositories and submodules with one Python package, one API, one UI, one model registry, and one test suite.

## Included capabilities

- Secure floor-plan image upload and validation.
- Deterministic room extraction using OpenCV, with safe fallbacks.
- Constraint-based furniture placement using Shapely.
- Optional OpenAI vision refinement and design briefs through `OPENAI_API_KEY`.
- Product catalog and persistent SQLite booking service.
- Verified external professional model bundle with safe installation and lazy use.
- Optional DETR, SAM 2.1, Depth Anything V2, and trained EfficientNet-B0 checkpoints.
- Reproducible training scripts and model manifests.
- FastAPI, Streamlit, Docker, CI, repository audits, and security tests.

## Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
cp .env.example .env
uvicorn furniture_ai.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

For the UI:

```bash
streamlit run apps/streamlit_app.py
```

## Install the professional models

The repaired model archive is deliberately kept outside ordinary Git history. Install it with:

```bash
python scripts/install_professional_bundle.py \
  /path/to/FurnitureAI_Professional_Models_v0.4.1_REPAIRED.zip
```

The installer verifies the archive SHA-256 and every allowlisted model file before placing weights in `models/professional/installed/`.

```bash
python scripts/install_professional_bundle.py --verify-installed
python scripts/model_manifest.py
```

See `docs/PROFESSIONAL_MODELS.md` for confirmed sources, revisions, licenses, metrics, and repair limitations.

## Secrets

Never commit API keys. GitHub Actions should provide `OPENAI_API_KEY` as a repository secret. The application only reports whether the integration is configured; it never returns or logs the value.

## Project layout

```text
src/furniture_ai/      Unified application package
apps/                  One Streamlit interface
training/              Supported training pipelines
models/                 Lightweight manifests and external-model installer metadata
data/                   Catalog data
scripts/                Validation, bundle installation, and audit utilities
tests/                  Unified tests
docs/                   Architecture, migration, security, and model docs
```

See `docs/MIGRATION.md` for the consolidation decisions and `PROVENANCE.json` for source commits.
