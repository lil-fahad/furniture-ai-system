# Furniture AI System

A single production-oriented repository for interior-design automation. It replaces the former collection of overlapping repositories and submodules with one Python package, one API, one UI, one model registry, and one test suite.

## Included capabilities

- Secure floor-plan image upload and validation.
- Deterministic room extraction using OpenCV, with safe fallbacks.
- Constraint-based furniture placement using Shapely.
- Optional OpenAI vision refinement and design briefs through `OPENAI_API_KEY`.
- Product catalog and persistent SQLite booking service.
- Optional local room-classifier and floor-plan-segmenter checkpoints.
- Reproducible training scripts and model manifests.
- FastAPI, Streamlit, Docker, CI, and security tests.

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

## Secrets

Never commit API keys. GitHub Actions should provide `OPENAI_API_KEY` as a repository secret. The application only reports whether the integration is configured; it never returns or logs the value.

## Project layout

```text
src/furniture_ai/      Unified application package
apps/                  One Streamlit interface
training/              Supported training pipelines
models/                 Checkpoint manifest and local checkpoint directory
data/                   Catalog data
scripts/                Validation and model-management utilities
tests/                  Unified tests
docs/                   Architecture, migration, security, and model docs
```

See `docs/MIGRATION.md` for the consolidation decisions and `PROVENANCE.json` for source commits.
