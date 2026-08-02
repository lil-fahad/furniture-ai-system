# Furniture AI System

مستودع موحّد لمشاريع الأثاث والتصميم الداخلي في حساب `lil-fahad`.

This repository is a secure monorepo control plane. It pins the useful source repositories to reviewed commits, exposes one FastAPI gateway for capability discovery, and keeps experimental, legacy, private, and quarantined projects clearly separated.

## Quick start

```bash
git clone --recurse-submodules https://github.com/lil-fahad/furniture-ai-system.git
cd furniture-ai-system
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn furniture_system.main:app --reload
```

Windows PowerShell:

```powershell
git clone --recurse-submodules https://github.com/lil-fahad/furniture-ai-system.git
cd furniture-ai-system
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn furniture_system.main:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Unified structure

- `src/furniture_system/` — secure API gateway and source registry.
- `components/` — reviewed public AI components pinned as Git submodules.
- `private/` — private product/demo repositories; GitHub authentication is required.
- `legacy/` — older prototypes retained for reference, not production.
- `sources.lock.json` — immutable source commit registry and integration status.
- `docs/AUDIT.md` — technical review and consolidation decisions.
- `docs/SECURITY.md` — urgent security actions and repository rules.

## Capabilities

The unified registry covers:

- floor-plan parsing and semantic segmentation;
- collision-aware furniture placement;
- furniture and room classification;
- generative interior-design experiments;
- Streamlit, React, Node, Flask, and FastAPI interfaces;
- commerce/catalog and designer-booking prototypes.

## Source policy

Source code is not flattened blindly. Submodules preserve each repository's history and license while the lock file makes builds reproducible. Private repositories remain private. A repository with a credential exposed in Git history is blocked and is not imported.

## Commands

```bash
make setup          # install gateway
make sync-public    # initialize public components only
make sync-all       # initialize public + private components
make test           # run tests
make api            # start unified API
```

## Security warning

One reviewed legacy repository contains an exposed Alibaba credential in Git history. It is intentionally excluded. Rotate/revoke the credential and clean the source history before any future import. See `docs/SECURITY.md`.
