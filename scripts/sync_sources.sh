#!/usr/bin/env bash
set -euo pipefail

mode="${1:---public-only}"

public_paths=(
  components/furniture_ai_suite
  components/floorplan_furnisher_pro
  components/furniture_designer_ai
  components/internal_designer
  legacy/furnishings_streamlit
  legacy/furnivers
)

private_paths=(
  private/home_furnishing_app
  private/furnishings_app
  private/furniture_ai_demo
)

git submodule sync --recursive

case "$mode" in
  --public-only)
    git submodule update --init --recursive --depth 1 "${public_paths[@]}"
    ;;
  --all)
    git submodule update --init --recursive --depth 1 "${public_paths[@]}" "${private_paths[@]}"
    ;;
  *)
    echo "Usage: $0 [--public-only|--all]" >&2
    exit 2
    ;;
esac

python - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("sources.lock.json").read_text(encoding="utf-8"))
missing = []
for source in payload["sources"]:
    path = source.get("path")
    if not path or source["tier"] == "blocked":
        continue
    if source["visibility"] == "private" and not Path(path).exists():
        continue
    if not Path(path).exists():
        missing.append(path)
if missing:
    raise SystemExit("Missing initialized sources: " + ", ".join(missing))
print("Source synchronization completed.")
PY
