from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    for model in payload["models"]:
        path = args.manifest.parent / model["path"]
        if path.is_file():
            model["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(model["id"], path.stat().st_size, model["sha256"])
        else:
            print(model["id"], "missing")
    if args.write:
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
