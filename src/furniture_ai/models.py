from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class ModelStatus:
    id: str
    task: str
    path: str
    present: bool
    verified: bool | None
    size_bytes: int | None
    notes: str


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, manifest_path: Path = Path("models/manifest.json")) -> None:
        candidate = Path(manifest_path)
        if candidate.is_file():
            self.manifest_path: Path | None = candidate
            self.base_dir = candidate.parent
            text = candidate.read_text(encoding="utf-8")
        elif candidate == Path("models/manifest.json"):
            self.manifest_path = None
            self.base_dir = Path("models")
            resource = resources.files("furniture_ai.resources").joinpath(
                "model_manifest.json"
            )
            text = resource.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"Model manifest not found: {candidate}")

        payload = json.loads(text)
        entries = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise ValueError("Model manifest must contain a models array")
        self.entries = entries

    def statuses(self) -> list[ModelStatus]:
        statuses: list[ModelStatus] = []
        for entry in self.entries:
            path = self.base_dir / entry["path"]
            present = path.is_file()
            expected = entry.get("sha256")
            verified: bool | None = None
            size: int | None = None
            if present:
                size = path.stat().st_size
                if expected:
                    verified = _sha256(path) == expected
            statuses.append(
                ModelStatus(
                    id=entry["id"],
                    task=entry["task"],
                    path=str(path),
                    present=present,
                    verified=verified,
                    size_bytes=size,
                    notes=entry.get("notes", ""),
                )
            )
        return statuses


def safe_load_room_classifier(path: Path, num_classes: int = 8):
    try:
        import timm
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the vision extra to load local classifiers") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dictionary")
    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model
