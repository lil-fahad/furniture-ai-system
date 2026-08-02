from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
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


class ModelRegistry:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        self.entries = json.loads(manifest_path.read_text(encoding="utf-8"))["models"]

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
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    verified = digest == expected
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
