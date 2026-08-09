from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelStatus:
    id: str
    name: str
    task: str
    architecture: str | None
    path: str
    present: bool
    verified: bool | None
    size_bytes: int | None
    expected_size_bytes: int | None
    source: str | None
    license: str | None
    required: bool
    bundle: str | None
    notes: str


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path
        self.base_dir = manifest_path.parent
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.schema_version = int(payload.get("schema_version", 1))
        self.entries = payload["models"]
        ids = [str(entry["id"]) for entry in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("Model manifest contains duplicate IDs")

    def statuses(self) -> list[ModelStatus]:
        statuses: list[ModelStatus] = []
        for entry in self.entries:
            path = self.base_dir / entry["path"]
            present = path.is_file()
            expected_hash = entry.get("sha256")
            expected_size = entry.get("size_bytes")
            verified: bool | None = None
            size: int | None = None
            if present:
                size = path.stat().st_size
                checks: list[bool] = []
                if expected_size is not None:
                    checks.append(size == int(expected_size))
                if expected_hash:
                    checks.append(_sha256_file(path) == str(expected_hash))
                verified = all(checks) if checks else None
            statuses.append(
                ModelStatus(
                    id=str(entry["id"]),
                    name=str(entry.get("name", entry["id"])),
                    task=str(entry["task"]),
                    architecture=(
                        str(entry["architecture"]) if entry.get("architecture") else None
                    ),
                    path=str(path),
                    present=present,
                    verified=verified,
                    size_bytes=size,
                    expected_size_bytes=(
                        int(expected_size) if expected_size is not None else None
                    ),
                    source=str(entry["source"]) if entry.get("source") else None,
                    license=str(entry["license"]) if entry.get("license") else None,
                    required=bool(entry.get("required", False)),
                    bundle=str(entry["bundle"]) if entry.get("bundle") else None,
                    notes=str(entry.get("notes", "")),
                )
            )
        return statuses


def _infer_num_classes(state_dict: dict) -> int | None:
    """Infer the classifier head width from state_dict tensor shapes.

    Looks for 2-D weight tensors on classifier/fc/head keys and returns the
    output dimension of the deepest one. Returns None when no candidate is
    found, letting the caller decide on a fallback.
    """
    candidates: list[tuple[int, str]] = []
    for key, tensor in state_dict.items():
        if not key.endswith("weight"):
            continue
        if not any(token in key for token in ("classifier", "fc", "head")):
            continue
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) == 2:
            candidates.append((key.count("."), key))
    if not candidates:
        return None
    _, key = max(candidates)
    return int(state_dict[key].shape[0])


def safe_load_room_classifier(path: Path, num_classes: int | None = None):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install the vision extra to load local classifiers") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must contain a state dictionary")
    state_dict = checkpoint.get(
        "model_state",
        checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint)),
    )
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a valid state dictionary")

    classes = checkpoint.get("classes")
    resolved_classes = (
        num_classes
        or (len(classes) if isinstance(classes, list) else 0)
        or _infer_num_classes(state_dict)
        or 8
    )
    architecture = str(checkpoint.get("architecture", "tf_efficientnet_b0"))
    # Detect the checkpoint layout from its keys first: torchvision
    # EfficientNet uses ``features.``/``classifier.N`` naming, timm does not.
    torchvision_layout = any(key.startswith("features.") for key in state_dict)
    if torchvision_layout:
        try:
            from torchvision.models import efficientnet_b0
        except ImportError as exc:
            raise RuntimeError("Install the vision extra to load EfficientNet") from exc
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, resolved_classes)
    else:
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError("Install the vision extra to load timm classifiers") from exc
        model = timm.create_model(architecture, pretrained=False, num_classes=resolved_classes)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.class_labels = tuple(classes) if isinstance(classes, list) else tuple()
    return model


_MODEL_CACHE: dict[tuple[str, int | None, int, int], object] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def load_room_classifier_cached(path: Path, num_classes: int | None = None):
    """Thread-safe lazy singleton wrapper around :func:`safe_load_room_classifier`.

    Repeated API calls must not re-read and re-build model weights. The cache
    key includes the resolved path, the requested class count, and the file's
    size/mtime so a replaced checkpoint is reloaded instead of served stale.
    """
    resolved = str(Path(path).resolve())
    stat = Path(path).stat()  # raises FileNotFoundError for missing files, as before
    key = (resolved, num_classes, stat.st_size, stat.st_mtime_ns)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
    # Load outside the lock so concurrent first-time callers do not serialize
    # on the (slow) weight load; setdefault keeps exactly one winner.
    model = safe_load_room_classifier(path, num_classes=num_classes)
    with _MODEL_CACHE_LOCK:
        return _MODEL_CACHE.setdefault(key, model)


def clear_model_cache() -> None:
    """Drop all cached model instances (primarily for tests)."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
