from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _optional_positive_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer when provided")
    return value


def _freeze_json(value: object, field: str) -> object:
    if isinstance(value, dict):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{field} keys must be non-empty strings")
            frozen[key] = _freeze_json(item, f"{field}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, list):
        return tuple(_freeze_json(item, field) for item in value)
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise ValueError(f"{field} contains an unsupported JSON value")


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    frozen = _freeze_json(value, field)
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{field} must be a JSON object")
    return frozen


def _relative_path(
    value: object,
    field: str,
    *,
    model_artifact: bool = False,
) -> str:
    raw = _required_string(value, field)
    if "\\" in raw:
        raise ValueError(f"{field} must use forward-slash repository paths")
    path = PurePosixPath(raw)
    unsafe_part = any(part in {"", ".", ".."} for part in path.parts)
    if path.is_absolute() or not path.parts or unsafe_part:
        raise ValueError(f"{field} must be a safe relative repository path")
    if model_artifact:
        if path.parts[0] == "data":
            raise ValueError("Model artifact binaries must live outside data/")
        if path.parts[0] != "models":
            raise ValueError("Model artifact paths must live under models/")
    return raw


def _sha256(value: object, field: str) -> str:
    sha = _required_string(value, field).lower()
    if not _SHA256_RE.fullmatch(sha):
        raise ValueError(f"{field} must be a 64-character SHA-256 hex digest")
    return sha


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty JSON array")
    items = tuple(_required_string(item, field) for item in value)
    if len(set(items)) != len(items):
        raise ValueError(f"{field} must not contain duplicates")
    return items


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    id: str
    name: str
    kind: str
    path: str | None
    availability: str | None
    record_count: int | None
    provenance: Mapping[str, object]
    notes: str | None


@dataclass(frozen=True, slots=True)
class TrainedModelRecord:
    id: str
    task: str
    architecture: str
    artifact_path: str
    sha256: str
    dataset_ids: tuple[str, ...]
    status: str
    size_bytes: int | None
    metrics_path: str | None
    trained_at: str | None
    source: str | None
    metrics: Mapping[str, object]
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelDatasetRegistry:
    schema_version: int
    datasets: tuple[DatasetRecord, ...]
    models: tuple[TrainedModelRecord, ...]
    policy: Mapping[str, object]

    @classmethod
    def load(cls, path: Path) -> ModelDatasetRegistry:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read model/dataset registry: {path}") from exc
        if not isinstance(payload, dict):
            raise ValueError("Model/dataset registry must contain a JSON object")
        if payload.get("schema_version") != 1:
            raise ValueError("Model/dataset registry schema_version must be 1")

        raw_datasets = payload.get("datasets")
        raw_models = payload.get("models")
        if not isinstance(raw_datasets, list) or not isinstance(raw_models, list):
            raise ValueError("Model/dataset registry datasets and models must be JSON arrays")

        datasets = tuple(_parse_dataset(item) for item in raw_datasets)
        models = tuple(_parse_model(item) for item in raw_models)
        _require_unique_ids(datasets, "dataset")
        _require_unique_ids(models, "model")

        dataset_ids = {dataset.id for dataset in datasets}
        for model in models:
            unknown = sorted(set(model.dataset_ids) - dataset_ids)
            if unknown:
                raise ValueError(
                    f"Model {model.id!r} references unknown dataset IDs: "
                    + ", ".join(unknown)
                )

        return cls(
            schema_version=1,
            datasets=datasets,
            models=models,
            policy=_mapping(payload.get("policy"), "policy"),
        )

    def trained_models_for_dataset(
        self,
        dataset_id: str,
    ) -> tuple[TrainedModelRecord, ...]:
        if dataset_id not in {dataset.id for dataset in self.datasets}:
            raise KeyError(dataset_id)
        return tuple(model for model in self.models if dataset_id in model.dataset_ids)

    def datasets_for_model(self, model_id: str) -> tuple[DatasetRecord, ...]:
        model = next((item for item in self.models if item.id == model_id), None)
        if model is None:
            raise KeyError(model_id)
        by_id = {dataset.id: dataset for dataset in self.datasets}
        return tuple(by_id[dataset_id] for dataset_id in model.dataset_ids)


def _parse_dataset(value: object) -> DatasetRecord:
    if not isinstance(value, dict):
        raise ValueError("Each dataset registry entry must be a JSON object")
    path_value = value.get("path")
    return DatasetRecord(
        id=_required_string(value.get("id"), "dataset.id"),
        name=_required_string(value.get("name"), "dataset.name"),
        kind=_required_string(value.get("kind"), "dataset.kind"),
        path=None if path_value is None else _relative_path(path_value, "dataset.path"),
        availability=_optional_string(value.get("availability"), "dataset.availability"),
        record_count=_optional_positive_int(value.get("record_count"), "dataset.record_count"),
        provenance=_mapping(value.get("provenance"), "dataset.provenance"),
        notes=_optional_string(value.get("notes"), "dataset.notes"),
    )


def _parse_model(value: object) -> TrainedModelRecord:
    if not isinstance(value, dict):
        raise ValueError("Each trained-model registry entry must be a JSON object")
    status = _required_string(value.get("status"), "model.status")
    if status != "trained":
        raise ValueError("Model/dataset registry accepts only status='trained' artifacts")
    metrics_path_value = value.get("metrics_path")
    raw_limitations = value.get("limitations", [])
    if not isinstance(raw_limitations, list):
        raise ValueError("model.limitations must be a JSON array")
    limitations = tuple(
        _required_string(item, "model.limitations") for item in raw_limitations
    )
    return TrainedModelRecord(
        id=_required_string(value.get("id"), "model.id"),
        task=_required_string(value.get("task"), "model.task"),
        architecture=_required_string(value.get("architecture"), "model.architecture"),
        artifact_path=_relative_path(
            value.get("artifact_path"),
            "model.artifact_path",
            model_artifact=True,
        ),
        sha256=_sha256(value.get("sha256"), "model.sha256"),
        dataset_ids=_string_tuple(value.get("dataset_ids"), "model.dataset_ids"),
        status=status,
        size_bytes=_optional_positive_int(value.get("size_bytes"), "model.size_bytes"),
        metrics_path=(
            None
            if metrics_path_value is None
            else _relative_path(metrics_path_value, "model.metrics_path")
        ),
        trained_at=_optional_string(value.get("trained_at"), "model.trained_at"),
        source=_optional_string(value.get("source"), "model.source"),
        metrics=_mapping(value.get("metrics"), "model.metrics"),
        limitations=limitations,
    )


def _require_unique_ids(
    records: tuple[DatasetRecord, ...] | tuple[TrainedModelRecord, ...],
    kind: str,
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for record in records:
        if record.id in seen:
            duplicates.add(record.id)
        seen.add(record.id)
    if duplicates:
        raise ValueError(f"Duplicate {kind} IDs: {', '.join(sorted(duplicates))}")
