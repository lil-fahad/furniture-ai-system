from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ALLOWED_TIERS = {"core", "experimental", "legacy", "private", "blocked"}


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    repository: str
    commit: str
    visibility: str
    tier: str
    capabilities: tuple[str, ...]
    review: str
    path: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Source":
        return cls(
            id=str(raw["id"]),
            repository=str(raw["repository"]),
            commit=str(raw["commit"]),
            visibility=str(raw["visibility"]),
            tier=str(raw["tier"]),
            capabilities=tuple(str(value) for value in raw.get("capabilities", [])),
            review=str(raw.get("review", "")),
            path=str(raw["path"]) if raw.get("path") else None,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "repository": self.repository,
            "commit": self.commit,
            "path": self.path,
            "visibility": self.visibility,
            "tier": self.tier,
            "capabilities": list(self.capabilities),
            "review": self.review,
        }


class RegistryError(ValueError):
    """Raised when the locked source registry is invalid."""


class SourceRegistry:
    def __init__(self, sources: tuple[Source, ...], metadata: dict[str, Any]) -> None:
        self.sources = sources
        self.metadata = metadata
        self._validate()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SourceRegistry":
        configured = os.getenv("FURNITURE_SOURCES_FILE")
        source_path = Path(path or configured or Path(__file__).resolve().parents[2] / "sources.lock.json")
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RegistryError(f"Source registry not found: {source_path}") from exc
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid JSON in source registry: {source_path}") from exc

        raw_sources = payload.get("sources")
        if not isinstance(raw_sources, list):
            raise RegistryError("sources.lock.json must contain a sources array")

        sources = tuple(Source.from_dict(item) for item in raw_sources)
        metadata = {key: value for key, value in payload.items() if key != "sources"}
        return cls(sources=sources, metadata=metadata)

    def _validate(self) -> None:
        ids: set[str] = set()
        repositories: set[str] = set()
        paths: set[str] = set()

        for source in self.sources:
            if source.id in ids:
                raise RegistryError(f"Duplicate source id: {source.id}")
            ids.add(source.id)

            if source.repository in repositories:
                raise RegistryError(f"Duplicate repository: {source.repository}")
            repositories.add(source.repository)

            if not _COMMIT_RE.fullmatch(source.commit):
                raise RegistryError(f"Invalid commit SHA for {source.id}")
            if source.tier not in _ALLOWED_TIERS:
                raise RegistryError(f"Invalid tier for {source.id}: {source.tier}")
            if source.visibility not in {"public", "private"}:
                raise RegistryError(f"Invalid visibility for {source.id}")

            if source.tier == "blocked":
                if source.path is not None:
                    raise RegistryError(f"Blocked source must not have an import path: {source.id}")
                continue

            if not source.path:
                raise RegistryError(f"Importable source is missing path: {source.id}")
            if source.path in paths:
                raise RegistryError(f"Duplicate source path: {source.path}")
            if Path(source.path).is_absolute() or ".." in Path(source.path).parts:
                raise RegistryError(f"Unsafe source path: {source.path}")
            paths.add(source.path)

    def list(
        self,
        *,
        tier: str | None = None,
        visibility: str | None = None,
        include_blocked: bool = False,
    ) -> list[Source]:
        results = self.sources
        if not include_blocked:
            results = tuple(source for source in results if source.tier != "blocked")
        if tier:
            results = tuple(source for source in results if source.tier == tier)
        if visibility:
            results = tuple(source for source in results if source.visibility == visibility)
        return list(results)

    def capabilities(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for source in self.sources:
            if source.tier == "blocked":
                continue
            for capability in source.capabilities:
                index.setdefault(capability, []).append(source.id)
        return {key: sorted(values) for key, values in sorted(index.items())}

    def summary(self) -> dict[str, Any]:
        tiers: dict[str, int] = {}
        for source in self.sources:
            tiers[source.tier] = tiers.get(source.tier, 0) + 1
        return {
            "total_sources": len(self.sources),
            "importable_sources": sum(source.tier != "blocked" for source in self.sources),
            "blocked_sources": sum(source.tier == "blocked" for source in self.sources),
            "tiers": dict(sorted(tiers.items())),
            "schema_version": self.metadata.get("schema_version"),
            "generated_at": self.metadata.get("generated_at"),
        }
