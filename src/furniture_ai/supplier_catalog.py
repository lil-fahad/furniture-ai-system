from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from urllib.parse import urlsplit, urlunsplit


def normalize_text(value: object) -> str:
    """Return a deterministic, whitespace-normalized Unicode string."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def normalize_url(value: object) -> str:
    """Normalize a URL without changing its meaning or inventing a scheme."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.casefold(), parts.netloc.casefold(), path, parts.query, ""))


def normalize_supplier_id(value: object) -> str:
    return normalize_text(value).replace(" ", "-")


def supplier_row_key(row: Mapping[str, object]) -> str:
    """Build a stable identity key; do not use model scores or descriptive claims."""
    supplier_id = normalize_supplier_id(row.get("supplier_id") or row.get("Supplier ID"))
    product_id = normalize_text(
        row.get("product_id") or row.get("Product ID") or row.get("SKU")
    )
    source_url = normalize_url(row.get("source_url") or row.get("Official Website"))
    supplier_name = normalize_text(row.get("supplier_name") or row.get("Supplier Name"))
    category = normalize_text(row.get("category") or row.get("Main Category"))
    identity = "|".join((supplier_id, product_id, source_url, supplier_name, category))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NormalizedSupplierRow:
    """Immutable normalized view; original fields remain untouched elsewhere."""

    identity_key: str
    values: Mapping[str, str]


def normalize_supplier_row(row: Mapping[str, object]) -> NormalizedSupplierRow:
    values = {str(key): normalize_text(value) for key, value in row.items()}
    for key in ("source_url", "Official Website", "profile_link", "Profile Link"):
        if key in row:
            values[key] = normalize_url(row[key])
    return NormalizedSupplierRow(
        identity_key=supplier_row_key(row),
        values=MappingProxyType(values),
    )


def deduplicate_supplier_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Remove exact supplier-record duplicates deterministically, preserving first-seen order.

    Rows are not merged: potentially conflicting records remain distinct when their
    identity keys differ, so this utility cannot silently overwrite supplier facts.
    """
    seen: set[str] = set()
    result: list[dict[str, object]] = []
    for row in rows:
        key = supplier_row_key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result
