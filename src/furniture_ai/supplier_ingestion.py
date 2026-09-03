from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from furniture_ai.supplier_catalog import supplier_row_key
from furniture_ai.supplier_provenance import SupplierProvenance, authorize_supplier_row


@dataclass(frozen=True)
class AuthorizedSupplierRecord:
    identity_key: str
    supplier_id: str
    provenance: SupplierProvenance
    values: Mapping[str, object]


def ingest_authorized_supplier_rows(
    rows: Iterable[Mapping[str, object]],
) -> list[AuthorizedSupplierRecord]:
    """Validate and deduplicate a supplier batch for production ingestion.

    The batch is fail-closed: every row is authorization-validated before any
    result is returned. Exact duplicate identities are removed only after
    authorization succeeds, and conflicting identities are never merged.
    """
    materialized = [dict(row) for row in rows]
    authorized: list[tuple[dict[str, object], SupplierProvenance]] = []

    for row in materialized:
        string_view = {str(key): str(value) for key, value in row.items()}
        provenance = authorize_supplier_row(string_view)
        authorized.append((row, provenance))

    seen: set[str] = set()
    result: list[AuthorizedSupplierRecord] = []
    for row, provenance in authorized:
        identity_key = supplier_row_key(row)
        if identity_key in seen:
            continue
        seen.add(identity_key)
        result.append(
            AuthorizedSupplierRecord(
                identity_key=identity_key,
                supplier_id=str(row["supplier_id"]).strip(),
                provenance=provenance,
                values=MappingProxyType(dict(row)),
            )
        )
    return result
