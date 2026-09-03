from __future__ import annotations

import pytest

from furniture_ai.supplier_ingestion import ingest_authorized_supplier_rows
from furniture_ai.supplier_provenance import SupplierAuthorizationError


def _row(*, supplier_id: str = "supplier-001", sku: str = "SKU-1") -> dict[str, object]:
    return {
        "supplier_id": supplier_id,
        "product_id": sku,
        "supplier_name": "Verified Supplier",
        "category": "Sofa",
        "source_url": "https://supplier.example/products/sku-1",
        "source_uri": "https://supplier.example/catalog.csv",
        "source_sha256": "a" * 64,
        "retrieved_at": "2026-09-03T00:00:00Z",
        "authorization_id": "AUTH-001",
        "authorized_by": "supplier-admin",
        "price": "1250.00",
        "currency": "SAR",
    }


def test_authorized_batch_returns_immutable_records() -> None:
    records = ingest_authorized_supplier_rows([_row()])
    assert len(records) == 1
    assert records[0].supplier_id == "supplier-001"
    assert records[0].provenance.authorization_id == "AUTH-001"
    with pytest.raises(TypeError):
        records[0].values["price"] = "0"  # type: ignore[index]


def test_exact_duplicate_identity_is_removed_after_authorization() -> None:
    row = _row()
    duplicate = dict(row)
    records = ingest_authorized_supplier_rows([row, duplicate])
    assert len(records) == 1


def test_one_unauthorized_row_rejects_entire_batch() -> None:
    good = _row()
    bad = _row(supplier_id="supplier-002", sku="SKU-2")
    bad["authorization_id"] = ""
    with pytest.raises(SupplierAuthorizationError, match="not production-authorized"):
        ingest_authorized_supplier_rows([good, bad])


def test_conflicting_identity_is_not_silently_merged() -> None:
    first = _row()
    second = _row(sku="SKU-2")
    second["price"] = "1400.00"
    records = ingest_authorized_supplier_rows([first, second])
    assert len(records) == 2
    assert records[0].identity_key != records[1].identity_key


def test_claimed_fields_cannot_bypass_authorization_gate() -> None:
    row = {
        "supplier_id": "supplier-claim",
        "supplier_name": "Claimed Supplier",
        "Image Rights": "Claimed",
        "API": "Available",
    }
    with pytest.raises(SupplierAuthorizationError):
        ingest_authorized_supplier_rows([row])
