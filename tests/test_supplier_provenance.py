from __future__ import annotations

import pytest

from furniture_ai.supplier_provenance import (
    SupplierAuthorizationError,
    authorize_supplier_row,
)


def _authorized_row() -> dict[str, str]:
    return {
        "supplier_id": "supplier-001",
        "source_uri": "https://supplier.example/catalog",
        "source_sha256": "a" * 64,
        "retrieved_at": "2026-09-02T12:00:00Z",
        "authorization_id": "AUTH-001",
        "authorized_by": "supplier-admin",
    }


def test_authorized_supplier_row_returns_immutable_provenance() -> None:
    provenance = authorize_supplier_row(_authorized_row())
    assert provenance.source_uri == "https://supplier.example/catalog"
    assert provenance.source_sha256 == "a" * 64
    assert provenance.authorization_id == "AUTH-001"


@pytest.mark.parametrize(
    "field",
    ["supplier_id", "source_uri", "source_sha256", "retrieved_at", "authorization_id", "authorized_by"],
)
def test_missing_authorization_metadata_is_rejected(field: str) -> None:
    row = _authorized_row()
    row[field] = ""
    with pytest.raises(SupplierAuthorizationError, match="not production-authorized"):
        authorize_supplier_row(row)


def test_invalid_source_uri_is_rejected() -> None:
    row = _authorized_row()
    row["source_uri"] = "supplier.example/catalog"
    with pytest.raises(SupplierAuthorizationError, match="absolute HTTP"):
        authorize_supplier_row(row)


def test_invalid_sha256_is_rejected() -> None:
    row = _authorized_row()
    row["source_sha256"] = "not-a-sha256"
    with pytest.raises(SupplierAuthorizationError, match="SHA-256"):
        authorize_supplier_row(row)


def test_claimed_supplier_fields_do_not_authorize_a_record() -> None:
    row = {
        "Supplier Name": "Example Supplier",
        "Official Website": "https://supplier.example",
        "Image Rights": "Claimed",
        "API": "Available",
    }
    with pytest.raises(SupplierAuthorizationError, match="not production-authorized"):
        authorize_supplier_row(row)
