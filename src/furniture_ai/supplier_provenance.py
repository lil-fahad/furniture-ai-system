from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse


class SupplierAuthorizationError(ValueError):
    """Raised when supplier data is not eligible for production use."""


@dataclass(frozen=True)
class SupplierProvenance:
    """Immutable provenance required for production supplier records."""

    source_uri: str
    source_sha256: str
    retrieved_at: str
    authorization_id: str
    authorized_by: str

    def validate(self) -> None:
        parsed = urlparse(self.source_uri)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise SupplierAuthorizationError(
                "Supplier source_uri must be an absolute HTTP(S) URL"
            )
        if len(self.source_sha256) != 64 or any(
            c not in "0123456789abcdefABCDEF" for c in self.source_sha256
        ):
            raise SupplierAuthorizationError(
                "Supplier source_sha256 must be a 64-character SHA-256 digest"
            )
        try:
            parsed_timestamp = datetime.fromisoformat(
                self.retrieved_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise SupplierAuthorizationError(
                "Supplier retrieved_at must be an ISO-8601 timestamp"
            ) from exc
        if parsed_timestamp.tzinfo is None:
            raise SupplierAuthorizationError(
                "Supplier retrieved_at must include an explicit timezone"
            )
        if not self.authorization_id.strip():
            raise SupplierAuthorizationError("Supplier authorization_id is required")
        if not self.authorized_by.strip():
            raise SupplierAuthorizationError("Supplier authorized_by is required")


REQUIRED_AUTHORIZATION_FIELDS = (
    "supplier_id",
    "source_uri",
    "source_sha256",
    "retrieved_at",
    "authorization_id",
    "authorized_by",
)


def authorize_supplier_row(row: Mapping[str, str]) -> SupplierProvenance:
    """Validate an external supplier row before it can enter a production path.

    Legacy catalog parsing remains backward-compatible; production ingestion
    must explicitly call this gate. Authorization is never inferred from
    supplier claims, URLs, catalog fields, or model scores.
    """
    missing = [
        field for field in REQUIRED_AUTHORIZATION_FIELDS
        if not str(row.get(field, "")).strip()
    ]
    if missing:
        raise SupplierAuthorizationError(
            "Supplier record is not production-authorized; missing: "
            + ", ".join(missing)
        )

    provenance = SupplierProvenance(
        source_uri=str(row["source_uri"]).strip(),
        source_sha256=str(row["source_sha256"]).strip(),
        retrieved_at=str(row["retrieved_at"]).strip(),
        authorization_id=str(row["authorization_id"]).strip(),
        authorized_by=str(row["authorized_by"]).strip(),
    )
    provenance.validate()
    return provenance


def utc_now_iso() -> str:
    """Return a UTC timestamp for ingestion metadata creation."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
