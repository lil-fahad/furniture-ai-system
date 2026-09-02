# Supplier Production Authorization & Provenance

Supplier data is **not production-eligible by default**.

The existing supplier CSV/ranker remains backward-compatible for current development and evaluation workflows. Production ingestion must explicitly validate each external record with `authorize_supplier_row()` before persistence, indexing, matching, ranking, or customer-facing use.

## Required authorization metadata

Every production supplier record must carry:

- `supplier_id`
- `source_uri` — absolute HTTP(S) source
- `source_sha256` — 64-character SHA-256 digest of the retrieved source artifact
- `retrieved_at` — ISO-8601 retrieval timestamp
- `authorization_id` — auditable authorization reference
- `authorized_by` — accountable authorizer identity/role

Authorization is never inferred from:

- an official-looking website;
- supplier claims such as `Available`, `Confirmed`, or `Claimed`;
- an API/catalog field;
- an ML score or ranking;
- an LLM output.

## Production rule

If required authorization/provenance metadata is missing or malformed, the record must be rejected from the production path with an explicit error. Do not silently fall back to an unauthorised record.

## Next increment

A future ingestion adapter should attach these fields at acquisition time, persist the provenance record alongside normalized supplier/product data, and expose provenance identifiers to downstream matching and audit logs without exposing secrets or private credentials.
