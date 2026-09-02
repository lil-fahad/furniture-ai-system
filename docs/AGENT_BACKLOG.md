# Autonomous Engineering Backlog

Priority order for future agents:

1. **P1 — Floor-plan benchmark contract:** define a versioned evaluation schema for walls, rooms, doors, and windows, including IoU, precision/recall, count accuracy, and abstention/calibration metrics.
2. **P1 — Physical geometry validation:** add scale-aware checks and reject physical-dimension claims when scale is unavailable or inconsistent.
3. **P1 — Supplier authorization:** implement supplier authorization records, scope, expiry/revocation, and provenance before catalog publication.
4. **P1 — Catalog normalization:** normalize SKU/category/dimensions/material/images/MOQ/price/currency/lead time/availability/shipping terms with supplier+SKU deduplication.
5. **P1 — Provenance ledger:** attach source URI/reference, revision timestamp, authorization reference, and ingestion run ID to external records.
6. **P1 — Sourcing constraints:** implement explainable matching for budget, MOQ, dimensions, destination, lead time, authorization, and availability.
7. **P1 — AI evaluation:** create offline regression cases for structured room classification and design briefs; record model/version/cost/latency without exposing prompts or secrets.
8. **P2 — Observability:** structured logs, correlation IDs, secret redaction, latency/error metrics, and ingestion-quality counters.
9. **P2 — Deployment safety:** startup/readiness smoke, migration checks, backup/rollback verification, rate limits, and bounded upload handling.
10. **P2 — Buyer UX:** Arabic/English localization, explainable results, supplier comparison, and RFQ state tracking.

## External ML gate

Use Hugging Face discovery only as candidate research. No model/dataset is promoted solely because it is popular, new, or highly downloaded. License, provenance, task fit, reproducibility, security, resource requirements, and benchmark performance must be verified first.
