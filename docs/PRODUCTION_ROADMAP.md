# FurnitureAI Production Roadmap

## Product goal

FurnitureAI is being evolved into a production-grade, supplier-aware interior-design platform while preserving existing API contracts and workflows.

## Architecture principles

1. **Deterministic geometry is the source of truth.** AI may refine semantics or produce design guidance, but it must not bypass geometric validation.
2. **Provenance is mandatory.** Every external catalog field, image, model, and training dataset must have a traceable source and revision.
3. **Authorization precedes publication.** Supplier data is not production-eligible until the supplier has explicitly authorized the relevant catalog/feed/image usage.
4. **Fail closed in production.** Invalid AI output, missing supplier authorization, unsafe uploads, invalid geometry, or missing security configuration must be rejected rather than silently substituted.
5. **No fake confidence.** Synthetic, annotation-only, or tiny expert-curated evaluations are labeled as such and never presented as production accuracy.
6. **No uncontrolled spend.** GPU/cloud jobs require explicit quota, budget, duplicate-run, and failure checks.
7. **Backward compatibility.** Existing API endpoints and persisted data are changed through versioned migrations/contracts, not breaking rewrites.

## Delivery sequence

### Phase 1 — Production hardening
- Python 3.11/3.12 CI.
- Lint, compile, tests, model metadata, secret scan, repository audit.
- Dependency vulnerability scanning where supported.
- API startup/health/readiness smoke checks.
- Regression tests for AI output validation and upload security.
- Correlation IDs and safe error handling.

**Exit gate:** all required CI checks pass on the exact PR head; no unresolved review/security findings.

### Phase 2 — Floor-plan intelligence
- Benchmark room, wall, door, and window extraction against real labeled plans.
- Add confidence calibration and abstention for ambiguous cases.
- Validate physical dimensions only when scale is available.
- Strengthen collision, circulation, doorway, clearance, and accessibility constraints.
- Track model/data/version provenance for every evaluation.

### Phase 3 — Authorized supplier catalog
- Supplier authorization entity and scope.
- Normalize SKU, category, dimensions, materials, images, MOQ, price, currency, lead time, stock/status, shipping terms.
- Store provenance and source revision for every record.
- Deduplicate by supplier + SKU.
- Maintain immutable source snapshots where licensing permits.
- Reject unauthorized or incomplete production records.
- Keep development fixtures explicitly separated from production catalog data.

### Phase 4 — Sourcing intelligence
- Explainable supplier/product matching.
- Constraints for budget, MOQ, lead time, dimensions, destination, authorization and availability.
- Buyer RFQ creation.
- Supplier response tracking and comparison.
- Ranking evaluation with held-out/repeated validation before production claims.

### Phase 5 — AI design engine
- Constrained structured outputs.
- AI room classification validated against local room IDs/types.
- Design briefs grounded in validated floor-plan JSON.
- Model/version/cost/latency telemetry.
- Offline evaluation set and regression suite.
- Explicit model fallback configuration; no silent provider substitution.

### Phase 6 — Operations and deployment
- Structured logs with correlation IDs and secret redaction.
- Metrics and alerting for latency, errors, AI calls, ingestion quality, and booking failures.
- Database migrations and backups.
- Rate limits and request-size limits.
- Deployment smoke tests.
- Cloud GPU quota/cost checks and duplicate-run locks.

### Phase 7 — Buyer experience
- Arabic/English localization.
- Clear upload and analysis progress.
- Explainable room/layout results.
- Supplier/product filters and comparison.
- RFQ workflow and status tracking.
- Accessible error and validation states.

## Release gates

A release is not considered production-ready until:

- the exact release commit has green required CI checks;
- the API starts successfully;
- `/health` and `/ready` return expected states;
- unit/integration/security tests pass;
- model manifests and SHA-256 pins validate;
- secret scanning is clean;
- no unresolved security/review conversations remain;
- supplier catalog records satisfy authorization and provenance rules;
- real-vs-mock boundaries are verified;
- deployment smoke tests pass;
- rollback and backup procedures are documented.
