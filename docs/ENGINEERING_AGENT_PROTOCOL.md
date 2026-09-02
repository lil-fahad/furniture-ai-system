# FurnitureAI Engineering Agent Protocol

This protocol defines how autonomous development agents work on the repository.

## Roles

- **Planner:** turns roadmap items into small, testable engineering tasks.
- **Implementer:** changes only the feature branch and preserves existing contracts.
- **Verifier:** checks tests, lint, compilation, security, model integrity, and runtime readiness.
- **ML reviewer:** evaluates datasets/models for task fit, provenance, license, leakage, and reproducibility before adoption.
- **AI reviewer:** validates structured outputs, prompt boundaries, cost/latency controls, and failure behavior.

## Non-negotiable rules

1. Never write directly to `main`.
2. Never expose secrets or place credentials in source, logs, fixtures, prompts, or artifacts.
3. Never invent production supplier data, authorization, pricing, inventory, model metrics, or benchmark results.
4. Never silently replace real data with mock data in production paths.
5. Preserve backward compatibility unless a versioned migration/contract explicitly permits a change.
6. Treat AI output as untrusted and validate it against deterministic application contracts.
7. Keep model artifacts SHA-256 pinned and reject unverifiable artifacts.
8. Require provenance for external datasets, models, catalog fields, images, and evaluations.
9. Do not launch paid GPU/cloud work without quota, budget, duplicate-run, and cancellation safeguards.
10. A feature is incomplete until its regression tests and release gates pass.

## Agent loop

1. Inspect the current branch, exact base/head, tests, and roadmap.
2. Select the smallest high-value change that reduces a production risk.
3. Implement with minimal surface-area change.
4. Add deterministic regression tests for the changed behavior.
5. Run repository lint, compile, tests, security scan, repository audit, and model metadata validation in CI.
6. Review the diff for compatibility, provenance, security, and accidental mock/fake behavior.
7. Record limitations explicitly instead of inventing confidence.
8. Open/update a PR and keep it isolated until all gates pass.
9. If a gate fails, fix the root cause; do not weaken or bypass the gate.
10. Continue to the next roadmap item only after the current change has a verifiable state.

## ML adoption gate

A Hugging Face model or dataset is only a candidate until its task, license, provenance, input/output contract, compute requirements, and evaluation methodology are documented. Production promotion requires a reproducible benchmark against the current baseline and an explicit rollback path.
