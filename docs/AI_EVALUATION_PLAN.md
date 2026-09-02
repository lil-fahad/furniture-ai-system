# FurnitureAI AI Evaluation Plan

## Purpose

FurnitureAI treats deterministic geometry and local domain contracts as the source of truth. External AI may refine semantics or produce design guidance, but model changes are promoted only after repeatable evaluation.

## Room-refinement contract

A room-refinement model receives:

- the source floor-plan image;
- the exact set of locally detected room ids;
- geometric metadata such as area and current heuristic label.

It may return only:

- a supplied room id;
- one supported room type;
- a finite confidence in the closed interval 0..1.

Unknown room ids, unsupported types, malformed records, non-finite confidence and schema violations are rejected. The model never creates or changes geometry.

## Structured output migration

The current text-to-JSON parser remains the compatibility baseline until the structured-output path passes regression tests. The target Responses API request uses `text.format.type=json_schema` with a strict schema. The prompt should not duplicate the JSON schema; schema enforcement belongs to the API contract.

## Candidate model evaluation

Do not change the production default from `gpt-5-mini` without an evaluation. Candidate models may include current OpenAI models that support image input and Structured Outputs. Each candidate is scored on the same held-out floor-plan cases.

Record per case:

- exact-room-id validity;
- supported-label validity;
- classification correctness when a human-reviewed label exists;
- abstention/uncertainty behavior;
- latency;
- input/output/cached token usage;
- request status and error class;
- model identifier and evaluation commit.

## Promotion gates

A candidate can replace the default only when:

1. schema-valid output rate does not regress;
2. semantic accuracy improves or remains within the agreed non-inferiority margin;
3. hallucinated room ids remain zero after local validation;
4. latency and estimated cost fit the deployment budget;
5. the fallback path works without an OpenAI key;
6. unit/integration/security tests and CI are green on the exact commit.

## Dataset policy

Evaluation examples must be labeled as synthetic, public-real, supplier-authorized, or private-internal. Synthetic metrics are never reported as real-world accuracy. Images and annotations require provenance and a license or authorization record before reuse.

## Reproducibility

Store evaluation configuration and aggregate results as versioned JSON. Do not store API keys, raw private images, or sensitive prompts in CI artifacts. Use fixed cases and deterministic local validation so model comparisons are repeatable.
