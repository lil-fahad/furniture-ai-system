# Furniture models toolkit — recovered v0.5

This directory records the verified recovery boundary for **FurnitureAI Professional Models v0.5**. It is intentionally isolated from the existing monorepo application because both projects use the `furniture_ai` Python namespace and should not be flattened blindly.

## Validation status

- 45/45 tests passed.
- Python 3.11–3.13 package metadata and packaged JSON resources were validated.
- Three pretrained vision bundles were verified: 421,713,661 bytes.
- Three recovered EfficientNet checkpoints were verified: 113,908,295 bytes.
- 166 recovered images were validated; the historical training dataset remains incomplete.

## Recovery limitation

The original multipart archive is missing logical bytes `0–99,614,719`. The uploaded `.000.part` and `.001(2).part` files are byte-for-byte duplicates of the second logical segment. Missing source/data bytes were not fabricated.

## Artifacts

The full verified archive and the standalone source archive are tracked by filename, size, and SHA-256 in `models.lock.json`. Large model weights are intentionally not committed directly to Git. The complete archive contains its own per-file `release_manifest.json`.

## Integration rule

Treat the recovered toolkit as a standalone package until namespace consolidation is performed deliberately. The recommended next step is to expose its model registry and inference adapters behind the monorepo's existing service interfaces rather than replacing the current application package.
