# FurnitureAI recovered model pack v0.5

This records the verified recovery and repair of the uploaded multipart archive `FurnitureAI_Professional_Models_v0.4`.

## Recovery status

The true first logical 99,614,720-byte segment was not available. The uploaded files named `.000.part` and `.001(2).part` were byte-identical, so the original archive cannot be reconstructed byte-for-byte. The v0.5 output is therefore explicitly marked **recovered upgrade**, and its dataset status is `partial_recovery`.

## Repairs and upgrades

- Reconstructed missing packaging, configuration, environment, and documentation files.
- Embedded configuration resources so installed wheels work outside the source checkout.
- Added SHA-256 verification for pretrained bundles and recovered EfficientNet checkpoints.
- Restricted PyTorch checkpoint loading to `weights_only=True`.
- Hardened ZIP/TAR inspection against traversal, links, collisions, encrypted entries, special files, decompression bombs, and oversized archives.
- Added deterministic source/full release creation with secret, cache, and partial-file exclusions.
- Updated optional OpenAI labeling to the Responses API with structured Pydantic output, explicit model selection, billable opt-in, `store=False`, image limits, path containment, and privacy-preserving safety identifiers.
- Hardened the local service for non-loopback binding with token authentication and explicit CORS origins.

## Validation

- Unit tests: **45/45 passed**.
- Python compile check: passed.
- Wheel build and clean-target install: passed.
- Pretrained model verification: passed, 421,713,661 bytes.
- Recovered checkpoint verification: passed, 113,908,295 bytes.
- Full ZIP CRC test and release manifest verification: passed.

The model weights are intentionally not committed to GitHub. Use `scripts/verify_recovered_model_pack.py` with the separately delivered artifacts and the committed checksum file.
