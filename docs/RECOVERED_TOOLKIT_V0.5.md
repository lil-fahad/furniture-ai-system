# FurnitureAI recovered toolkit v0.5

The uploaded split archive was inspected and partially reconstructed. The supplied file named `.000.part` is byte-identical to the later `.001(2).part`, so the original first 95 MiB segment is still unavailable. The original archive must therefore not be represented as complete.

A separate recovered v0.5 toolkit was rebuilt from the readable ZIP directory and surviving members. It contains the recovered Python source, tests, documentation, three verified local classifier checkpoints, and three verified pretrained vision-model bundles. Dataset assets are explicitly marked as partial recovery; missing records were not fabricated.

## Verified local results

- 43 unit tests passed.
- Python source and tests pass `compileall`.
- The wheel builds and packaged JSON resources load outside the repository working directory.
- Recovered checkpoints: 3 files, 113,908,295 bytes, SHA-256 verified.
- Pretrained local models: 3 bundles, 421,713,661 bytes, size and SHA-256 verified.
- Optional Stable Diffusion inpainting snapshot is not installed and is not claimed as present.
- Full recovered archive SHA-256: `4079d5fd42a6ebe065c5e69813aaddae57941119a011266f449afd984439f58e`.
- Source-only archive SHA-256: `93afa4cce37083a93d225c25ff797ecd83e079d6fe9c190ba0771fa9112bf210`.
- Wheel SHA-256: `d8e230daefe4c11482667cc8c7b8492e64efadb60b898a640e1230cde5a4b992`.

## Hardening merged into this repository

This update carries forward the reusable fixes discovered during recovery:

- catalog and model-manifest defaults are packaged in the wheel;
- explicit missing paths fail instead of silently falling back;
- uploaded image MIME type is checked against decoded image content;
- pixel limits are applied before RGB conversion;
- model SHA-256 verification streams files instead of reading entire weights into memory;
- segmentation masks use nearest-neighbor resizing;
- classifier validation uses deterministic, non-augmented transforms;
- CI builds and smoke-tests the installed wheel from outside the source tree.
