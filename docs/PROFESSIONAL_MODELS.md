# Professional model bundle

The repository supports the verified external bundle `FurnitureAI Professional Models v0.4.1 Repaired` without committing large weights to ordinary Git history.

## Why the release is marked repaired

The uploaded `.000.part` and `.001(2).part` files were byte-identical. The genuine first 95 MiB segment of the original v0.4 archive was unavailable, so byte-for-byte reconstruction of the original archive was impossible.

The repaired release keeps every model file that passed CRC, size, and SHA-256 checks. It does not claim the original v0.4 archive hash and does not insert fabricated replacement images.

Verified repaired archive:

- File: `FurnitureAI_Professional_Models_v0.4.1_REPAIRED.zip`
- Size: `564218692` bytes
- SHA-256: `6b9cba60ef3cf86cfc5a04620e00f9c55caa08a82b6184fb1c955be46fb47e10`
- Required model files installed by this repository: 22

The authoritative installation specification is `models/professional/bundle.json`.

## Install

Keep the downloaded ZIP outside the repository, then run:

```bash
python scripts/install_professional_bundle.py \
  /path/to/FurnitureAI_Professional_Models_v0.4.1_REPAIRED.zip
```

The command verifies the whole archive and extracts only allowlisted model files into:

```text
models/professional/installed/
```

Verify an existing installation:

```bash
python scripts/install_professional_bundle.py --verify-installed
python scripts/model_manifest.py
```

## Confirmed models

### DETR ResNet-50

- Source repository recorded in the bundle: `facebook/detr-resnet-50`
- Revision: `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`
- Task: object detection
- License recorded in the bundle: Apache-2.0
- Weight SHA-256: `830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81`

### SAM 2.1 Hiera Tiny

- Source repository recorded in the bundle: `facebook/sam2.1-hiera-tiny`
- Revision: `de431c4043854a71d8101e17995dfe596bf101a5`
- Task: promptable segmentation
- License recorded in the bundle: Apache-2.0
- Weight SHA-256: `48c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7`

### Depth Anything V2 Small

- Source repository recorded in the bundle: `depth-anything/Depth-Anything-V2-Small-hf`
- Revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Task: relative monocular depth estimation
- License recorded in the bundle: Apache-2.0
- Weight SHA-256: `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`

Depth output is relative. It must not be treated as real-world dimensions without calibration or a known scale reference.

## Confirmed trained furniture classifiers

The repaired bundle contains EfficientNet-B0 checkpoints for 12 labels:

`bed`, `cabinet`, `chair`, `desk`, `dresser`, `lamp`, `ottoman`, `rug`, `shelf`, `sofa`, `stool`, and `table`.

The evaluation reports inside the bundle record:

| Checkpoint | Test samples | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Starter-view best checkpoint | 84 | 0.773810 | 0.742626 |
| Multiview best checkpoint | 241 | 0.659751 | 0.651498 |

These values describe the bundle's documented test splits only. They are not a claim of general production accuracy.

## Runtime policy

- The core API remains functional without the bundle.
- Large models are loaded only when an explicit inference path requests them.
- Model files are excluded from normal Git commits.
- Every registered weight has an expected size and SHA-256.
- PyTorch checkpoints are loaded with `weights_only=True`.
- OpenAI remains optional and is not used for deterministic collision or circulation constraints.
