# ResPlan training data flow

FurnitureAI uses ResPlan only through a pinned, auditable conversion path.

## Source

- Repository: `https://github.com/m-agour/ResPlan`
- Pinned revision: `e2b78fe069aee1ab1e1828a612743f308e3c32a7`
- Data license: CC BY 4.0
- Code license: MIT
- Published size: 17,000 residential floor plans
- Canonical splits: train 13,053 / validation 1,632 / test 1,632 / augmented 683

The source repository publishes vector geometry and connectivity information, not the original listing images. FurnitureAI therefore treats the generated raster data as **vector-derived floor-plan drawings**, not as real scanned or photographed floor plans.

## Security boundary

The upstream release distributes geometry in a Python pickle. Pickle can execute code while loading and is not a safe training input format.

FurnitureAI therefore enforces this flow:

```text
Pinned ResPlan release
        │
        ▼
SHA-256 verification
        │
        ▼
pickletools inspection (no execution)
        │
        ▼
explicit trusted-pickle opt-in
        │
        ▼
one-time WKT/JSONL export
        │
        ▼
vector rasterization + leakage filtering
        │
        ▼
PNG train / validation / test pairs
        │
        ▼
train_floorplan_segmenter.py
```

The training script must never consume the pickle directly.

## 1. Obtain and pin the upstream files

Checkout the exact configured revision, then extract `ResPlan.pkl` and retain the accompanying `split.json`.

Calculate the local pickle hash:

```bash
sha256sum ResPlan.pkl
```

Do not copy a hash from an unrelated mirror or a different revision.

## 2. Inspect without unpickling

```bash
python scripts/convert_trusted_resplan_pickle.py \
  ResPlan.pkl \
  split.json \
  --expected-sha256 <EXACT_LOCAL_SHA256> \
  --inspect-only
```

This uses `pickletools` and does not execute the pickle payload. A hash mismatch fails closed.

## 3. Convert only after provenance review

After confirming that the file is the expected upstream artifact:

```bash
python scripts/convert_trusted_resplan_pickle.py \
  ResPlan.pkl \
  split.json \
  --expected-sha256 <EXACT_LOCAL_SHA256> \
  --allow-trusted-pickle \
  --output data/resplan/resplan.safe.jsonl
```

The converter emits only scalar metadata and WKT strings. It also records the source hash, pickle inspection summary and split counts in a companion metadata file.

## 4. Prepare segmentation data

```bash
python training/prepare_resplan_segmentation.py \
  data/resplan/resplan.safe.jsonl \
  --output data/resplan/segmentation \
  --size 512
```

The output layout is compatible with the production-hardened segmentation trainer:

```text
data/resplan/segmentation/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
├── masks/
│   ├── train/
│   ├── validation/
│   └── test/
├── samples.jsonl
└── manifest.json
```

Class ids:

| Id | Class |
|---:|---|
| 0 | background |
| 1 | room |
| 2 | wall |
| 3 | door |
| 4 | window |

The generated input image is a neutral architectural rendering. It is not semantic RGB labeling and is not represented as real imagery.

## Leakage policy

ResPlan's own documentation reports near-duplicate plans, including some test plans with near-duplicates in training. FurnitureAI therefore adds a second leakage gate.

Each generated semantic mask is normalized, downsampled to a coarse 32×32 representation, canonicalized across rotations and reflections, and hashed. Exact signature collisions:

- are removed within each split;
- cause validation/test records matching training to be removed;
- are recorded in the generated manifest.

This is deliberately described as a **coarse leakage heuristic**, not a complete near-duplicate detector. A stronger geometry-similarity audit is still required for publishable production benchmark claims.

The `augmented` split is excluded by default. When `--include-augmented` is explicitly supplied, augmented records may enter training only; they never enter validation or test.

## 5. Train

```bash
python training/train_floorplan_segmenter.py \
  data/resplan/segmentation \
  --classes 5 \
  --epochs 30 \
  --batch-size 8 \
  --device cuda \
  --amp \
  --output models/floorplan_segmenter/resplan-unet.pt
```

The trainer records validation/test mIoU, Dice, pixel accuracy and per-class metrics, retains the best validation checkpoint, and produces resumable training state.

## Promotion rules

A ResPlan-trained model is not automatically a production model. Promotion requires all of the following:

1. source revision, license, local source SHA-256 and generated manifest retained;
2. no unresolved cross-split leakage finding;
3. held-out metrics from the exact promoted checkpoint;
4. SHA-256 pin for the final model artifact;
5. comparison with the deterministic OpenCV baseline;
6. evaluation on a separate real-raster floor-plan set before making claims about scanned/photo inputs;
7. documented regional-bias limitations, because ResPlan is drawn from South Asian residential layouts.

Do not report smoke-test or vector-derived validation metrics as production accuracy.
