# FurnitureAI style training pipeline v2

This pipeline turns the project's seven-style ImageFolder corpus into a
versioned, leakage-resistant dataset and a reproducibly evaluated classifier.
The supported labels remain:

- `minimalist`
- `scandinavian`
- `industrial`
- `bohemian`
- `luxury`
- `mid_century_modern`
- `japandi`

## Why v2 exists

The original trainer created a deterministic stratified train/validation split
from one ImageFolder. That is useful for smoke tests, but visually identical or
near-identical images can land on both sides of a random split and inflate
validation scores. It also has no untouched test split.

The v2 path therefore performs dataset quality work before any model training:

1. Decode and validate every supported image.
2. Compute a file SHA-256 and a 128-bit perceptual difference hash.
3. Cluster exact and conservative near duplicates.
4. Remove same-label duplicates.
5. Quarantine duplicate clusters that disagree on the class label.
6. Exclude upstream records marked `review_required` by default.
7. Create deterministic class-stratified `train`, `validation`, and `test` sets.
8. Write an immutable dataset fingerprint derived from accepted image SHA-256,
   class, and split assignments.

The source ImageFolder is never modified.

## Prepare a local dataset

```bash
python scripts/prepare_style_dataset.py data/styles \
  --source-manifest data/style_sources.jsonl \
  --output data/styles_prepared
```

Important output files:

- `summary.json` — counts, duplicate/conflict statistics, split distribution,
  and `dataset_fingerprint`.
- `manifest.jsonl` — one record per scanned image, including SHA-256,
  perceptual hash, status, split, duplicate relationship, and matching source
  provenance when available.
- `conflicts.json` — clusters whose duplicate images carry conflicting labels.
- `rejected.jsonl` — unreadable or policy-rejected image files.
- `train/`, `validation/`, `test/` — model-ready ImageFolder splits.

A label-conflict cluster makes the command fail closed unless
`--allow-label-conflicts` is supplied. The managed Vertex path uses that flag
because conflicting clusters are already quarantined and retained as an audit
artifact.

## Train

```bash
python training/train_style_classifier.py data/styles_prepared \
  --device cuda \
  --precision auto \
  --class-balance sampler \
  --epochs 30 \
  --batch-size 64 \
  --output models/style_classifier/efficientnet_b0.pth
```

The trainer requires the prepared dataset metadata by default. It selects the
best epoch using validation macro-F1 instead of raw accuracy and keeps the test
split untouched until model selection is complete.

The final checkpoint records dataset fingerprint, manifest SHA-256, class
order, architecture, image size, split sizes, validation metrics, and test
metrics. A sibling `*.metrics.json` report includes:

- accuracy;
- balanced accuracy;
- macro-F1;
- negative log likelihood;
- 15-bin expected calibration error;
- per-class precision, recall, F1, and support;
- confusion matrix.

## Re-evaluate a checkpoint

```bash
python training/evaluate_style_classifier.py \
  data/styles_prepared \
  models/style_classifier/efficientnet_b0.pth \
  --split test
```

The evaluator rejects a dataset whose fingerprint differs from the one stored
in the checkpoint. `--allow-dataset-mismatch` is reserved for deliberate
external evaluation and should not be used to hide accidental dataset drift.

## Vertex AI / Open Images V7

The Open Images managed container now enters through
`cloud/style_quality_vertex_job.py`. It reuses the existing licensed Open
Images V7 collection, furniture filtering, GCS resume behavior, and pinned
SigLIP weak-label model, then inserts the quality gate before training.

For each run the job publishes the raw source manifest plus:

- `quality/summary.json`;
- `quality/manifest.jsonl`;
- `quality/conflicts.json`;
- `quality/rejected.jsonl`;
- `models/style_classifier.pth`;
- `models/style_classifier.pth.metrics.json`;
- an updated `status.json` containing the quality/evaluation summary.

The existing launcher remains the operator entry point. Keeping the same
`--run-id` preserves the existing resumable GCS run behavior.

## Evaluation integrity

The automatic Open Images validation/test sets are derived from SigLIP
pseudo-labels. They are useful for detecting regressions and choosing among
models trained on the same labeling pipeline, but **they are not independent
human ground truth and must not be presented as production accuracy**.

Before a production quality claim or commercial model release, construct a
separate human-reviewed test set whose images are not present, or near-duplicate
with images present, in the training/validation data. Preserve its provenance
and freeze its dataset fingerprint before comparing candidate models.

## Release identity

A reproducible FurnitureAI style-model release should identify at least:

- Git commit SHA;
- dataset fingerprint;
- prepared manifest SHA-256;
- checkpoint SHA-256;
- model architecture and image size;
- validation and test metrics;
- source-license/provenance manifest;
- whether the reported test labels are pseudo-labels or human-reviewed ground
  truth.
