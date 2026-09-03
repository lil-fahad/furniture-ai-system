# Open Images V7 Furniture Benchmark

This benchmark uses real public Open Images V7 validation data for FurnitureAI object-detection evaluation.

## Provenance

- Dataset: Google Open Images V7
- Split: validation
- Target classes: Chair, Couch, Table
- Upstream dataset documentation: https://storage.googleapis.com/openimages/web/index.html
- Dataset license: CC BY 4.0 for Open Images annotations/content as documented by the upstream project; individual image licenses/attribution remain attached to source image metadata and must be preserved by the downloader.
- This repository does not commit third-party image bytes.

## Why this benchmark exists

The registered furniture-tuned DETR candidate exposes four labels (`furniture`, `Chair`, `Sofa`, `Table`) but smoke inference alone cannot establish detection quality. This benchmark adds a reproducible path to evaluate the candidate on real images and real human-verified bounding boxes rather than generated examples.

## Reproducibility rules

1. Fetch only validation rows that contain the requested target classes.
2. Keep the original Open Images image IDs, normalized bounding boxes, source URLs/licence metadata, and class IDs.
3. Materialize images into a local ignored benchmark directory; do not commit third-party images.
4. Run evaluation with a pinned model revision and record artifact SHA-256.
5. Report metrics without promoting a candidate to production automatically.

## Metrics

The evaluator is expected to report per-class AP, mAP@0.5, precision, recall, false positives, false negatives, latency p50/p95 and an error-analysis sample. A model remains evaluation-only until its measured quality is compared to the current baseline and explicitly approved.
