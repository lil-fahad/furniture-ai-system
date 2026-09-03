# Detection Evaluation Lab

FurnitureAI must not promote a detector to Production based on model-card popularity,
raw confidence scores, or subjective visual inspection. Detection quality is measured
against a pinned, provenance-tracked benchmark before a model can be compared with the
current baseline.

## Benchmark source

The first supported benchmark builder consumes the Open Images bounding-box CSV format.
The intended public validation source is the Open Images V6 download surface; its
validation boxes use normalized coordinates and human-verified object annotations.
FurnitureAI does not silently download the annotation file. The operator supplies a local
CSV and may provide an expected SHA-256; the observed SHA-256 is always written into the
benchmark metadata.

The furniture subset currently covers the same Open Images MIDs used by the existing data
ingestion pipeline:

- Chair
- Table
- Sofa / Couch MID
- Bed
- Cabinetry
- Desk
- Shelf

By default, group-of boxes and depictions are excluded. The exact policy is recorded in
`metadata.json`. A benchmark generated with a different policy is a different benchmark
and must not be compared as though it were identical.

## Build a benchmark

```bash
python scripts/build_openimages_detection_eval.py \
  --annotations /data/openimages/validation-annotations-bbox.csv \
  --output artifacts/eval/openimages-furniture-v1 \
  --source-url https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv \
  --expected-sha256 <PINNED_SOURCE_SHA256>
```

The command writes:

- `ground_truth.jsonl`: deterministic normalized furniture boxes;
- `metadata.json`: source file name, URL, source SHA-256, generation timestamp, filtering
  policy, record/image counts, per-class counts, and skip counts.

## Evaluate predictions

Prediction JSONL records must contain:

```json
{"image_id":"...","label":"Chair","score":0.91,"box":{"x_min":0.1,"y_min":0.2,"x_max":0.4,"y_max":0.7}}
```

Run:

```bash
python scripts/evaluate_detection_predictions.py \
  --ground-truth artifacts/eval/openimages-furniture-v1/ground_truth.jsonl \
  --predictions artifacts/eval/detr/predictions.jsonl \
  --output artifacts/eval/detr/report.json \
  --model-id detr_resnet50 \
  --model-revision <PINNED_REVISION> \
  --model-sha256 <PINNED_MODEL_SHA256> \
  --iou-threshold 0.5
```

Every report records the model identity/revision/SHA-256 plus SHA-256 values for both the
ground-truth and prediction files. This prevents an evaluation number from becoming
detached from the exact model and data that produced it.

## Metrics

`furnitureai-detection-ap-v1` performs deterministic, one-to-one matching per class and
image at the requested IoU threshold. It reports per-class precision, recall, F1,
average precision, mean matched IoU, and micro/global aggregates including mAP across
classes that have ground truth.

This metric is deliberately named as a FurnitureAI metric. It is **not** presented as the
full official Open Images challenge evaluator: Open Images has dataset-specific semantics,
including group-of handling, that are outside this v1 metric. If official Open Images mAP
is later required, it should be added as a separate evaluator rather than silently changing
this metric's meaning.

## Promotion policy

A new detector such as an open-vocabulary Grounding DINO variant remains Experimental
until all of the following are true:

1. model source, revision, license, artifact size, and SHA-256 are pinned;
2. inference runs without runtime weight downloads;
3. the exact benchmark inputs are provenance-tracked;
4. baseline and candidate predictions are evaluated with the same metric and data;
5. per-class regressions are reviewed, not only aggregate mAP;
6. latency, memory, and GPU/CPU cost are measured on target hardware;
7. CI and security checks pass for the integration code.

No detector confidence score is treated as a system-level accuracy or trust score.
