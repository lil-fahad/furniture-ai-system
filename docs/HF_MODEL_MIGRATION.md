# FurnitureAI Hugging Face model migration

This document defines the controlled migration from the recovered v0.4.x models to stronger pretrained backbones that can be fine-tuned on FurnitureAI data.

## Rules

1. Production candidates require an explicit commercial-friendly license. Apache-2.0 is preferred.
2. A newer model never replaces a working model by name alone; it must beat the current model on the same held-out dataset.
3. Training remains GPU-only and resumable through `training/local_worker.py`.
4. Preprocessing is part of the checkpoint contract. Mean/std, input size, crop percentage, and interpolation are read from the backbone data configuration and stored in checkpoints.
5. Pretrained download failures stop training by default. Random initialization requires the explicit `--allow-random-init` flag.

## Phase 1: style and room classification

Baseline candidate: `timm/convnext_base.fb_in22k_ft_in1k`, loaded through `hf_hub:timm/convnext_base.fb_in22k_ft_in1k`.

Why first:
- Apache-2.0 model card.
- Strong pretrained ConvNeXt Base image representation.
- Its ImageNet preprocessing is compatible with the previous classifier pipeline, making it a low-risk first migration.
- The new trainer is no longer tied to ImageNet preprocessing, so later backbones can supply their own data configuration.

Training outputs:
- `models/style_classifier/convnext_base_v2.pth`
- `models/room_classifier/convnext_base_v2.pth`

Promotion gate:
- Style: macro F1 must exceed the current production candidate on the same versioned test split.
- Room: validation/test accuracy must exceed the current production candidate on the same split contract.
- No promotion if calibration, class imbalance, or minority-class recall materially regresses.

## Phase 2: shared semantic visual encoder

Candidate: `timm/vit_base_patch16_siglip_224.v2_webli` (SigLIP2 image tower).

Target role:
- style classification
- room classification
- furniture classification
- reusable visual embeddings

SigLIP2 is not enabled in the autonomous queue yet. It will first be benchmarked through the new model-specific preprocessing path. If it wins, FurnitureAI can consolidate multiple visual classifiers around one shared encoder with task-specific heads.

## Detection

Production replacement candidate: `PekingU/rtdetr_v2_r50vd`.

Target: replace the fixed COCO-label DETR checkpoint with a detector fine-tuned on FurnitureAI's furniture taxonomy and room-specific objects.

Teacher candidate: `IDEA-Research/grounding-dino-base`.

Grounding DINO should be used primarily for open-vocabulary discovery and pseudo-label generation. Human/quality gates must validate pseudo-labels before they enter the trusted training set.

## Segmentation and depth

Keep `facebook/sam2.1-hiera-tiny` as the lightweight promptable segmentation runtime model until a benchmark proves a larger SAM2 model is worth the latency/VRAM cost.

Keep `depth-anything/Depth-Anything-V2-Small-hf` for relative depth. Do not call its output real-world metric dimensions without a calibration stage.

## Floor-plan segmentation

The current `SmallUNet` remains the production fallback for now.

SegFormer/Mask2Former floor-plan checkpoints discovered during research have unclear `other` licenses. Their pretrained weights are therefore blocked from the production pipeline until licensing is explicitly resolved. Architecture experiments may proceed separately, but replacement requires:
- a commercially safe training source,
- the same five-class mask contract,
- equal or better mean IoU and Dice on the FurnitureAI holdout,
- preserved resume/checkpoint behavior,
- export/runtime compatibility.

## Current recovered furniture classifier benchmark

The recovered EfficientNet-B0 furniture classifier remains the reference until the new furniture classifier is trained on the same 12 classes:

`bed, cabinet, chair, desk, dresser, lamp, ottoman, rug, shelf, sofa, stool, table`

Recovered v0.4.1 metrics recorded in the repository manifest:
- test accuracy: 0.77381
- macro F1: 0.742626

The next furniture classifier must exceed these metrics on a controlled, versioned benchmark before replacement.
