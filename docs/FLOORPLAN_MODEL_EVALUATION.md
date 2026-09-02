# Floor-plan Model Evaluation

## Goal

Add learned structural perception without replacing deterministic geometry. Candidate models may propose walls, doors, windows and room regions; the local geometry engine validates, reconciles and rejects unusable output.

## Candidate A — lightweight five-class U-Net

Source: `hallelu/floorplan-segmentation` on Hugging Face.

- reported license: MIT;
- architecture: lightweight U-Net;
- input: 224x224 RGB;
- output classes: background, walls, doors, windows, rooms;
- model card states training on CubiCasa5K.

This is a useful low-memory baseline. Reported metadata is not accepted as FurnitureAI accuracy until reproduced on our evaluation split.

## Candidate B — ResNet34 U-Net structural segmentation

Source: `Yytsi/floorplan-to-3d-walls` on Hugging Face.

- reported license: MIT;
- safetensors artifact;
- input pipeline described as 512x512 aspect-preserving letterbox;
- classes: floor, wall, door, window;
- model card reports validation mIoU 0.983 on CubiCasa5K.

The mIoU value is self-reported and must be independently reproduced before promotion.

## Dataset candidate — CubiCasa5K COCO derivative

Source: `phungpx/cubicassa5k-coco`.

The repository describes 4,976 floor-plan images with polygon annotations including rooms, walls, doors and windows. The derivative dataset and upstream CubiCasa5K terms must both be reviewed before production/commercial training. Dataset metadata alone is not sufficient authorization.

## Explicit exclusions

Do not promote a model when:

- license is `other`, absent or incompatible with intended use;
- inference requires `trust_remote_code=True` without a source/security audit;
- only a self-reported metric exists and no reproducible evaluation is available;
- the artifact cannot be SHA-256 pinned;
- training provenance is unclear;
- the model makes semantic or geometric claims outside its labeled classes.

## Evaluation protocol

Keep a held-out set containing multiple drawing styles and resolutions. Measure per class:

- IoU and Dice;
- precision and recall;
- door/window detection recall;
- topology break rate after polygonization;
- false exterior-room rate;
- latency and peak memory;
- deterministic post-processing failure/abstention rate.

Compare against the existing OpenCV baseline on the exact same cases. Save dataset revision, model revision, preprocessing settings, code commit and SHA-256 for every run.

## Promotion strategy

1. benchmark candidates offline;
2. add model behind an explicit feature flag;
3. reconcile model masks with deterministic geometry;
4. preserve OpenCV fallback when weights are absent or rejected;
5. add regression cases for ambiguous and out-of-distribution plans;
6. promote only after exact-commit CI, security and evaluation gates pass.
