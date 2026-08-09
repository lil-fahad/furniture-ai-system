# Models

## Core room classifier

`training/train_room_classifier.py` fine-tunes EfficientNet-B0 on an ImageFolder dataset. Its checkpoint includes class labels, architecture, validation accuracy, and deterministic seed metadata. Pass `--no-pretrained` to skip the ImageNet weight download and train from random initialization (useful for offline CPU smoke runs).

## Floor-plan segmenter

`training/train_floorplan_segmenter.py` trains a compact U-Net for five semantic classes and exports TorchScript. Mask pixel values must be valid class indices (`0 <= label < classes`); training validates this and fails with a clear error otherwise. Masks stored in the common 0/255 convention can be remapped at load time with `--mask-remap`, and masks are always resized with nearest-neighbor interpolation so no phantom class ids are invented.

## Professional bundle

`models/professional/bundle.json` defines the repaired v0.4.1 archive, its SHA-256, and the exact files allowed to be extracted. The installer is:

```bash
python scripts/install_professional_bundle.py /path/to/bundle.zip
```

The registered external models include DETR ResNet-50, SAM 2.1 Hiera Tiny, Depth Anything V2 Small, and verified EfficientNet-B0 furniture checkpoints. Full details are in `docs/PROFESSIONAL_MODELS.md`.

## OpenAI vision refinement

When `use_openai=true` and `OPENAI_API_KEY` is configured, the whole floor-plan image and geometric room candidates are sent through the Responses API to refine semantic room labels. The geometry engine remains authoritative for placement and circulation.

## Checkpoint policy

Weights are not committed by default. Put locally trained weights at paths listed in `models/manifest.json`, or install the professional bundle through the verified installer. Commit only manifests, source metadata, checksums, and evaluation summaries unless an approved artifact service or Git LFS policy is in place.
