# Models

## Room classifier

`training/train_room_classifier.py` fine-tunes EfficientNet-B0 on an ImageFolder dataset. The checkpoint includes class labels, architecture, validation accuracy, and deterministic seed metadata.

## Floor-plan segmenter

`training/train_floorplan_segmenter.py` trains a compact U-Net for five semantic classes and exports TorchScript. Expected mask labels should be documented by the dataset producer.

## OpenAI vision refinement

When `use_openai=true` and `OPENAI_API_KEY` is configured, the whole floor-plan image and geometric room candidates are sent through the Responses API to refine semantic room labels. The geometry engine remains authoritative for placement and circulation.

## Checkpoint policy

Weights are not committed by default. Put them at paths listed in `models/manifest.json`, run `python scripts/model_manifest.py --write`, and commit only the manifest checksum when storage is managed externally or through an approved artifact system.
