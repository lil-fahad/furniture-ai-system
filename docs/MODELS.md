# Models

## Core room classifier

`training/train_room_classifier.py` fine-tunes EfficientNet-B0 on an ImageFolder dataset. Its checkpoint includes class labels, architecture, validation accuracy, and deterministic seed metadata.

## Floor-plan segmenter

`training/train_floorplan_segmenter.py` trains a compact U-Net for five semantic classes and exports TorchScript. Expected mask labels must be documented by the dataset producer.

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


## Supplier suitability ranker

`training/train_supplier_ranker.py` trains an `ExtraTreesRegressor` on `data/suppliers_master.csv.gz.b64`. The committed model lives at `models/supplier_ranker/model.parts.json`; validation metrics and predictions are in `models/supplier_ranker/metrics.json` and `reports/`.

The ranker predicts the curated suitability score from structured supplier attributes. Runtime preference adjustments remain explicit and inspectable. It must not make autonomous purchasing or compliance decisions.
