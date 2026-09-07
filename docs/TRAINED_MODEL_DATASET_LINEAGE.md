# Trained Model ↔ Dataset Lineage

FurnitureAI keeps training data, trained artifacts, and lineage metadata separate so the repository can answer **which verified model came from which dataset evidence** without placing large checkpoint binaries inside dataset directories.

## Source of truth

`data/model_dataset_registry.json` is the trained-model lineage registry. It records only artifacts that have verifiable training/evaluation metadata and integrity information.

The runtime inference registry remains `models/manifest.json`, mirrored byte-for-byte into `src/furniture_ai/resources/model_manifest.json` for standalone wheel installs. Dataset lineage is additive metadata; it does not change the eight-model runtime manifest contract.

## Storage policy

- Dataset records live under `data/` or point to externally managed datasets.
- Trained model artifacts live under `models/` or an external artifact store.
- Raw licensed image datasets and large checkpoints are not committed to ordinary Git history.
- Every registered trained artifact must have a SHA-256 digest and at least one dataset lineage identifier.
- Placeholder model definitions with no verified checkpoint are not presented as trained models.

The validator in `src/furniture_ai/model_dataset_registry.py` rejects unknown dataset references, duplicate identifiers, unsafe paths, malformed SHA-256 values, and model artifacts placed inside `data/`.

## Current verified lineage

| Trained model | Dataset lineage | Evidence |
| --- | --- | --- |
| `efficientnet_b0-efficientnet_b0_best` | `abo-starter-recovered` | Recovered professional-bundle checkpoint metadata, verified SHA-256, held-out test metrics |
| `efficientnet_b0_multiview-efficientnet_b0_best` | `abo-multiview-recovered` | Recovered professional-bundle checkpoint metadata, verified SHA-256, held-out test metrics |
| `efficientnet_b0_multiview-efficientnet_b0_last` | `abo-multiview-recovered` | Recovered professional-bundle checkpoint metadata and verified SHA-256; no confirmed test metrics for the last checkpoint |
| `supplier-ranker-ridge-v1` | `supplier-master-v1` | `models/supplier_ranker/metrics.json`, 41-record supplier snapshot, deterministic seed and validation metrics |

The recovered ABO records deliberately describe **confirmed evaluation lineage**. The original complete training/validation split provenance was not reconstructed, so the registry does not claim information that the recovered bundle cannot prove.

## Querying lineage

```python
from pathlib import Path

from furniture_ai.model_dataset_registry import ModelDatasetRegistry

registry = ModelDatasetRegistry.load(Path("data/model_dataset_registry.json"))

for model in registry.trained_models_for_dataset("abo-multiview-recovered"):
    print(model.id, model.sha256)

for dataset in registry.datasets_for_model("supplier-ranker-ridge-v1"):
    print(dataset.id, dataset.path)
```

Unknown model or dataset identifiers fail explicitly rather than returning ambiguous empty results.

## Registering future training runs

A future training pipeline should preserve enough metadata to reproduce and audit the relationship before its checkpoint is added to this registry:

1. Dataset identity and provenance, including authorization/license state where relevant.
2. Exact split or split-manifest fingerprint used for train/validation/test.
3. Random seed, preprocessing configuration, class mapping, and important hyperparameters.
4. Final artifact path, size, and SHA-256 digest.
5. Held-out evaluation metrics and the metrics-report path.
6. Known limitations, including synthetic-data use, small sample sizes, weak labels, or recovered/incomplete provenance.

Only after those facts are available should the model receive `status: "trained"` in `data/model_dataset_registry.json`. A model definition in `models/manifest.json` by itself is not evidence that training happened.

## Promotion rule

The lineage registry is provenance, not a quality endorsement. Production promotion still requires the relevant evaluation, licensing, security, runtime compatibility, and CI gates documented elsewhere in the repository.
