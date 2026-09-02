# Real furniture datasets

The project keeps large raw datasets outside Git history. GitHub rejects individual files larger than 100 MB, and committing thousands of training images would make every clone unnecessarily large. The repository stores the reproducible importer, source registry, license notes, and per-download provenance instead.

## Download the automatic real-image datasets

From the repository root:

```bash
python scripts/download_real_furniture_datasets.py --list
python scripts/download_real_furniture_datasets.py
```

This downloads and extracts:

- **HomeObjects-3K** — 2,689 real indoor images with YOLO detection labels.
- **Pix3D** — 10,069 real furniture images paired with masks, poses, and 395 3D shapes.
- **IKEA Dataset** — more than 12,600 real product images with product metadata and dimensions.

The default destination is `data/raw/real_furniture/`. Downloads are resumable, ZIP entries are checked against path traversal, and each completed dataset gets a `SOURCE.json` provenance record. Use `--keep-archives` if the original ZIPs must remain after extraction.

Download one source only:

```bash
python scripts/download_real_furniture_datasets.py --dataset homeobjects-3k
python scripts/download_real_furniture_datasets.py --dataset pix3d
python scripts/download_real_furniture_datasets.py --dataset ikea-products
```

## Large or terms-gated datasets

These sources are registered but are not silently downloaded:

- **Open Images V7 furniture subset** — use the existing managed Google Cloud pipeline:
  ```bash
  bash cloud/launch_gcp_training.sh --execute \
    --project round-office-505007-q4 \
    --max-images 100000
  ```
- **SUN RGB-D** — accept and follow the upstream research terms at <https://rgbd.cs.princeton.edu/>.
- **ObjectNet3D** — obtain images and annotations from <https://cvgl.stanford.edu/projects/objectnet3d/> and follow the original ImageNet/ShapeNet terms.

## Licensing

“Real” describes how the images were captured; it does not mean unrestricted commercial rights. Before production or commercial training:

1. Review each upstream license and source-image license.
2. Preserve attribution and `SOURCE.json` records.
3. Remove images whose terms do not cover the intended use.
4. Keep a human-reviewed validation and test set.
