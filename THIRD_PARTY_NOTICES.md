# Source provenance notice

This repository consolidates code and concepts from repositories owned by `lil-fahad`. Exact source commits and their roles are recorded in `PROVENANCE.json`. The final code is a rewritten unified implementation; obsolete repository structures, histories, generated assets, unrelated weights, and credentials are intentionally excluded.

## Optional Open Images training pipeline

The optional Vertex AI pipeline reads metadata and images from [Open Images
V7](https://storage.googleapis.com/openimages/web/download_v7.html). Source
images are not relicensed by this repository. The pipeline accepts only image
records declaring CC BY 2.0 or CC0 and preserves the per-image author, source
page, license URL, and retrieval metadata in its generated manifest. Users
remain responsible for reviewing that metadata and satisfying each image's
license terms.

Weak style labels are generated with
[`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224),
pinned to revision `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed` and distributed under
Apache-2.0. The labels are pseudo-labels and require human quality review.

The training image derives from NVIDIA PyTorch container `26.07-py3`. Use of
that container is governed by the NVIDIA container and software terms that
accompany it; the container itself is not redistributed in this repository.
